"""Vertex AI Gemini image labeling."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from vision_tag.checkpoint import (
    append_checkpoint,
    count_failed_in_checkpoint,
    is_successfully_labeled,
    load_completed,
    pending_images,
    successful_rows_for_images,
)
from vision_tag.export import row_to_csv_fields, write_csv
from vision_tag.prompt import build_prompt, empty_labels
from vision_tag.schema import LabelSchema

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

INPUT_COST_PER_M = float(os.environ.get("VERTEX_INPUT_COST_PER_M", "0.15"))
OUTPUT_COST_PER_M = float(os.environ.get("VERTEX_OUTPUT_COST_PER_M", "0.60"))


@dataclass
class RunConfig:
    images_dir: Path
    output_path: Path
    checkpoint_path: Path
    schema: LabelSchema
    project: str
    location: str
    model: str
    limit: int | None = None
    sleep: float = 1.0
    retry_passes: int = 5
    retry_sleep: float = 45.0
    retry_attempts: int = 4


def collect_images(images_dir: Path) -> list[Path]:
    files = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def extract_usage(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost_usd(usage: dict) -> float:
    input_cost = usage["input_tokens"] * INPUT_COST_PER_M / 1_000_000
    output_cost = usage["output_tokens"] * OUTPUT_COST_PER_M / 1_000_000
    return round(input_cost + output_cost, 6)


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def filter_list(values: list, allowed: list[str], max_items: int | None = None) -> list[str]:
    allowed_set = set(allowed)
    cleaned: list[str] = []
    for value in values:
        if isinstance(value, str) and value in allowed_set and value not in cleaned:
            cleaned.append(value)
        if max_items and len(cleaned) >= max_items:
            break
    return cleaned


def normalize_label(raw: dict, schema: LabelSchema) -> dict:
    labels = empty_labels(schema)
    for name, spec in schema.fields.items():
        value = raw.get(name, labels[name])
        if spec.type == "multi_choice":
            items = value if isinstance(value, list) else []
            labels[name] = filter_list(items, spec.choices, spec.max_items)
        elif spec.type == "single_choice":
            choice = str(value).strip() if value is not None else ""
            labels[name] = choice if choice in spec.choices else ""
        else:
            labels[name] = str(value).strip() if value is not None else ""
    return labels


def check_credentials() -> None:
    try:
        import google.auth

        google.auth.default()
    except Exception as exc:
        raise RuntimeError(
            "GCP credentials not found. Run:\n"
            "  gcloud auth application-default login\n"
            "  gcloud config set project YOUR_PROJECT_ID\n"
            "Or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key."
        ) from exc


def init_vertex(project: str, location: str, model_name: str):
    import vertexai
    from vertexai.generative_models import GenerationConfig, GenerativeModel, Part

    check_credentials()
    vertexai.init(project=project, location=location)
    model = GenerativeModel(model_name)
    generation_config = GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )
    return model, Part, generation_config


def label_image(model, Part, generation_config, image_path: Path, prompt: str, schema: LabelSchema) -> tuple[dict, dict]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"
    image_part = Part.from_data(data=image_path.read_bytes(), mime_type=mime_type)
    response = model.generate_content(
        [image_part, prompt],
        generation_config=generation_config,
    )
    usage = extract_usage(response)
    raw = parse_json_response(response.text)
    return normalize_label(raw, schema), usage


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "Resource exhausted" in message


def label_image_with_retry(
    model,
    Part,
    generation_config,
    image_path: Path,
    prompt: str,
    schema: LabelSchema,
    *,
    max_attempts: int = 4,
    retry_base_sleep: float = 45.0,
) -> tuple[dict, dict]:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return label_image(model, Part, generation_config, image_path, prompt, schema)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1 or not is_rate_limit_error(exc):
                raise
            delay = retry_base_sleep * (2**attempt)
            print(
                f"  Rate limited (429), waiting {delay:.0f}s before retry {attempt + 2}/{max_attempts}...",
                flush=True,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def process_one_image(
    image_path: Path,
    *,
    model,
    Part,
    generation_config,
    prompt: str,
    schema: LabelSchema,
    checkpoint_path: Path,
    completed: dict[str, dict],
    max_attempts: int,
    retry_base_sleep: float,
) -> bool:
    image_name = image_path.name
    print(f"Labeling {image_name} ...", flush=True)
    try:
        labels, usage = label_image_with_retry(
            model,
            Part,
            generation_config,
            image_path,
            prompt,
            schema,
            max_attempts=max_attempts,
            retry_base_sleep=retry_base_sleep,
        )
        cost = estimate_cost_usd(usage)
        row = row_to_csv_fields(image_name, labels, schema, usage, cost)
        append_checkpoint(checkpoint_path, row)
        completed[image_name] = row
        print(
            f"  -> tokens={usage['total_tokens']} "
            f"(in={usage['input_tokens']}, out={usage['output_tokens']}), "
            f"est. ${cost:.6f}"
        )
        return True
    except Exception as exc:
        print(f"  !! Failed: {exc}", file=sys.stderr)
        error_labels = empty_labels(schema)
        error_labels["notes"] = f"ERROR: {exc}"
        row = row_to_csv_fields(image_name, error_labels, schema)
        append_checkpoint(checkpoint_path, row)
        return False


def run_labeling(config: RunConfig) -> int:
    if not config.project:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required")

    if not config.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {config.images_dir}")

    images = collect_images(config.images_dir)
    if config.limit:
        images = images[: config.limit]

    completed = load_completed(config.checkpoint_path, config.output_path, config.schema)
    pending = pending_images(images, completed)
    failed_existing = count_failed_in_checkpoint(config.checkpoint_path, config.schema)
    prompt = build_prompt(config.schema)

    print(f"Project: {config.project}")
    print(f"Location: {config.location}")
    print(f"Model: {config.model}")
    print(f"Schema: {config.schema.name}")
    print(f"Images in batch: {len(images)}")
    print(f"Already labeled (skipping): {len(images) - len(pending)}")
    print(f"Failed from earlier runs (will retry): {failed_existing}")
    print(f"To label now: {len(pending)}")
    print()

    if not pending:
        rows = successful_rows_for_images(images, completed, config.schema)
        write_csv(rows, config.output_path, config.schema)
        print("Nothing to do — all images in this batch are already labeled.")
        print(f"Wrote {len(rows)} row(s) to {config.output_path}")
        return 0

    model, Part, generation_config = init_vertex(config.project, config.location, config.model)
    failed_queue: list[Path] = []

    for index, image_path in enumerate(images, start=1):
        if image_path.name in completed:
            print(f"[{index}/{len(images)}] Skipping {image_path.name} (already labeled)", flush=True)
            continue

        print(f"[{index}/{len(images)}]", end=" ", flush=True)
        ok = process_one_image(
            image_path,
            model=model,
            Part=Part,
            generation_config=generation_config,
            prompt=prompt,
            schema=config.schema,
            checkpoint_path=config.checkpoint_path,
            completed=completed,
            max_attempts=2,
            retry_base_sleep=config.retry_sleep,
        )
        if not ok:
            failed_queue.append(image_path)
        if config.sleep > 0:
            time.sleep(config.sleep)

    for pass_num in range(1, config.retry_passes + 1):
        if not failed_queue:
            break
        wait = config.retry_sleep * pass_num
        print(
            f"\n--- Retry pass {pass_num}/{config.retry_passes}: "
            f"{len(failed_queue)} failed image(s), waiting {wait:.0f}s ---",
            flush=True,
        )
        time.sleep(wait)
        still_failed: list[Path] = []
        for image_path in failed_queue:
            ok = process_one_image(
                image_path,
                model=model,
                Part=Part,
                generation_config=generation_config,
                prompt=prompt,
                schema=config.schema,
                checkpoint_path=config.checkpoint_path,
                completed=completed,
                max_attempts=config.retry_attempts,
                retry_base_sleep=config.retry_sleep,
            )
            if not ok:
                still_failed.append(image_path)
            if config.sleep > 0:
                time.sleep(config.sleep)
        failed_queue = still_failed

    rows = successful_rows_for_images(images, completed, config.schema)
    missing = [
        img.name
        for img in images
        if img.name not in completed or not is_successfully_labeled(completed[img.name], config.schema)
    ]

    write_csv(rows, config.output_path, config.schema)
    print()
    print(f"Wrote {len(rows)} successful row(s) to {config.output_path}")

    if missing:
        print(f"\nStill failed after retries: {len(missing)} image(s)", file=sys.stderr)
        for name in missing[:10]:
            print(f"  - {name}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        print("\nRe-run the same command later to retry remaining failures.", file=sys.stderr)
        return 1

    if rows:
        batch_cost = sum(float(r.get("estimated_cost_usd") or 0) for r in rows)
        print(f"Estimated batch cost: ${batch_cost:.4f} USD")
    return 0
