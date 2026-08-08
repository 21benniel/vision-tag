"""Checkpoint and resume logic for labeling runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vision_tag.schema import LabelSchema


def is_successfully_labeled(row: dict, schema: LabelSchema) -> bool:
    if str(row.get("notes", "")).startswith("ERROR:"):
        return False

    for name, spec in schema.fields.items():
        value = row.get(name, "")
        if spec.type == "multi_choice":
            raw = value if isinstance(value, list) else _parse_json_list(str(value))
            if raw:
                return True
        elif spec.type == "single_choice" and value:
            return True
        elif spec.type == "text" and str(value).strip():
            return True
    return False


def _parse_json_list(value: str) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def load_checkpoint(path: Path, schema: LabelSchema) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if is_successfully_labeled(row, schema):
            done[row["image_name"]] = row
    return done


def load_labels_csv(path: Path, schema: LabelSchema) -> dict[str, dict]:
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if is_successfully_labeled(row, schema):
                done[row["image_name"]] = row
    return done


def load_completed(checkpoint_path: Path, output_path: Path, schema: LabelSchema) -> dict[str, dict]:
    completed = load_labels_csv(output_path, schema)
    completed.update(load_checkpoint(checkpoint_path, schema))
    return completed


def count_failed_in_checkpoint(checkpoint_path: Path, schema: LabelSchema) -> int:
    if not checkpoint_path.exists():
        return 0
    latest: dict[str, dict] = {}
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[row["image_name"]] = row
    return sum(1 for row in latest.values() if not is_successfully_labeled(row, schema))


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pending_images(images: list[Path], completed: dict[str, dict]) -> list[Path]:
    return [img for img in images if img.name not in completed]


def successful_rows_for_images(images: list[Path], completed: dict[str, dict], schema: LabelSchema) -> list[dict]:
    return [
        completed[img.name]
        for img in images
        if img.name in completed and is_successfully_labeled(completed[img.name], schema)
    ]
