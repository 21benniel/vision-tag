"""Build Label Studio import JSON with AI predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import quote

from vision_tag.label_studio.predictions import is_failed_row, row_to_prediction
from vision_tag.schema import LabelSchema
from vision_tag.vertex import IMAGE_EXTENSIONS, collect_images


def build_image_url(image_name: str, images_dir: Path, project_root: Path) -> str:
    try:
        relative = (images_dir / image_name).resolve().relative_to(project_root.resolve())
    except ValueError:
        relative = Path(images_dir.name) / image_name
    relative_str = str(relative).replace("\\", "/")
    return f"/data/local-files/?d={quote(relative_str)}"


def load_labels(csv_path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["image_name"]] = row
    return labels


def build_task(
    image_name: str,
    row: dict[str, str] | None,
    schema: LabelSchema,
    images_dir: Path,
    project_root: Path,
) -> dict:
    task = {
        "data": {
            schema.image_field: build_image_url(image_name, images_dir, project_root),
            "image_name": image_name,
        }
    }
    if row and not is_failed_row(row):
        prediction = row_to_prediction(row, schema)
        if prediction["result"]:
            task["predictions"] = [prediction]
    return task


def build_import_json(
    labels_csv: Path,
    output_path: Path,
    schema: LabelSchema,
    images_dir: Path,
    project_root: Path,
    include_unlabeled: bool = False,
) -> tuple[int, int]:
    labels = load_labels(labels_csv)
    image_names = sorted(labels.keys(), key=str.lower)

    if include_unlabeled:
        all_names = sorted([p.name for p in collect_images(images_dir)], key=str.lower)
        image_names = all_names

    tasks = [
        build_task(name, labels.get(name), schema, images_dir, project_root)
        for name in image_names
    ]
    labeled_count = sum(1 for t in tasks if t.get("predictions"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    return len(tasks), labeled_count
