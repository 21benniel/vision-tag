"""CSV export utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vision_tag.schema import LabelSchema, csv_fieldnames


def serialize_labels_for_csv(labels: dict, schema: LabelSchema) -> dict:
    row: dict = {}
    for name, spec in schema.fields.items():
        value = labels.get(name, [] if spec.type == "multi_choice" else "")
        if spec.type == "multi_choice":
            row[name] = json.dumps(value if isinstance(value, list) else [])
        else:
            row[name] = str(value) if value is not None else ""
    return row


def row_to_csv_fields(
    image_name: str,
    labels: dict,
    schema: LabelSchema,
    usage: dict | None = None,
    estimated_cost: float | None = None,
) -> dict:
    row = {"image_name": image_name}
    row.update(serialize_labels_for_csv(labels, schema))
    if usage:
        row["input_tokens"] = usage.get("input_tokens", 0)
        row["output_tokens"] = usage.get("output_tokens", 0)
        row["total_tokens"] = usage.get("total_tokens", 0)
    if estimated_cost is not None:
        row["estimated_cost_usd"] = estimated_cost
    return row


def write_csv(rows: list[dict], output_path: Path, schema: LabelSchema) -> None:
    fieldnames = csv_fieldnames(schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
