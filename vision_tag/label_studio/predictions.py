"""Build Label Studio prediction JSON from CSV rows."""

from __future__ import annotations

import json
from typing import Any

from vision_tag.schema import LabelSchema

MODEL_VERSION = "visiontag-vertex-ai"


def parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]


def _choices(from_name: str, to_name: str, values: list[str]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "from_name": from_name,
        "to_name": to_name,
        "type": "choices",
        "value": {"choices": values},
    }


def _textarea(from_name: str, to_name: str, text: str) -> dict[str, Any] | None:
    if not text:
        return None
    return {
        "from_name": from_name,
        "to_name": to_name,
        "type": "textarea",
        "value": {"text": [text]},
    }


def row_to_prediction_result(row: dict[str, str], schema: LabelSchema) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    image_field = schema.image_field

    for name, spec in schema.fields.items():
        if spec.type == "multi_choice":
            item = _choices(name, image_field, parse_json_list(row.get(name, "")))
        elif spec.type == "single_choice":
            value = row.get(name, "")
            item = _choices(name, image_field, [value] if value else [])
        else:
            item = _textarea(name, image_field, str(row.get(name, "")).strip())
        if item is not None:
            result.append(item)

    return result


def row_to_prediction(row: dict[str, str], schema: LabelSchema) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "score": 0.85,
        "result": row_to_prediction_result(row, schema),
    }


def is_failed_row(row: dict[str, str] | None) -> bool:
    return bool(row and str(row.get("notes", "")).startswith("ERROR:"))
