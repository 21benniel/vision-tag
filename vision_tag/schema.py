"""Load and validate label schema YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

FIELD_TYPES = {"multi_choice", "single_choice", "text"}


@dataclass
class FieldSpec:
    name: str
    type: str
    label: str = ""
    choices: list[str] = field(default_factory=list)
    max_items: int | None = None
    placeholder: str = ""


@dataclass
class LabelSchema:
    name: str
    description: str
    image_field: str
    fields: dict[str, FieldSpec]

    def choice_field_names(self) -> list[str]:
        return [name for name, spec in self.fields.items() if spec.type in {"multi_choice", "single_choice"}]

    def text_field_names(self) -> list[str]:
        return [name for name, spec in self.fields.items() if spec.type == "text"]

    def json_schema_example(self) -> dict[str, Any]:
        example: dict[str, Any] = {}
        for name, spec in self.fields.items():
            if spec.type == "multi_choice":
                example[name] = ["..."]
            elif spec.type == "single_choice":
                example[name] = "..."
            else:
                example[name] = "..."
        return example


def load_schema(path: Path) -> LabelSchema:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid schema file: {path}")

    fields_raw = data.get("fields", {})
    if not fields_raw:
        raise ValueError("Schema must define at least one field under 'fields'")

    fields: dict[str, FieldSpec] = {}
    for name, spec in fields_raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Field '{name}' must be a mapping")
        field_type = spec.get("type", "")
        if field_type not in FIELD_TYPES:
            raise ValueError(f"Field '{name}' has unsupported type: {field_type}")

        choices = list(spec.get("choices", []) or [])
        if field_type in {"multi_choice", "single_choice"} and not choices:
            raise ValueError(f"Field '{name}' requires non-empty choices")

        fields[name] = FieldSpec(
            name=name,
            type=field_type,
            label=str(spec.get("label", name.replace("_", " ").title())),
            choices=[str(c) for c in choices],
            max_items=spec.get("max_items"),
            placeholder=str(spec.get("placeholder", "")),
        )

    return LabelSchema(
        name=str(data.get("name", "labels")),
        description=str(data.get("description", "Label images for AI training.")),
        image_field=str(data.get("image_field", "image")),
        fields=fields,
    )


def csv_fieldnames(schema: LabelSchema) -> list[str]:
    names = ["image_name"]
    names.extend(schema.fields.keys())
    names.extend(["input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"])
    return names
