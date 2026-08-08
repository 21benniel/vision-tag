"""Build Vertex AI prompts from a label schema."""

from __future__ import annotations

import json

from vision_tag.schema import LabelSchema


def build_prompt(schema: LabelSchema) -> str:
    example = json.dumps(schema.json_schema_example(), indent=2)
    rules: list[str] = [
        "Return ONLY valid JSON (no markdown).",
        "Pick values ONLY from the allowed lists below.",
    ]

    for name, spec in schema.fields.items():
        if spec.type == "multi_choice":
            limit = f"choose 0-{spec.max_items}" if spec.max_items else "choose relevant labels"
            rules.append(f"- {name}: {limit} from allowed list")
        elif spec.type == "single_choice":
            rules.append(f"- {name}: exactly one value")
        else:
            rules.append(f"- {name}: short text; use \"\" if none")

    allowed_blocks: list[str] = []
    for name, spec in schema.fields.items():
        if spec.choices:
            allowed_blocks.append(f"Allowed {name}: {json.dumps(spec.choices)}")

    return f"""You are labeling images for: {schema.description}

For the image, return ONLY valid JSON with this schema:
{example}

Rules:
{chr(10).join(rules)}

{chr(10).join(allowed_blocks)}
"""


def empty_labels(schema: LabelSchema) -> dict:
    labels: dict = {}
    for name, spec in schema.fields.items():
        if spec.type == "multi_choice":
            labels[name] = []
        else:
            labels[name] = ""
    return labels
