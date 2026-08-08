"""Smoke tests for Vertex AI prompt generation."""

import json

from vision_tag.env import default_schema_path
from vision_tag.prompt import build_prompt
from vision_tag.schema import load_schema


def test_prompt_includes_all_choice_values():
    schema = load_schema(default_schema_path())
    prompt = build_prompt(schema)

    for name, spec in schema.fields.items():
        if spec.type in {"multi_choice", "single_choice"}:
            for choice in spec.choices[:3]:
                assert choice in prompt, f"Missing choice '{choice}' for field '{name}'"


def test_prompt_includes_schema_description():
    schema = load_schema(default_schema_path())
    prompt = build_prompt(schema)
    assert schema.description in prompt


def test_prompt_requests_json():
    schema = load_schema(default_schema_path())
    prompt = build_prompt(schema)
    assert "json" in prompt.lower()
    example = schema.json_schema_example()
    assert json.dumps(example)[:20] in prompt or "reactions" in prompt
