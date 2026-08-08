"""Smoke tests for schema loading and Label Studio XML generation."""

from pathlib import Path

from vision_tag.env import default_schema_path, package_root
from vision_tag.label_studio.xml_generator import generate_label_studio_xml
from vision_tag.schema import load_schema


def test_load_default_schema():
    schema = load_schema(default_schema_path())
    assert schema.name == "meme-reactions"
    assert schema.image_field == "image"
    assert "reactions" in schema.fields
    assert "sentiment" in schema.fields
    assert len(schema.fields["reactions"].choices) > 10


def test_load_minimal_schema():
    path = package_root() / "config" / "label_schema.minimal.yaml"
    schema = load_schema(path)
    assert len(schema.fields) >= 2


def test_generate_xml_from_default_schema():
    schema = load_schema(default_schema_path())
    xml = generate_label_studio_xml(schema)
    assert xml.startswith("<View>")
    assert xml.endswith("</View>")
    assert f'name="{schema.image_field}"' in xml
    for name in schema.choice_field_names():
        assert f'name="{name}"' in xml
    for name in schema.text_field_names():
        assert f'name="{name}"' in xml


def test_csv_fieldnames_include_tokens():
    from vision_tag.schema import csv_fieldnames

    schema = load_schema(default_schema_path())
    names = csv_fieldnames(schema)
    assert names[0] == "image_name"
    assert "estimated_cost_usd" in names
