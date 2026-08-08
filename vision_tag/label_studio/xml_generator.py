"""Generate Label Studio XML configuration from a label schema."""

from __future__ import annotations

import html

from vision_tag.schema import FieldSpec, LabelSchema


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _choices_block(field: FieldSpec, image_field: str) -> str:
    choice_type = "multiple" if field.type == "multi_choice" else "single"
    lines = [
        f'  <Header value="{_escape(field.label)}"/>',
        f'  <Choices name="{field.name}" toName="{image_field}" choice="{choice_type}" showInline="true">',
    ]
    for choice in field.choices:
        lines.append(f'    <Choice value="{_escape(choice)}"/>')
    lines.append("  </Choices>")
    return "\n".join(lines)


def _textarea_block(field: FieldSpec, image_field: str) -> str:
    placeholder = _escape(field.placeholder or "")
    return f"""  <Header value="{_escape(field.label)}"/>
  <TextArea
    name="{field.name}"
    toName="{image_field}"
    placeholder="{placeholder}"
    rows="{"5" if field.name == "meaning" else "3"}"
    maxSubmissions="1"
  />"""


def generate_label_studio_xml(schema: LabelSchema) -> str:
    image_field = schema.image_field
    parts = [
        "<View>",
        f'  <Image name="{image_field}" value="${image_field}" zoom="true" zoomControl="true"/>',
        "",
    ]

    for field in schema.fields.values():
        if field.type in {"multi_choice", "single_choice"}:
            parts.append(_choices_block(field, image_field))
        elif field.type == "text":
            parts.append(_textarea_block(field, image_field))
        parts.append("")

    parts.append("</View>")
    return "\n".join(parts)
