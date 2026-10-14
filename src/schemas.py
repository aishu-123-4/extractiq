"""Extraction schemas (JSON Schema) and validation.

A schema defines the target structure for extraction. ExtractIQ is
schema-driven: add a schema here and both the LLM and baseline extractors can
target it. The bundled `contact` schema pulls structured contact details out of
free-form text (email signatures, bios, directory blurbs).
"""

from __future__ import annotations

from jsonschema import Draft7Validator

CONTACT_SCHEMA = {
    "type": "object",
    "title": "contact",
    "properties": {
        "name": {"type": ["string", "null"]},
        "email": {"type": ["string", "null"]},
        "phone": {"type": ["string", "null"]},
        "company": {"type": ["string", "null"]},
        "title": {"type": ["string", "null"]},
    },
    "required": ["name", "email", "phone", "company", "title"],
    "additionalProperties": False,
}

SCHEMAS = {"contact": CONTACT_SCHEMA}


def get_schema(name: str) -> dict:
    if name not in SCHEMAS:
        raise KeyError(f"Unknown schema '{name}'. Available: {list(SCHEMAS)}")
    return SCHEMAS[name]


def validate(data: dict, schema: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    validator = Draft7Validator(schema)
    return [e.message for e in validator.iter_errors(data)]


def fields_of(schema: dict) -> list[str]:
    return list(schema.get("properties", {}).keys())
