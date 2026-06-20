"""A tiny, dependency-free JSON Schema validator.

This is NOT a full JSON Schema implementation - it covers the subset used by the
lab's tool input schemas and the final-output schema: object/array/string/
integer/number/boolean/null types, `properties`, `required`,
`additionalProperties: false`, and `items`. That subset is also the subset the
Anthropic structured-outputs feature supports for `output_config.format`.

In a real app you would let the API enforce the schema (pass it via
output_config.format / strict tool use) or use the `jsonschema` package. This
local validator exists so the "validate the final structured output" step works
with the standard library only, and so you can see exactly what "conforms to the
schema" means.
"""

from __future__ import annotations

from typing import Any

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python; exclude it from integer/number.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


class SchemaError(ValueError):
    """Raised when a value does not conform to the schema."""


def validate(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate `value` against `schema`. Raises SchemaError on the first
    violation; returns None on success."""
    expected = schema.get("type")
    if expected is not None:
        check = _TYPE_CHECKS.get(expected)
        if check is None:
            raise SchemaError(f"{path}: unsupported schema type {expected!r}")
        if not check(value):
            raise SchemaError(f"{path}: expected {expected}, got {type(value).__name__}")

    if expected == "object":
        _validate_object(value, schema, path)
    elif expected == "array":
        _validate_array(value, schema, path)


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in value:
            raise SchemaError(f"{path}: missing required property {key!r}")

    if schema.get("additionalProperties") is False:
        extra = set(value) - set(properties)
        if extra:
            raise SchemaError(f"{path}: unexpected properties {sorted(extra)}")

    for key, subschema in properties.items():
        if key in value:
            validate(value[key], subschema, path=f"{path}.{key}")


def _validate_array(value: list[Any], schema: dict[str, Any], path: str) -> None:
    items_schema = schema.get("items")
    if items_schema is None:
        return
    for i, item in enumerate(value):
        validate(item, items_schema, path=f"{path}[{i}]")
