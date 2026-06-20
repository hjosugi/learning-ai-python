"""Tool definitions and local tool execution.

A "tool" here is exactly the shape the real Anthropic Messages API expects in the
`tools=[...]` request field:

    {"name": ..., "description": ..., "input_schema": {JSON Schema}}

The model never runs your code. It only emits a `tool_use` block naming a tool
and an `input` dict; YOUR harness looks the name up in a registry, runs the
matching Python function, and feeds a `tool_result` block back. Keeping tool
definitions as plain dicts means the exact same list works against FakeTransport
here and against a real `anthropic.Anthropic()` client later (see README upgrade
path) with zero changes.
"""

from __future__ import annotations

from typing import Any, Callable

# ---------------------------------------------------------------------------
# Tool schemas: JSON-schema dicts, identical to the real Messages API `tools`.
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns a short text summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Tokyo"}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations. Deterministic so tests and evals never flake.
# These run on YOUR machine, not on the model. A real harness would gate
# side-effecting tools (sending email, deleting data) behind confirmation.
# ---------------------------------------------------------------------------

# A tiny fixed "database" so get_weather is deterministic and offline.
_WEATHER = {
    "tokyo": "18C and clear",
    "london": "11C and rainy",
    "san francisco": "15C and foggy",
}


def _get_weather(city: str) -> str:
    return _WEATHER.get(city.strip().lower(), f"no data for {city}")


def _add(a: int, b: int) -> int:
    return int(a) + int(b)


# Registry mapping tool name -> python callable. The agent loop dispatches on
# this; the model only ever sees the schemas in TOOLS above.
TOOL_IMPLS: dict[str, Callable[..., Any]] = {
    "get_weather": _get_weather,
    "add": _add,
}


def run_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool by name with the model-supplied input.

    Returns (content, is_error). On any failure we return is_error=True with a
    readable message instead of raising, mirroring how a tool_result block can
    carry `"is_error": true` so the model can recover on the next turn.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return (f"unknown tool: {name}", True)
    try:
        # **tool_input keys must match the JSON schema property names.
        return (str(impl(**tool_input)), False)
    except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
        return (f"tool '{name}' failed: {exc}", True)
