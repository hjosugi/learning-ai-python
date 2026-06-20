"""The agent loop: the Anthropic Messages-API tool-calling pattern.

This is the heart of the lab and the part you would keep UNCHANGED when swapping
FakeTransport for a real `anthropic` client. The loop:

    1. Call the transport with system + tools + the running message history.
    2. Append the assistant turn (verbatim) to history.
    3. If stop_reason == "end_turn": done, return the final text.
    4. If stop_reason == "tool_use": execute every tool_use block, append ONE
       user message containing all the tool_result blocks, and loop.

That four-step shape is identical to the real API's agentic loop (see the
"Manual Agentic Loop" pattern in the Anthropic docs). Two real-world details are
preserved here:

  * The assistant `content` is appended verbatim (tool_use blocks must survive
    the round-trip or the next request 400s).
  * All tool_result blocks for one assistant turn go in a SINGLE user message
    (splitting them trains the model to stop making parallel calls).

It also wires in retry-with-backoff and a timeout around each transport call,
simulates token streaming, and validates the final structured output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from schema import SchemaError, validate
from tools import TOOLS, run_tool
from transport import Transport, call_with_retry, with_timeout

SYSTEM_PROMPT = (
    "You are a precise assistant. Use the provided tools when they help, "
    "then give a short final answer."
)

MODEL = "fake-claude"  # swap for e.g. "claude-opus-4-8" with a real transport.


@dataclass
class AgentRun:
    """Everything that happened during one run, for tests and inspection."""

    final_text: str
    messages: list[dict[str, Any]]
    turns: int
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    stop_reason: str = ""


def run_agent(
    user_message: str,
    transport: Transport,
    *,
    max_turns: int = 8,
    timeout: float = 5.0,
    max_retries: int = 4,
) -> AgentRun:
    """Drive the tool-calling loop to completion against any Transport.

    Identical against FakeTransport and a real Anthropic client adapter.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    last_stop = ""

    for turn in range(1, max_turns + 1):
        # One transport call, wrapped in timeout + retry. The lambda is what
        # gets retried; with a real client this is client.messages.create(...).
        def _do_call() -> dict[str, Any]:
            return with_timeout(
                lambda: transport.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                ),
                timeout=timeout,
            )

        response = call_with_retry(_do_call, max_retries=max_retries)
        last_stop = response["stop_reason"]

        # Append the assistant turn verbatim (tool_use blocks must survive).
        messages.append({"role": "assistant", "content": response["content"]})

        if last_stop == "end_turn":
            return AgentRun(
                final_text=_first_text(response["content"]),
                messages=messages,
                turns=turn,
                tool_calls=tool_calls,
                stop_reason=last_stop,
            )

        if last_stop == "tool_use":
            tool_results: list[dict[str, Any]] = []
            for block in response["content"]:
                if block.get("type") != "tool_use":
                    continue
                name = block["name"]
                tool_input = block["input"]
                tool_calls.append((name, tool_input))
                content, is_error = run_tool(name, tool_input)
                # tool_result must reference the originating tool_use id.
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": content,
                        "is_error": is_error,
                    }
                )
            # ALL results for this turn go in ONE user message.
            messages.append({"role": "user", "content": tool_results})
            continue

        raise RuntimeError(f"unexpected stop_reason: {last_stop!r}")

    raise RuntimeError(f"agent did not finish within {max_turns} turns")


def _first_text(content: list[dict[str, Any]]) -> str:
    for block in content:
        if block.get("type") == "text":
            return block["text"]
    return ""


# ---------------------------------------------------------------------------
# Streaming simulation: yield the final answer in chunks, like text_stream.
# ---------------------------------------------------------------------------

def stream_text(text: str, *, chunk_size: int = 8) -> Iterator[str]:
    """Yield `text` in fixed-size chunks.

    The real SDK exposes `with client.messages.stream(...) as s: for t in
    s.text_stream`. The contract a caller cares about is "text arrives in
    pieces"; we reproduce that offline so streaming-aware UI code can be written
    and tested against the same iterator shape.
    """
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


# ---------------------------------------------------------------------------
# Structured output: ask for, then VALIDATE, a JSON object.
# ---------------------------------------------------------------------------

# The schema the final structured answer must satisfy. With a real client you
# would pass this via output_config.format so the API constrains generation;
# here we parse + validate the model's JSON ourselves to teach the check.
FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "used_tool": {"type": "boolean"},
        "tool_names": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "used_tool", "tool_names"],
    "additionalProperties": False,
}


def structured_result(run: AgentRun) -> dict[str, Any]:
    """Build the structured object from a run and validate it against the schema.

    Returns the validated dict. Raises SchemaError if it does not conform — the
    same guarantee `output_config.format` gives you with the real API, enforced
    locally here so the failure mode is visible.
    """
    obj = {
        "answer": run.final_text,
        "used_tool": len(run.tool_calls) > 0,
        "tool_names": [name for name, _ in run.tool_calls],
    }
    validate(obj, FINAL_OUTPUT_SCHEMA)  # raises SchemaError on mismatch
    return obj


# ---------------------------------------------------------------------------
# Demo entry point.
# ---------------------------------------------------------------------------

def _demo(transport_factory: Callable[[], Transport] | None = None) -> None:
    from transport import FakeTransport  # local import keeps the seam obvious

    factory = transport_factory or (lambda: FakeTransport(fail_times=1))

    examples = [
        "What is the weather in Tokyo?",
        "What is 21 + 21?",
        "Say hello.",
    ]
    for question in examples:
        run = run_agent(question, factory())
        print(f"\nQ: {question}")
        # Demonstrate streaming the final answer.
        print("A (streamed): ", end="", flush=True)
        for chunk in stream_text(run.final_text):
            print(chunk, end="", flush=True)
        print()
        print(f"   turns={run.turns} tools={[n for n, _ in run.tool_calls]}")
        print("   structured: " + json.dumps(structured_result(run), ensure_ascii=False))


if __name__ == "__main__":
    _demo()
