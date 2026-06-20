"""Transport layer: the seam where a real Anthropic client would plug in.

`Transport` is the interface the agent loop talks to. It has ONE method,
`create(...)`, that takes Messages-API-shaped arguments and returns a
Messages-API-shaped response dict. The agent loop (agent.py) never imports
`anthropic` and never branches on which transport it has — it just calls
`transport.create(...)`. That is the whole point: swapping FakeTransport for a
real client (see README "Upgrade path") requires NO change to the loop.

`FakeTransport` is a deterministic stand-in that runs with no credentials and no
network. Given the conversation so far, it decides what the assistant "would"
say and returns the same content-block shape the real API uses:

    {
        "role": "assistant",
        "model": "fake-claude",
        "stop_reason": "tool_use" | "end_turn",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}},
        ],
        "usage": {"input_tokens": N, "output_tokens": M},
    }

The block types (`text`, `tool_use`), the `tool_use` shape (`id`/`name`/`input`),
and the `stop_reason` values (`tool_use`, `end_turn`) are exactly what
`POST /v1/messages` returns, so the agent loop is written against the real
contract, not a toy one.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Protocol


class TransportError(Exception):
    """A retryable transient failure (stands in for HTTP 429 / 5xx / timeout)."""


class TransportTimeout(TransportError):
    """The transport exceeded its per-call deadline."""


class Transport(Protocol):
    """The seam. A real `anthropic.Anthropic().messages` satisfies a thin
    adapter with this same shape — see README upgrade path for the adapter."""

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# FakeTransport: deterministic, offline, mirrors the Messages response shape.
# ---------------------------------------------------------------------------

_TOOLU_COUNTER = {"n": 0}


def _next_tool_use_id() -> str:
    _TOOLU_COUNTER["n"] += 1
    return f"toolu_fake_{_TOOLU_COUNTER['n']:04d}"


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Pull the most recent plain-text user message (the original question)."""
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if isinstance(content, str):
            return content
        # A list-form user turn may carry tool_result blocks (not text) - skip.
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
    return ""


def _has_tool_result(messages: list[dict[str, Any]]) -> bool:
    """True once the harness has appended at least one tool_result block,
    i.e. the model has already been given a tool's output to read."""
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False


class FakeTransport:
    """Deterministic Messages-API-shaped transport with NO credentials.

    It plans tool calls by simple keyword routing on the user's question. After
    a tool_result has come back, it emits a final `end_turn` text answer. This is
    the same control flow a real model drives via stop_reason; we just make the
    decisions hard-coded so tests and evals are reproducible.

    `fail_times` simulates transient failures: the first `fail_times` calls raise
    TransportError, after which calls succeed. This lets the agent's retry logic
    be exercised offline (see agent.call_with_retry).
    """

    def __init__(self, fail_times: int = 0, latency: float = 0.0) -> None:
        self._remaining_failures = fail_times
        self._latency = latency

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Simulate network latency (used by the timeout demo/tests).
        if self._latency:
            time.sleep(self._latency)

        # Simulate a transient failure that retry-with-backoff should survive.
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise TransportError("simulated transient 529 overloaded_error")

        question = _last_user_text(messages)

        # Turn 2+: a tool_result is already in the history -> give the final
        # answer and stop. stop_reason="end_turn" ends the agent loop.
        if _has_tool_result(messages):
            answer = self._final_answer(messages)
            return self._message(
                content=[{"type": "text", "text": answer}],
                stop_reason="end_turn",
            )

        # Turn 1: decide whether to call a tool. stop_reason="tool_use" tells
        # the loop to execute the tool and call again.
        plan = self._plan_tool_call(question)
        if plan is not None:
            name, tool_input = plan
            return self._message(
                content=[
                    {"type": "text", "text": f"I'll use the {name} tool."},
                    {
                        "type": "tool_use",
                        "id": _next_tool_use_id(),
                        "name": name,
                        "input": tool_input,
                    },
                ],
                stop_reason="tool_use",
            )

        # No tool needed -> answer directly.
        return self._message(
            content=[{"type": "text", "text": "I don't need a tool for that."}],
            stop_reason="end_turn",
        )

    # -- internal helpers ---------------------------------------------------

    def _message(self, *, content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
        # Mirrors the real Message object's serialized shape.
        out_tokens = sum(len(b.get("text", "")) for b in content) // 4 + 1
        return {
            "id": f"msg_fake_{random.randint(1000, 9999)}",  # noqa: S311 - cosmetic
            "type": "message",
            "role": "assistant",
            "model": "fake-claude",
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 42, "output_tokens": out_tokens},
        }

    def _plan_tool_call(self, question: str) -> tuple[str, dict[str, Any]] | None:
        q = question.lower()
        if "weather" in q:
            city = self._extract_city(question)
            return ("get_weather", {"city": city})
        if "+" in question or "add" in q or "plus" in q or "sum" in q:
            a, b = self._extract_two_ints(question)
            return ("add", {"a": a, "b": b})
        return None

    @staticmethod
    def _extract_city(question: str) -> str:
        # Look for "in <City>" else fall back to the last capitalized run.
        words = question.replace("?", "").split()
        if "in" in [w.lower() for w in words]:
            idx = [w.lower() for w in words].index("in")
            return " ".join(words[idx + 1 :]) or "Tokyo"
        return "Tokyo"

    @staticmethod
    def _extract_two_ints(question: str) -> tuple[int, int]:
        # Replace common separators with spaces, then strip trailing punctuation
        # off each token before the digit check ("21?" -> "21").
        cleaned = question.replace("+", " ").replace("?", " ").replace(",", " ")
        nums = [int(tok) for tok in cleaned.split() if tok.lstrip("-").isdigit()]
        if len(nums) >= 2:
            return nums[0], nums[1]
        return 0, 0

    @staticmethod
    def _final_answer(messages: list[dict[str, Any]]) -> str:
        # Echo the most recent tool_result content into a natural answer.
        result_text = ""
        for msg in reversed(messages):
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        rc = block.get("content", "")
                        result_text = rc if isinstance(rc, str) else str(rc)
                        break
            if result_text:
                break
        return f"Result: {result_text}"


def call_with_retry(
    fn: Callable[[], dict[str, Any]],
    *,
    max_retries: int = 4,
    base_delay: float = 0.05,
    max_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> dict[str, Any]:
    """Call `fn` with exponential backoff + jitter on TransportError.

    The real Anthropic SDK already retries 429/5xx with backoff (configurable via
    `max_retries`); we reimplement it here so the concept is visible and testable
    offline. `sleep`/`rand` are injectable so tests run instantly and
    deterministically.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except TransportError as exc:
            last_exc = exc
            # exponential backoff: base * 2**attempt, capped, plus jitter.
            delay = min(base_delay * (2**attempt), max_delay) + base_delay * rand()
            sleep(delay)
    assert last_exc is not None
    raise last_exc


def with_timeout(
    fn: Callable[[], dict[str, Any]],
    *,
    timeout: float,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Enforce a wall-clock deadline around a synchronous transport call.

    Real SDK clients take a `timeout=` and raise on slow requests. We model the
    timeout *concept* here: the deadline is checked after the call returns
    (a single synchronous call can't be interrupted mid-flight without threads),
    so a call that took longer than `timeout` is treated as a timeout failure.
    This keeps the example dependency-free while making the contract explicit.
    """
    start = clock()
    result = fn()
    elapsed = clock() - start
    if elapsed > timeout:
        raise TransportTimeout(f"transport call took {elapsed:.3f}s > timeout {timeout:.3f}s")
    return result
