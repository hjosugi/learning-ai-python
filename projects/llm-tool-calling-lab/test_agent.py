"""Non-interactive tests. Runs with plain `python3 test_agent.py`.

Exits non-zero on the first failed assertion (AssertionError propagates), so it
works as a CI gate without pytest. Mirrors the repo's existing test style.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent import (  # noqa: E402
    FINAL_OUTPUT_SCHEMA,
    run_agent,
    stream_text,
    structured_result,
)
from schema import SchemaError, validate  # noqa: E402
from transport import (  # noqa: E402
    FakeTransport,
    TransportError,
    TransportTimeout,
    call_with_retry,
    with_timeout,
)


def test_weather_tool_loop() -> None:
    run = run_agent("What is the weather in Tokyo?", FakeTransport())
    assert run.tool_calls == [("get_weather", {"city": "Tokyo"})], run.tool_calls
    assert "18C" in run.final_text, run.final_text
    assert run.stop_reason == "end_turn"
    # Two transport calls: one tool_use turn, one end_turn turn.
    assert run.turns == 2, run.turns


def test_add_tool_loop() -> None:
    run = run_agent("What is 21 + 21?", FakeTransport())
    assert run.tool_calls == [("add", {"a": 21, "b": 21})], run.tool_calls
    assert "42" in run.final_text, run.final_text


def test_no_tool_needed() -> None:
    run = run_agent("Say hello.", FakeTransport())
    assert run.tool_calls == [], run.tool_calls
    assert run.turns == 1, run.turns


def test_response_shape_matches_messages_api() -> None:
    # The assistant turn appended to history must carry typed content blocks,
    # exactly like the real Messages API response.
    run = run_agent("What is the weather in Tokyo?", FakeTransport())
    assistant_turns = [m for m in run.messages if m["role"] == "assistant"]
    first = assistant_turns[0]["content"]
    types = {b["type"] for b in first}
    assert "tool_use" in types, types
    tool_use = next(b for b in first if b["type"] == "tool_use")
    assert set(tool_use) >= {"type", "id", "name", "input"}, tool_use
    # The follow-up user turn must carry a tool_result referencing that id.
    user_turns = [m for m in run.messages if m["role"] == "user"]
    result_blocks = [
        b
        for m in user_turns
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert result_blocks, "no tool_result block appended"
    assert result_blocks[0]["tool_use_id"] == tool_use["id"]


def test_retry_survives_transient_failure() -> None:
    # Two simulated failures, then success: retry must absorb both.
    transport = FakeTransport(fail_times=2)
    calls = {"n": 0}

    def fn() -> dict:
        calls["n"] += 1
        return transport.create(
            model="fake",
            max_tokens=10,
            system="s",
            tools=[],
            messages=[{"role": "user", "content": "Say hello."}],
        )

    # Inject a no-op sleep and fixed rand so the test is instant + deterministic.
    result = call_with_retry(
        fn, max_retries=4, sleep=lambda _d: None, rand=lambda: 0.0
    )
    assert result["stop_reason"] == "end_turn"
    assert calls["n"] == 3, calls["n"]  # 2 failures + 1 success


def test_retry_gives_up_after_max_retries() -> None:
    transport = FakeTransport(fail_times=99)
    raised = False
    try:
        call_with_retry(
            lambda: transport.create(
                model="f", max_tokens=1, system="", tools=[], messages=[{"role": "user", "content": "hi"}]
            ),
            max_retries=3,
            sleep=lambda _d: None,
            rand=lambda: 0.0,
        )
    except TransportError:
        raised = True
    assert raised, "expected TransportError after retries exhausted"


def test_timeout_concept() -> None:
    # A call that takes longer than the deadline is reported as a timeout.
    slow = FakeTransport(latency=0.02)
    raised = False
    try:
        with_timeout(
            lambda: slow.create(
                model="f", max_tokens=1, system="", tools=[], messages=[{"role": "user", "content": "hi"}]
            ),
            timeout=0.001,
        )
    except TransportTimeout:
        raised = True
    assert raised, "expected TransportTimeout"

    # A fast call under the deadline succeeds.
    fast = FakeTransport()
    ok = with_timeout(
        lambda: fast.create(
            model="f", max_tokens=1, system="", tools=[], messages=[{"role": "user", "content": "hi"}]
        ),
        timeout=5.0,
    )
    assert ok["stop_reason"] == "end_turn"


def test_streaming_reassembles() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    chunks = list(stream_text(text, chunk_size=8))
    assert len(chunks) > 1, "expected multiple chunks"
    assert "".join(chunks) == text


def test_structured_output_valid() -> None:
    run = run_agent("What is the weather in Tokyo?", FakeTransport())
    obj = structured_result(run)
    assert obj["used_tool"] is True
    assert obj["tool_names"] == ["get_weather"]
    # Validating again must not raise.
    validate(obj, FINAL_OUTPUT_SCHEMA)


def test_schema_rejects_bad_output() -> None:
    bad = {"answer": "x", "used_tool": "yes", "tool_names": []}  # used_tool not bool
    raised = False
    try:
        validate(bad, FINAL_OUTPUT_SCHEMA)
    except SchemaError:
        raised = True
    assert raised, "schema should reject non-boolean used_tool"

    missing = {"answer": "x", "used_tool": True}  # missing tool_names
    raised = False
    try:
        validate(missing, FINAL_OUTPUT_SCHEMA)
    except SchemaError:
        raised = True
    assert raised, "schema should reject missing required field"

    extra = {"answer": "x", "used_tool": True, "tool_names": [], "oops": 1}
    raised = False
    try:
        validate(extra, FINAL_OUTPUT_SCHEMA)
    except SchemaError:
        raised = True
    assert raised, "schema should reject additional properties"


def test_evals_all_pass() -> None:
    # The eval suite itself must be green.
    from evals.run_evals import run_evals

    failures = run_evals()
    assert failures == 0, f"{failures} eval case(s) failed"


def _all_tests() -> list:
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    tests = _all_tests()
    for t in tests:
        t()
    print(f"ok ({len(tests)} tests)")
