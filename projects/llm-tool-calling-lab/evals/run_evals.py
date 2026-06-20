"""Eval runner: score the agent against a tiny expected-behavior dataset.

An "eval" is the regression test for agent behavior: a fixed set of inputs, each
with what we expect to happen (which tools the agent should call, what the final
answer should contain). The runner executes the real agent loop against the
deterministic FakeTransport and scores each case pass/fail.

This is deliberately small and offline so it can run on every commit. With a
real model you would keep the SAME dataset and assertions and just swap the
transport - the eval is what catches a prompt or model change that silently
breaks tool selection.

Exit code is non-zero if any case fails, so this doubles as a CI gate.
"""

from __future__ import annotations

import json
import os
import sys

# Make the project root importable when run as a script from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent import run_agent, structured_result  # noqa: E402
from transport import FakeTransport  # noqa: E402

DATASET_PATH = os.path.join(_HERE, "dataset.json")


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_case(case: dict) -> tuple[bool, str]:
    """Run one case and return (passed, detail)."""
    # fail_times=1 also exercises retry-with-backoff inside the eval.
    run = run_agent(case["input"], FakeTransport(fail_times=1))

    got_tools = [name for name, _ in run.tool_calls]
    expect_tools = case["expect_tools"]
    if got_tools != expect_tools:
        return (False, f"tools {got_tools!r} != expected {expect_tools!r}")

    needle = case["expect_answer_contains"]
    if needle and needle not in run.final_text:
        return (False, f"answer {run.final_text!r} missing {needle!r}")

    # Every passing case must also produce a schema-valid structured object.
    try:
        structured_result(run)
    except Exception as exc:  # noqa: BLE001
        return (False, f"structured output invalid: {exc}")

    return (True, "ok")


def run_evals() -> int:
    """Run all cases, print a report, and return the count of failures."""
    cases = load_dataset()
    failures = 0
    for case in cases:
        passed, detail = score_case(case)
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        print(f"[{status}] {case['name']}: {detail}")

    total = len(cases)
    print(f"\n{total - failures}/{total} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_evals() else 0)
