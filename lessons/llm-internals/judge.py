"""An LLM-as-judge style eval harness with a deterministic rubric scorer.

When a model's output is open-ended ("explain X", "summarize Y") there is no
single correct string to diff against, so substring/exact-match evals fail. The
industry pattern is **LLM-as-judge**: a second model scores the candidate answer
against a rubric and (optionally) a reference answer, returning a score and a
pass/fail.

The *interface* of an LLM judge is what matters and is what this file teaches:

    score(case) -> JudgeResult(score in [0,1], passed: bool, reasons: [...])

Here the judge body is a **deterministic rubric** (pure functions over the text)
instead of a model call, so the harness is reproducible, free, and testable. The
upgrade path replaces only ``_rubric_score`` with a real ``messages.create``
call that asks a model to grade against the same rubric and return JSON. The
dataset, the runner, and the pass/fail gate stay identical.

The rubric rewards answers that:
* overlap in content with the reference (token F1),
* contain the rubric's required keywords,
* are grounded -- do not introduce claims absent from input+reference, and
* are an appropriate length (not empty, not padded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_f1(candidate: str, reference: str) -> float:
    """Token-overlap F1 between candidate and reference (multiset based).

    F1 = 2*precision*recall / (precision+recall), where precision/recall are
    over the shared multiset of tokens. This is a standard cheap proxy for
    "did the answer say roughly the right things" (used by SQuAD-style evals).
    """
    cand = _tokens(candidate)
    ref = _tokens(reference)
    if not cand or not ref:
        return 0.0
    # Multiset intersection size.
    from collections import Counter

    overlap = Counter(cand) & Counter(ref)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = shared / len(cand)
    recall = shared / len(ref)
    return 2 * precision * recall / (precision + recall)


@dataclass
class JudgeCase:
    """One eval case fed to the judge."""

    input: str
    candidate: str
    reference: str
    must_include: list[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    """The judge's verdict for one case."""

    score: float          # in [0, 1]
    passed: bool
    reasons: list[str]


# Rubric weights. They sum to 1.0 so the final score stays in [0, 1].
_W_F1 = 0.5
_W_KEYWORDS = 0.3
_W_GROUNDED = 0.2
_PASS_THRESHOLD = 0.5


def _rubric_score(case: JudgeCase) -> JudgeResult:
    """The deterministic stand-in for a model judge.

    Replace ONLY this function with a real ``messages.create`` grading call to
    go from a rubric judge to a true LLM-as-judge (see the README upgrade path).
    The returned :class:`JudgeResult` shape is the contract the rest of the
    harness depends on.
    """
    reasons: list[str] = []

    # 1) Content overlap with the reference.
    f1 = token_f1(case.candidate, case.reference)
    reasons.append(f"reference token-F1 = {f1:.2f}")

    # 2) Required keyword coverage.
    cand_tokens = set(_tokens(case.candidate))
    required = [kw.lower() for kw in case.must_include]
    if required:
        hits = sum(1 for kw in required if kw in cand_tokens)
        keyword_score = hits / len(required)
        reasons.append(f"required keywords {hits}/{len(required)} present")
    else:
        keyword_score = 1.0
        reasons.append("no required keywords")

    # 3) Groundedness: fraction of *content* candidate tokens that also appear
    #    in the input or reference. A low value means the answer invented
    #    material not supported by the source -- a hallucination signal.
    allowed = set(_tokens(case.input)) | set(_tokens(case.reference))
    cand_list = _tokens(case.candidate)
    if cand_list:
        grounded_hits = sum(1 for t in cand_list if t in allowed)
        grounded = grounded_hits / len(cand_list)
    else:
        grounded = 0.0
    reasons.append(f"groundedness = {grounded:.2f}")

    # 4) Length sanity: empty answers fail outright.
    if not cand_list:
        reasons.append("empty candidate -> automatic fail")
        return JudgeResult(score=0.0, passed=False, reasons=reasons)

    score = _W_F1 * f1 + _W_KEYWORDS * keyword_score + _W_GROUNDED * grounded
    score = max(0.0, min(1.0, score))
    passed = score >= _PASS_THRESHOLD
    reasons.append(f"weighted score = {score:.2f} ({'PASS' if passed else 'FAIL'})")
    return JudgeResult(score=score, passed=passed, reasons=reasons)


def score(case: JudgeCase) -> JudgeResult:
    """Score a single case. This is the stable public entry point."""
    return _rubric_score(case)


def run_judge(cases: list[JudgeCase]) -> tuple[list[JudgeResult], int]:
    """Score every case. Returns ``(results, num_failed)``."""
    results = [score(c) for c in cases]
    failed = sum(1 for r in results if not r.passed)
    return results, failed


# A tiny bundled dataset: each case has a deliberately good and bad candidate so
# tests can assert good > bad. Here we expose the *good* answers; the matching
# bad answers live in the demo/tests.
def default_dataset() -> list[JudgeCase]:
    return [
        JudgeCase(
            input="What does BPE stand for and what does it operate on?",
            candidate=(
                "BPE stands for byte pair encoding. It operates on bytes, "
                "repeatedly merging the most frequent adjacent pair of tokens."
            ),
            reference=(
                "BPE means byte pair encoding; it merges the most frequent "
                "adjacent byte pairs into new tokens."
            ),
            must_include=["bpe", "byte", "pair", "merge"],
        ),
        JudgeCase(
            input="Why do n-gram models need smoothing?",
            candidate=(
                "Smoothing gives unseen n-grams a small nonzero probability so "
                "the model never assigns zero probability to a valid sequence."
            ),
            reference=(
                "Without smoothing, unseen n-grams get zero probability, making "
                "whole sentences impossible; add-k smoothing fixes this."
            ),
            must_include=["smoothing", "zero", "probability"],
        ),
    ]
