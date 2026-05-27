"""Pure scoring functions over (TestCase, Answer) pairs.

Each metric is a deterministic check that returns one of:

- ``PASS``: the answer satisfies the expectation
- ``FAIL``: hard violation — counts against the aggregate pass rate
- ``WARN``: soft violation (currently only latency) — doesn't count
  against pass rate but surfaces in the report
- ``SKIP``: the test case doesn't exercise this metric (e.g. no
  must_mention list provided)

Keeping the scoring pure means the eval runner can be tested without
spinning up the full pipeline — we synthesize Answer objects and call
``score_answer`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from common.models import Answer
from eval.dataset import TestCase


class Verdict(Enum):
    PASS = "pass"  # noqa: S105 — not a password, an enum value
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass(frozen=True)
class MetricResult:
    name: str
    verdict: Verdict
    detail: str = ""


@dataclass(frozen=True)
class ScoredCase:
    """All metric outcomes for one (test-case, answer) pair.

    ``passed`` is the hard aggregate: True iff every non-SKIP, non-WARN
    metric is PASS. WARN does not count against passed.
    """

    case_id: str
    elapsed_s: float
    metrics: list[MetricResult]

    @property
    def passed(self) -> bool:
        return all(m.verdict != Verdict.FAIL for m in self.metrics)

    @property
    def fail_count(self) -> int:
        return sum(1 for m in self.metrics if m.verdict == Verdict.FAIL)

    @property
    def warn_count(self) -> int:
        return sum(1 for m in self.metrics if m.verdict == Verdict.WARN)


# ==================== individual metrics ====================


def _check_behavior(case: TestCase, answer: Answer) -> MetricResult:
    """answer vs refuse — the most important single check."""
    if case.expected_behavior == "answer":
        if answer.claims:
            return MetricResult("behavior", Verdict.PASS, f"{len(answer.claims)} claim(s)")
        return MetricResult(
            "behavior", Verdict.FAIL, "expected an answer but got empty claims (refusal)"
        )
    # Reaching here means expected_behavior == "refuse".
    if not answer.claims:
        return MetricResult("behavior", Verdict.PASS, "empty claims as expected")
    return MetricResult(
        "behavior",
        Verdict.FAIL,
        f"expected refusal but got {len(answer.claims)} claim(s)",
    )


def _check_language(case: TestCase, answer: Answer) -> MetricResult:
    if answer.answer_language == case.expected_language:
        return MetricResult("language", Verdict.PASS, answer.answer_language.value)
    return MetricResult(
        "language",
        Verdict.FAIL,
        f"expected {case.expected_language.value}, got {answer.answer_language.value}",
    )


def _check_required_citations(case: TestCase, answer: Answer) -> MetricResult:
    if not case.must_cite_doc_ids:
        return MetricResult("citations", Verdict.SKIP)
    # Empty-claims answers can't have citations; reframe the check to
    # match the runner's behavior (a refusal trumps citation checks).
    if not answer.claims:
        return MetricResult(
            "citations", Verdict.FAIL, "no claims, so no citations to satisfy requirement"
        )
    cited_docs = {c.doc_id for cl in answer.claims for c in cl.citations}
    missing = [d for d in case.must_cite_doc_ids if d not in cited_docs]
    if missing:
        return MetricResult(
            "citations", Verdict.FAIL, f"missing required doc citation(s): {missing}"
        )
    return MetricResult(
        "citations", Verdict.PASS, f"all {len(case.must_cite_doc_ids)} required doc(s) cited"
    )


def _check_must_mention(case: TestCase, answer: Answer) -> MetricResult:
    if not case.must_mention:
        return MetricResult("must_mention", Verdict.SKIP)
    if not answer.claims:
        return MetricResult(
            "must_mention", Verdict.FAIL, "no claims to search for required keywords"
        )
    blob = " ".join(c.text for c in answer.claims).lower()
    missing = [k for k in case.must_mention if k.lower() not in blob]
    if missing:
        return MetricResult("must_mention", Verdict.FAIL, f"missing keyword(s): {missing}")
    return MetricResult(
        "must_mention", Verdict.PASS, f"all {len(case.must_mention)} keyword(s) present"
    )


def _check_must_not_mention(case: TestCase, answer: Answer) -> MetricResult:
    if not case.must_not_mention:
        return MetricResult("must_not_mention", Verdict.SKIP)
    if not answer.claims:
        # No claims means nothing to forbid — vacuously true.
        return MetricResult("must_not_mention", Verdict.PASS, "no claims")
    blob = " ".join(c.text for c in answer.claims).lower()
    hits = [k for k in case.must_not_mention if k.lower() in blob]
    if hits:
        return MetricResult(
            "must_not_mention", Verdict.FAIL, f"forbidden keyword(s) present: {hits}"
        )
    return MetricResult(
        "must_not_mention",
        Verdict.PASS,
        f"all {len(case.must_not_mention)} forbidden keyword(s) absent",
    )


def _check_latency(case: TestCase, elapsed_s: float) -> MetricResult:
    if case.max_latency_s is None:
        return MetricResult("latency", Verdict.SKIP, f"{elapsed_s:.1f}s")
    if elapsed_s <= case.max_latency_s:
        return MetricResult(
            "latency", Verdict.PASS, f"{elapsed_s:.1f}s ≤ {case.max_latency_s:.1f}s"
        )
    # Latency is a WARN, never a FAIL — hardware varies and we don't
    # want CPU/GPU swaps to regress the eval pass rate.
    return MetricResult(
        "latency",
        Verdict.WARN,
        f"{elapsed_s:.1f}s > {case.max_latency_s:.1f}s budget",
    )


def score_answer(case: TestCase, answer: Answer, elapsed_s: float) -> ScoredCase:
    """Run every metric over one (case, answer) pair."""
    return ScoredCase(
        case_id=case.id,
        elapsed_s=elapsed_s,
        metrics=[
            _check_behavior(case, answer),
            _check_language(case, answer),
            _check_required_citations(case, answer),
            _check_must_mention(case, answer),
            _check_must_not_mention(case, answer),
            _check_latency(case, elapsed_s),
        ],
    )
