"""Unit tests for eval.score.

The scoring functions are pure — given a TestCase + Answer, the
verdict is deterministic. We synthesize Answer objects directly
(no live pipeline) and assert the metric outcomes.
"""

from __future__ import annotations

from common.models import Answer, Citation, Claim, Language
from eval.dataset import TestCase as EvalCase  # avoid pytest "TestCase" collection
from eval.score import Verdict, score_answer


def _case(**overrides: object) -> EvalCase:
    defaults: dict[str, object] = {
        "id": "c1",
        "question": "Q?",
        "expected_language": Language.EN,
        "expected_behavior": "answer",
        "must_cite_doc_ids": [],
        "must_mention": [],
        "must_not_mention": [],
        "max_latency_s": None,
    }
    defaults.update(overrides)
    return EvalCase(**defaults)


def _answer(
    *,
    claims: list[Claim] | None = None,
    language: Language = Language.EN,
) -> Answer:
    cs = claims if claims is not None else []
    used = sorted({c.chunk_id for cl in cs for c in cl.citations})
    return Answer(
        question="Q?",
        answer_language=language,
        claims=cs,
        used_chunks=used,
        hop_count=1,
    )


def _claim(text: str, doc_id: str, chunk_id: str | None = None) -> Claim:
    return Claim(
        text=text,
        citations=[
            Citation(
                chunk_id=chunk_id or f"{doc_id}-c1",
                doc_id=doc_id,
                page_numbers=[1],
            )
        ],
    )


def _verdict(scored: object, metric_name: str) -> Verdict:
    for m in scored.metrics:  # type: ignore[attr-defined]
        if m.name == metric_name:
            return m.verdict  # type: ignore[no-any-return]
    raise AssertionError(f"metric {metric_name!r} not in result")


# ==================== behavior ====================


def test_behavior_pass_when_answer_has_claims() -> None:
    case = _case(expected_behavior="answer")
    ans = _answer(claims=[_claim("X.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "behavior") == Verdict.PASS


def test_behavior_fail_when_expected_answer_but_empty_claims() -> None:
    case = _case(expected_behavior="answer")
    ans = _answer(claims=[])
    assert _verdict(score_answer(case, ans, 1.0), "behavior") == Verdict.FAIL


def test_behavior_pass_when_expected_refusal_and_empty_claims() -> None:
    case = _case(expected_behavior="refuse")
    ans = _answer(claims=[])
    assert _verdict(score_answer(case, ans, 1.0), "behavior") == Verdict.PASS


def test_behavior_fail_when_expected_refusal_but_claims_present() -> None:
    """A false-accept on an out-of-scope query is the worst failure
    mode — the user gets a confidently wrong answer."""
    case = _case(expected_behavior="refuse")
    ans = _answer(claims=[_claim("Spurious.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "behavior") == Verdict.FAIL


# ==================== language ====================


def test_language_pass_on_exact_match() -> None:
    case = _case(expected_language=Language.AR)
    ans = _answer(language=Language.AR, claims=[_claim("ادعاء.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "language") == Verdict.PASS


def test_language_fail_on_mismatch() -> None:
    """Caught a real bug pre-translation: Arabic question, English answer."""
    case = _case(expected_language=Language.AR)
    ans = _answer(language=Language.EN, claims=[_claim("Wrong language.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "language") == Verdict.FAIL


# ==================== citations ====================


def test_citations_skip_when_no_requirement() -> None:
    case = _case()
    ans = _answer(claims=[_claim("X.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "citations") == Verdict.SKIP


def test_citations_pass_when_required_doc_cited() -> None:
    case = _case(must_cite_doc_ids=["fema-x"])
    ans = _answer(claims=[_claim("X.", "fema-x")])
    assert _verdict(score_answer(case, ans, 1.0), "citations") == Verdict.PASS


def test_citations_fail_when_required_doc_missing() -> None:
    case = _case(must_cite_doc_ids=["fema-x"])
    ans = _answer(claims=[_claim("X.", "fema-y")])
    assert _verdict(score_answer(case, ans, 1.0), "citations") == Verdict.FAIL


def test_citations_fail_on_refusal_when_required() -> None:
    """If we require a citation and got an empty answer, that's a fail
    — caller must explicitly model refusals with expected_behavior."""
    case = _case(must_cite_doc_ids=["fema-x"])
    ans = _answer(claims=[])
    assert _verdict(score_answer(case, ans, 1.0), "citations") == Verdict.FAIL


def test_citations_check_is_case_sensitive_for_doc_ids() -> None:
    """doc_ids are deterministic slugs — a case difference should not
    silently pass."""
    case = _case(must_cite_doc_ids=["fema-x"])
    ans = _answer(claims=[_claim("X.", "FEMA-X")])
    assert _verdict(score_answer(case, ans, 1.0), "citations") == Verdict.FAIL


# ==================== must_mention ====================


def test_must_mention_skip_when_empty() -> None:
    case = _case()
    ans = _answer(claims=[_claim("anything", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_mention") == Verdict.SKIP


def test_must_mention_pass_when_keyword_present() -> None:
    case = _case(must_mention=["base shear"])
    ans = _answer(claims=[_claim("The base shear V = Cs * W.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_mention") == Verdict.PASS


def test_must_mention_is_case_insensitive() -> None:
    case = _case(must_mention=["NEHRP"])
    ans = _answer(claims=[_claim("The nehrp provisions...", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_mention") == Verdict.PASS


def test_must_mention_fail_when_keyword_missing() -> None:
    case = _case(must_mention=["soil"])
    ans = _answer(claims=[_claim("Foundation analysis here.", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_mention") == Verdict.FAIL


def test_must_mention_can_span_multiple_claims() -> None:
    """The blob check joins all claims; one keyword in claim A and
    another in claim B both satisfy."""
    case = _case(must_mention=["base shear", "Cs"])
    ans = _answer(
        claims=[
            _claim("Base shear V = ...", "doc-a"),
            _claim("...where Cs is the response coefficient.", "doc-a"),
        ]
    )
    assert _verdict(score_answer(case, ans, 1.0), "must_mention") == Verdict.PASS


# ==================== must_not_mention ====================


def test_must_not_mention_skip_when_empty() -> None:
    case = _case()
    ans = _answer(claims=[_claim("anything", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_not_mention") == Verdict.SKIP


def test_must_not_mention_pass_when_forbidden_absent() -> None:
    case = _case(must_not_mention=["Egyptian Code"])
    ans = _answer(claims=[_claim("Per ASCE 7-22, ...", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_not_mention") == Verdict.PASS


def test_must_not_mention_fail_when_forbidden_present() -> None:
    """Caught a real failure mode: out-of-scope query about ECP that
    nevertheless mentioned 'Egyptian Code' in the answer text."""
    case = _case(must_not_mention=["Egyptian Code"])
    ans = _answer(claims=[_claim("The Egyptian Code requires...", "doc-a")])
    assert _verdict(score_answer(case, ans, 1.0), "must_not_mention") == Verdict.FAIL


def test_must_not_mention_passes_vacuously_on_refusal() -> None:
    """No claims = nothing forbidden to find."""
    case = _case(must_not_mention=["Egyptian Code"])
    ans = _answer(claims=[])
    assert _verdict(score_answer(case, ans, 1.0), "must_not_mention") == Verdict.PASS


# ==================== latency ====================


def test_latency_skip_when_no_budget() -> None:
    case = _case(max_latency_s=None)
    ans = _answer(claims=[_claim("X.", "doc-a")])
    assert _verdict(score_answer(case, ans, 999.0), "latency") == Verdict.SKIP


def test_latency_pass_when_under_budget() -> None:
    case = _case(max_latency_s=240.0)
    ans = _answer(claims=[_claim("X.", "doc-a")])
    assert _verdict(score_answer(case, ans, 90.0), "latency") == Verdict.PASS


def test_latency_warns_but_does_not_fail_over_budget() -> None:
    """Latency is hardware-dependent — a slow CI runner shouldn't fail
    the eval. WARN keeps the signal without blocking the build."""
    case = _case(max_latency_s=60.0)
    ans = _answer(claims=[_claim("X.", "doc-a")])
    scored = score_answer(case, ans, 120.0)
    assert _verdict(scored, "latency") == Verdict.WARN
    assert scored.passed is True  # warnings don't fail the case
    assert scored.warn_count == 1


# ==================== aggregate ====================


def test_scored_case_passed_iff_no_fail_metrics() -> None:
    case = _case(must_mention=["soil"])
    ans = _answer(claims=[_claim("Discusses foundations.", "doc-a")])
    scored = score_answer(case, ans, 1.0)
    assert scored.passed is False
    assert scored.fail_count == 1


def test_scored_case_passes_on_full_satisfaction() -> None:
    case = _case(
        expected_language=Language.EN,
        must_cite_doc_ids=["fema-x"],
        must_mention=["Cs"],
        must_not_mention=["Egyptian"],
        max_latency_s=240.0,
    )
    ans = _answer(claims=[_claim("Cs is calculated as...", "fema-x")])
    scored = score_answer(case, ans, 90.0)
    assert scored.passed is True
    assert scored.fail_count == 0
    assert scored.warn_count == 0
