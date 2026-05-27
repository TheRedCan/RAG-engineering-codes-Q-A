"""Unit tests for generation.scope.

The LLM is mocked. The verifier must default to PASS under any kind of
uncertainty (LLM error, malformed JSON, unexpected verdict text) — we
removed an earlier verifier specifically because it defaulted to refuse
and nuked good answers.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from common.models import Answer, Citation, Claim, Language
from generation.scope import verify_scope


def _answer(claims: list[Claim] | None = None) -> Answer:
    cs = claims if claims is not None else []
    used = sorted({c.chunk_id for cl in cs for c in cl.citations})
    return Answer(
        question="Q?",
        answer_language=Language.EN,
        claims=cs,
        used_chunks=used,
        hop_count=1,
    )


def _claim(text: str, doc_id: str = "doc-a", section: str | None = None) -> Claim:
    return Claim(
        text=text,
        citations=[
            Citation(
                chunk_id=f"{doc_id}-c1",
                doc_id=doc_id,
                page_numbers=[1],
                section_path=section,
            )
        ],
    )


def _verdict(verdict: str, reason: str = "ok") -> str:
    return json.dumps({"verdict": verdict, "reason": reason})


def test_empty_answer_passes_through_without_llm_call() -> None:
    """Already-empty answer = nothing to verify; the orchestrator
    surfaces the refusal regardless."""
    original = _answer([])
    with patch("generation.scope.chat") as chat_mock:
        out = verify_scope("anything?", original)
    chat_mock.assert_not_called()
    assert out is original


def test_pass_verdict_returns_original_answer() -> None:
    ans = _answer([_claim("Good claim.")])
    with patch("generation.scope.chat", return_value=_verdict("pass")):
        out = verify_scope("anything?", ans)
    # Same content (Pydantic equality), not necessarily same identity.
    assert out.claims == ans.claims
    assert out.used_chunks == ans.used_chunks


def test_refuse_verdict_produces_empty_claims_with_preserved_metadata() -> None:
    ans = _answer([_claim("Spurious claim about a fabricated paper.")])
    reason = "user asked about Smith 2024 but answer has no Smith citation"
    with patch("generation.scope.chat", return_value=_verdict("refuse", reason)):
        out = verify_scope("What does the Smith 2024 paper say?", ans)
    assert out.claims == []
    assert out.used_chunks == []
    # Language stamp preserved so UI can render the refusal in the right language.
    assert out.answer_language == ans.answer_language
    assert out.hop_count == ans.hop_count
    assert out.question == ans.question


def test_malformed_verifier_output_defaults_to_pass() -> None:
    """Critical: an unparseable verifier reply must NOT nuke the answer.
    This is why we removed the previous verifier — it defaulted to
    refuse on bad output and dropped legitimate replies."""
    ans = _answer([_claim("Legitimate claim.")])
    with patch("generation.scope.chat", return_value="not json at all"):
        out = verify_scope("anything?", ans)
    assert out.claims == ans.claims  # PASS, answer kept


def test_llm_exception_defaults_to_pass() -> None:
    """Ollama 500s / timeouts must not nuke the answer either."""
    ans = _answer([_claim("Legitimate claim.")])
    with patch("generation.scope.chat", side_effect=RuntimeError("ollama down")):
        out = verify_scope("anything?", ans)
    assert out.claims == ans.claims


def test_invalid_verdict_string_defaults_to_pass() -> None:
    """If the verifier returns a verdict outside the allowed pattern
    (e.g. 'maybe', 'unsure'), schema validation fails -> PASS."""
    ans = _answer([_claim("Legitimate claim.")])
    bad = json.dumps({"verdict": "maybe", "reason": "unclear"})
    with patch("generation.scope.chat", return_value=bad):
        out = verify_scope("anything?", ans)
    assert out.claims == ans.claims


def test_prompt_includes_question_claims_and_citations() -> None:
    """The verifier must see what the user asked AND the structured
    answer (claims + cited doc_ids + sections) — that's how it spots
    explicit mismatches."""
    ans = _answer(
        [
            _claim("Steel moment frames per ASCE 7.", "fema-x", section="§12.8.1"),
            _claim("Additional context.", "nist-y"),
        ]
    )
    captured: dict[str, str] = {}

    def _capture(*, system: str, user: str, **_: object) -> str:
        captured["system"] = system
        captured["user"] = user
        return _verdict("pass")

    with patch("generation.scope.chat", side_effect=_capture):
        verify_scope("Egyptian Code procedure for steel?", ans)

    # User prompt carries everything the verifier needs to spot mismatch.
    assert "Egyptian Code" in captured["user"]
    assert "Steel moment frames" in captured["user"]
    assert "fema-x" in captured["user"]
    assert "nist-y" in captured["user"]
    assert "§12.8.1" in captured["user"]
