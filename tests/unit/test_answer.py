"""Unit tests for the end-to-end generation.answer orchestrator.

Everything below the function is mocked: retrieval (hybrid+multihop),
rerank, translation, and the LLM HTTP call. We're verifying the
*wiring*, not the model.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from common.errors import LlmUnavailableError
from common.models import Chunk, Language, RetrievedChunk
from generation.answer import answer_question


def _retrieved(chunk_id: str, doc_id: str = "doc-a", rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_numbers=[1, 2],
            section_path=None,
            text=f"text for {chunk_id}",
            language=Language.EN,
            char_count=10,
        ),
        rank=rank,
    )


def test_empty_retrieval_returns_refusal_without_calling_llm() -> None:
    """If retrieval surfaces nothing, we must NOT invoke the LLM — there's
    nothing for it to ground on, and asking it anyway would risk a
    hallucinated reply."""
    with (
        patch("generation.answer.health_check"),
        patch("generation.answer.multihop_search", return_value=[]),
        patch("generation.answer.chat") as chat_mock,
    ):
        out = answer_question("anything", use_multihop=True)

    chat_mock.assert_not_called()
    assert out.claims == []
    assert out.used_chunks == []
    assert out.answer_language == Language.EN


def test_happy_path_returns_canonical_answer() -> None:
    """Retrieve -> rerank -> LLM -> parse: all wired together correctly."""
    chunks = [_retrieved("a"), _retrieved("b")]
    llm_raw = json.dumps(
        {
            "answer_language": "en",
            "claims": [{"text": "Claim grounded in source a.", "cites": [1]}],
        }
    )

    with (
        patch("generation.answer.health_check"),
        patch("generation.answer.multihop_search", return_value=chunks) as mh_mock,
        patch("generation.answer.rerank_chunks", return_value=chunks) as rr_mock,
        patch("generation.answer.chat", return_value=llm_raw) as chat_mock,
    ):
        out = answer_question("Q?", use_multihop=True)

    mh_mock.assert_called_once()
    rr_mock.assert_called_once()
    chat_mock.assert_called_once()
    assert len(out.claims) == 1
    assert out.claims[0].text.startswith("Claim grounded")
    # Citation was resolved from index 1 to chunk "a".
    assert out.claims[0].citations[0].chunk_id == "a"


def test_no_multihop_path_uses_hybrid_only() -> None:
    chunks = [_retrieved("a")]
    llm_raw = json.dumps({"answer_language": "en", "claims": [{"text": "X.", "cites": [1]}]})

    with (
        patch("generation.answer.health_check"),
        patch("generation.answer.hybrid_search", return_value=chunks) as hs_mock,
        patch("generation.answer.multihop_search") as mh_mock,
        patch("generation.answer.rerank_chunks", return_value=chunks),
        patch("generation.answer.chat", return_value=llm_raw),
    ):
        answer_question("Q?", use_multihop=False)

    hs_mock.assert_called_once()
    mh_mock.assert_not_called()


def test_health_check_failure_propagates() -> None:
    """If Ollama is unreachable we surface that immediately — no
    retrieval, no LLM call."""
    with (
        patch(
            "generation.answer.health_check",
            side_effect=LlmUnavailableError(host="x", model="y", reason="z"),
        ),
        patch("generation.answer.multihop_search") as mh_mock,
        patch("generation.answer.chat") as chat_mock,
        pytest.raises(LlmUnavailableError),
    ):
        answer_question("Q?")

    mh_mock.assert_not_called()
    chat_mock.assert_not_called()


def test_hop_count_floor_one() -> None:
    """Even a pure hop-0 result set should yield hop_count >= 1 in the
    final Answer (the schema enforces hop_count >= 1)."""
    chunks = [_retrieved("a")]
    llm_raw = json.dumps({"answer_language": "en", "claims": [{"text": "X.", "cites": [1]}]})

    with (
        patch("generation.answer.health_check"),
        patch("generation.answer.multihop_search", return_value=chunks),
        patch("generation.answer.rerank_chunks", return_value=chunks),
        patch("generation.answer.chat", return_value=llm_raw),
    ):
        out = answer_question("Q?")

    assert out.hop_count >= 1


def test_unparseable_llm_output_becomes_clean_refusal() -> None:
    """If the LLM produces garbage (out-of-range citation, broken JSON,
    etc.) we must NOT exit 1 — we surface a clean empty-claims refusal so
    the user sees a "no answer" message, not a stack trace."""
    chunks = [_retrieved("a")]
    # Cite index 7 but we only show the model one chunk -> parser raises.
    bad_raw = json.dumps(
        {"answer_language": "en", "claims": [{"text": "X.", "cites": [7]}]}
    )
    with (
        patch("generation.answer.health_check"),
        patch("generation.answer.multihop_search", return_value=chunks),
        patch("generation.answer.rerank_chunks", return_value=chunks),
        patch("generation.answer.chat", return_value=bad_raw),
    ):
        out = answer_question("Q?")

    assert out.claims == []
    assert out.used_chunks == []


def test_translation_runs_for_non_english_query() -> None:
    """Arabic queries must be translated BEFORE retrieval is called, so
    the retrieval/rerank stages see the English version that aligns
    better with our English-dominant corpus. The ORIGINAL Arabic question
    must still be the one passed to the answer LLM and stored on the
    Answer model (so the user gets Arabic prose back)."""
    chunks = [_retrieved("a")]
    llm_raw = json.dumps(
        {"answer_language": "ar", "claims": [{"text": "ادعاء.", "cites": [1]}]}
    )
    arabic = "ما هي تركيبات الأحمال للتصميم الزلزالي في الكود الأمريكي ASCE 7-22؟"

    with (
        patch("generation.answer.health_check"),
        patch(
            "generation.answer.translate_query_for_retrieval",
            return_value=("load combinations for seismic design in ASCE 7-22", Language.AR),
        ) as tr_mock,
        patch("generation.answer.multihop_search", return_value=chunks) as mh_mock,
        patch("generation.answer.rerank_chunks", return_value=chunks) as rr_mock,
        patch("generation.answer.chat", return_value=llm_raw),
    ):
        out = answer_question(arabic, use_multihop=True)

    tr_mock.assert_called_once_with(arabic)
    # Retrieval and rerank must use the ENGLISH translation, not the original.
    mh_args, _mh_kwargs = mh_mock.call_args
    assert "ASCE 7-22" in mh_args[0]
    assert "ما" not in mh_args[0]
    rr_args, _rr_kwargs = rr_mock.call_args
    assert "ASCE 7-22" in rr_args[0]
    # But the Answer carries the ORIGINAL question and the model's language stamp.
    assert out.question == arabic
    assert out.answer_language == Language.AR


def test_translation_pass_through_for_english_query() -> None:
    """An English query must skip the translator's LLM call entirely
    (translate_query_for_retrieval will short-circuit), but the
    orchestrator must still flow normally."""
    chunks = [_retrieved("a")]
    llm_raw = json.dumps({"answer_language": "en", "claims": [{"text": "X.", "cites": [1]}]})
    english_q = "How is the seismic design coefficient Cs calculated under ASCE 7?"

    with (
        patch("generation.answer.health_check"),
        patch(
            "generation.answer.translate_query_for_retrieval",
            return_value=(english_q, Language.EN),
        ) as tr_mock,
        patch("generation.answer.multihop_search", return_value=chunks) as mh_mock,
        patch("generation.answer.rerank_chunks", return_value=chunks),
        patch("generation.answer.chat", return_value=llm_raw),
    ):
        answer_question(english_q)

    tr_mock.assert_called_once_with(english_q)
    # English query went into retrieval verbatim.
    mh_args, _mh_kwargs = mh_mock.call_args
    assert mh_args[0] == english_q
