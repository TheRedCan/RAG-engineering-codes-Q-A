"""Unit tests for the end-to-end generation.answer orchestrator.

Everything below the function is mocked: retrieval (hybrid+multihop),
rerank, the LLM HTTP call. We're verifying the *wiring*, not the model.
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
