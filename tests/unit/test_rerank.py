"""Unit tests for retrieval.rerank."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.models import Chunk, Language, RetrievedChunk
from retrieval.rerank import BgeReranker, rerank


def _make_candidate(chunk_id: str, rank: int, text: str = "irrelevant text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc-a",
            page_numbers=[1],
            section_path=None,
            text=text,
            language=Language.EN,
            char_count=len(text),
        ),
        dense_score=0.9 - 0.01 * rank,
        sparse_score=0.5,
        rerank_score=None,
        rank=rank,
    )


@pytest.fixture(autouse=True)
def _reset_reranker() -> None:
    BgeReranker.reset_for_tests()
    yield
    BgeReranker.reset_for_tests()


def test_rerank_reorders_by_score() -> None:
    """Reranker scores override the input order. The chunk with the
    highest cross-encoder score must end up at rank 1."""
    candidates = [
        _make_candidate("a", 1),
        _make_candidate("b", 2),
        _make_candidate("c", 3),
    ]
    # Reranker prefers c, then a, then b — opposite-ish of input order.
    mock_model = MagicMock()
    mock_model.score.return_value = [0.4, 0.1, 0.9]

    with patch("retrieval.rerank.BgeReranker.get", return_value=mock_model):
        out = rerank("query", candidates, top_k=3)

    assert [r.chunk.chunk_id for r in out] == ["c", "a", "b"]
    assert [r.rank for r in out] == [1, 2, 3]
    # Original signal scores survive.
    assert out[0].dense_score is not None
    assert out[0].sparse_score is not None
    # Rerank score is now populated.
    assert out[0].rerank_score == pytest.approx(0.9)


def test_rerank_respects_top_k() -> None:
    candidates = [_make_candidate(f"c{i}", i) for i in range(1, 11)]
    mock_model = MagicMock()
    mock_model.score.return_value = [float(10 - i) for i in range(10)]

    with patch("retrieval.rerank.BgeReranker.get", return_value=mock_model):
        out = rerank("query", candidates, top_k=3)

    assert len(out) == 3
    # Best score is index 0 (10.0), then index 1 (9.0), then index 2 (8.0).
    assert [r.chunk.chunk_id for r in out] == ["c1", "c2", "c3"]


def test_rerank_empty_input_returns_empty() -> None:
    """No candidates -> no model invocation, no crash."""
    with patch("retrieval.rerank.BgeReranker.get") as get_mock:
        out = rerank("query", [], top_k=5)
    assert out == []
    get_mock.assert_not_called()


def test_rerank_preserves_chunk_payload() -> None:
    """The chunk object passes through unchanged — only rank + rerank_score
    are updated. No silent payload mutation."""
    original = _make_candidate("a", 1, text="canonical text")
    mock_model = MagicMock()
    mock_model.score.return_value = [0.5]

    with patch("retrieval.rerank.BgeReranker.get", return_value=mock_model):
        out = rerank("query", [original], top_k=1)

    assert out[0].chunk == original.chunk  # full Pydantic equality
    assert out[0].chunk.text == "canonical text"
