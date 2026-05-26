"""Unit tests for retrieval.hybrid.

We test in three layers:

1. Pure RRF + payload-conversion logic (no Qdrant, no model).
2. ``hybrid_search`` end-to-end with mocked embedder + mocked Qdrant.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.errors import QdrantUnavailableError
from common.models import Language
from retrieval import hybrid as hybrid_mod
from retrieval.hybrid import (
    chunk_from_payload,
    hybrid_search,
    reciprocal_rank_fusion,
)

# ==================== group 1: pure helpers ====================


def test_rrf_higher_for_better_ranked() -> None:
    """An id ranked 1st in both lists must outscore one ranked 10th in both."""
    scores = dict(
        reciprocal_rank_fusion(
            [
                ["winner", "B", "C", "D", "E", "F", "G", "H", "I", "loser"],
                ["winner", "X", "Y", "Z", "W", "V", "U", "T", "S", "loser"],
            ]
        )
    )
    assert scores["winner"] > scores["loser"]


def test_rrf_rewards_appearing_in_both_lists() -> None:
    """An id in both lists outranks an id only in one, all else equal."""
    scores = dict(
        reciprocal_rank_fusion(
            [
                ["both", "only_a", "only_a2"],
                ["both", "only_b", "only_b2"],
            ]
        )
    )
    assert scores["both"] > scores["only_a"]
    assert scores["both"] > scores["only_b"]


def test_rrf_preserves_unique_ids() -> None:
    """Every id in any input list appears exactly once in the output."""
    pairs = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    ids = [cid for cid, _ in pairs]
    assert sorted(ids) == ["a", "b", "c", "d"]


def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_chunk_from_payload_validates_fields() -> None:
    payload: dict[str, Any] = {
        "chunk_id": "doc-x#00001",
        "doc_id": "doc-x",
        "page_numbers": [1, 2],
        "section_path": None,
        "text": "some chunk text",
        "language": "en",
        "char_count": 15,
    }
    chunk = chunk_from_payload(payload)
    assert chunk.chunk_id == "doc-x#00001"
    assert chunk.doc_id == "doc-x"
    assert chunk.page_numbers == [1, 2]
    assert chunk.language == Language.EN
    assert chunk.char_count == 15


def test_chunk_from_payload_rejects_invalid() -> None:
    """A malformed payload (missing field) must raise loudly, not silently
    produce a Chunk with bogus defaults."""
    bad_payload: dict[str, Any] = {"chunk_id": "x", "doc_id": "y"}  # missing fields
    with pytest.raises(KeyError):
        chunk_from_payload(bad_payload)


# ==================== group 2: hybrid_search ====================


def _scored_point(chunk_id: str, score: float, page: int = 1) -> MagicMock:
    """A MagicMock that walks like a qdrant_client ScoredPoint."""
    return MagicMock(
        id=chunk_id,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "doc_id": "doc-a",
            "page_numbers": [page],
            "section_path": None,
            "text": f"text body for {chunk_id}",
            "language": "en",
            "char_count": 18 + len(chunk_id),
        },
    )


def _query_points_response(points: list[MagicMock]) -> MagicMock:
    return MagicMock(points=points)


@pytest.fixture(autouse=True)
def _reset_embedder() -> None:
    """Ensure no real model leaks across tests."""
    from common.embedder import BgeM3  # noqa: PLC0415

    BgeM3.reset_for_tests()
    yield
    BgeM3.reset_for_tests()


def test_hybrid_search_returns_fused_results() -> None:
    # Mock embedder
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = ([[0.1] * 1024], [{1: 0.5, 2: 0.3}])

    # Mock Qdrant client with two ranked lists overlapping in the middle
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    dense_points = [
        _scored_point("d#0", 0.95),
        _scored_point("d#1", 0.92),
        _scored_point("d#2", 0.88),
    ]
    sparse_points = [
        _scored_point("d#1", 0.81),  # also in dense list
        _scored_point("d#3", 0.79),
        _scored_point("d#2", 0.77),  # also in dense list
    ]
    # query_points is called twice: once for dense, once for sparse
    client.query_points.side_effect = [
        _query_points_response(dense_points),
        _query_points_response(sparse_points),
    ]

    with (
        patch("common.embedder.BgeM3.get", return_value=mock_encoder),
        patch("qdrant_client.QdrantClient", return_value=client),
    ):
        results = hybrid_search("anything", top_k=3)

    assert len(results) == 3
    # d#1 should be the top result: appears in both lists, both near top.
    assert results[0].chunk.chunk_id == "d#1"
    # Per-signal scores should be populated where each chunk appeared.
    top = results[0]
    assert top.dense_score == pytest.approx(0.92)
    assert top.sparse_score == pytest.approx(0.81)
    # rerank_score must remain unset — that's the next stage's job.
    assert top.rerank_score is None
    # Rank is 1-indexed and matches list position.
    for i, r in enumerate(results, start=1):
        assert r.rank == i


def test_hybrid_search_dense_only_when_sparse_empty() -> None:
    """When the encoded sparse vector has no tokens (degenerate input),
    we still get dense-only results without crashing."""
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = ([[0.0] * 1024], [{}])

    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    dense_points = [_scored_point("d#0", 0.9), _scored_point("d#1", 0.8)]
    client.query_points.return_value = _query_points_response(dense_points)

    with (
        patch("common.embedder.BgeM3.get", return_value=mock_encoder),
        patch("qdrant_client.QdrantClient", return_value=client),
    ):
        results = hybrid_search("x", top_k=2)

    assert [r.chunk.chunk_id for r in results] == ["d#0", "d#1"]
    # Only dense_score should be set; sparse_score stays None.
    assert all(r.sparse_score is None for r in results)
    assert all(r.dense_score is not None for r in results)


def test_hybrid_search_hard_fails_on_qdrant_unreachable() -> None:
    """Qdrant connection failure must surface as QdrantUnavailableError,
    not as a silent empty result list."""
    with (
        patch("qdrant_client.QdrantClient", side_effect=ConnectionRefusedError("nope")),
        pytest.raises(QdrantUnavailableError),
    ):
        hybrid_search("x")


# Keep import alive for ruff
assert hybrid_mod is not None
