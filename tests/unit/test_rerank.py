"""Unit tests for retrieval.rerank."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from common.models import Chunk, Language, RetrievedChunk
from retrieval.rerank import (
    _MAX_BONUS,
    BgeReranker,
    _equation_bonus,
    _is_direct_question,
    rerank,
)


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


# ==================== direct-question boost ====================


@pytest.mark.parametrize(
    "query",
    [
        "How is Cs calculated?",
        "how do I calculate the seismic response coefficient",
        "How do we determine the base shear?",
        "What is the equation for Cs?",
        "What is the value of SDS?",
        "What does Section 12.8.1 require?",
        "What are the load combinations for seismic design?",
        "What are the requirements for diaphragm design?",
        "كيف يتم حساب معامل الاستجابة الزلزالية؟",
        "ما هي معادلة Cs؟",
    ],
)
def test_is_direct_question_recognises_calc_and_requirement_queries(query: str) -> None:
    assert _is_direct_question(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What is soil-structure interaction?",
        "Explain the equivalent lateral force procedure",
        "Why is Cs important?",
        "Tell me about seismic design",
        "How does base isolation work conceptually?",
    ],
)
def test_is_direct_question_skips_conceptual_queries(query: str) -> None:
    """Conceptual queries must NOT trigger the boost — otherwise every
    explanation would get pulled toward equation-bearing chunks."""
    assert _is_direct_question(query) is False


def test_equation_bonus_is_zero_for_plain_narrative() -> None:
    text = (
        "Soil-structure interaction refers to the way a structure and "
        "its supporting soil interact under dynamic loading conditions."
    )
    assert _equation_bonus(text) == 0.0


def test_equation_bonus_rewards_chunks_with_eq_references() -> None:
    text = "Cs is computed from Eq. 12.8-2 as a function of SDS and R."
    bonus = _equation_bonus(text)
    assert bonus > 0
    assert bonus <= _MAX_BONUS


def test_equation_bonus_rewards_chunks_with_assignments() -> None:
    text = "For this example, SDS = 1.25, R = 6.5, Ie = 1.00, Cs = 0.192."
    assert _equation_bonus(text) > 0


def test_equation_bonus_is_capped() -> None:
    """A chunk packed with every kind of marker shouldn't get unbounded bonus."""
    text = (
        "Eq. 12.8-2 references Table 12.8-1 under Section 12.8.1. "
        "SDS = 1.25, R = 6.5, Cs = 0.192, V = 100 kips."
    )
    assert _equation_bonus(text) == _MAX_BONUS


def test_rerank_boost_swaps_near_tied_chunks_for_direct_questions() -> None:
    """A narrative chunk that *just* edges out an equation chunk in
    cross-encoder score should lose to it after the direct-question boost."""
    narrative = _make_candidate(
        "narrative",
        1,
        text="Cs is the seismic response coefficient, a key parameter in design.",
    )
    equation = _make_candidate(
        "equation",
        2,
        text="Cs = SDS / (R/Ie) per Eq. 12.8-2 with SDS = 1.25.",
    )
    # Narrative scores 0.5 higher than equation pre-boost — small lead.
    mock_model = MagicMock()
    mock_model.score.return_value = [2.5, 2.0]

    with patch("retrieval.rerank.BgeReranker.get", return_value=mock_model):
        out = rerank("How is Cs calculated?", [narrative, equation], top_k=2)

    # After the equation chunk's bonus (~1.5), it beats the narrative chunk.
    assert [r.chunk.chunk_id for r in out] == ["equation", "narrative"]


def test_rerank_boost_does_not_apply_to_conceptual_queries() -> None:
    """For 'what is X' conceptual queries the boost must NOT fire — we
    don't want equation chunks crowding out explanatory chunks."""
    narrative = _make_candidate(
        "narrative", 1, text="SSI is the interaction between structure and soil."
    )
    equation = _make_candidate("equation", 2, text="Some Eq. 12.8-2 SDS = 1.25 stuff.")
    mock_model = MagicMock()
    mock_model.score.return_value = [2.5, 2.0]

    with patch("retrieval.rerank.BgeReranker.get", return_value=mock_model):
        out = rerank("What is soil-structure interaction?", [narrative, equation], top_k=2)

    # Original cross-encoder order preserved; no boost applied.
    assert [r.chunk.chunk_id for r in out] == ["narrative", "equation"]


def test_rerank_boost_cannot_pull_clear_loser_to_the_top() -> None:
    """The boost is small and additive — it must not overwhelm a clear
    cross-encoder winner. A chunk that scores 5.0 lower must stay below
    even with the maximum bonus (1.5)."""
    narrative = _make_candidate(
        "narrative", 1, text="High-relevance narrative about Cs calculations."
    )
    equation = _make_candidate(
        "equation", 2, text="Cs = SDS / (R/Ie) Eq. 12.8-2 Table 12.8-1 § 12.8.1"
    )
    mock_model = MagicMock()
    mock_model.score.return_value = [5.0, 0.0]  # 5.0 gap > _MAX_BONUS

    with patch("retrieval.rerank.BgeReranker.get", return_value=mock_model):
        out = rerank("How is Cs calculated?", [narrative, equation], top_k=2)

    assert out[0].chunk.chunk_id == "narrative"
