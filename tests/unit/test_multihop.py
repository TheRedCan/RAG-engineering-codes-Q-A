"""Unit tests for retrieval.multihop.

Two layers:

1. Pure regex tests for the reference extractor — no Qdrant, no model.
2. ``multihop_search`` orchestration with a mocked ``hybrid_search``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from common.models import Chunk, Language, RetrievedChunk
from retrieval.multihop import (
    _ref_to_query,
    extract_references,
    multihop_search,
    references_in_chunks,
)

# ==================== group 1: regex extractor ====================


def test_extract_section_sign() -> None:
    refs = extract_references("Per §22.5.1 the wall shall...")
    assert refs == [("section", "22.5.1")]


def test_extract_section_word() -> None:
    refs = extract_references("Refer to Section 22.5.1 for details.")
    assert refs == [("section", "22.5.1")]


def test_extract_section_abbrev() -> None:
    refs = extract_references("See Sec. 4-3 for definitions.")
    assert refs == [("section", "4-3")]


def test_extract_chapter() -> None:
    refs = extract_references("Chapter 5 contains the load combinations.")
    assert refs == [("chapter", "5")]


def test_extract_table_and_figure() -> None:
    refs = extract_references("From Table 4-1 and Figure 3.2 we observe...")
    assert refs == [("table", "4-1"), ("figure", "3.2")]


def test_extract_equation_with_parens() -> None:
    refs = extract_references("Substituting into Eq. (2-3) yields...")
    assert refs == [("eq", "2-3")]


def test_extract_arabic_section() -> None:
    """Bring-your-own ECP/SBC will use Arabic terms. The patterns are
    in place even though the v0.1 corpus is English-only."""
    refs = extract_references("راجع البند 22-5-1 لمزيد من التفاصيل.")
    # ASCII digits inside Arabic text are common in code docs.
    assert ("ar_section", "22-5-1") in refs


def test_extract_dedupes_across_kinds() -> None:
    """The same reference written in two styles collapses to one entry."""
    text = "Section 22.5.1 applies here. Per §22.5.1 we further require..."
    refs = extract_references(text)
    assert refs == [("section", "22.5.1")]


def test_extract_ignores_bare_numbers() -> None:
    """'page 5' or arbitrary digits are NOT references."""
    refs = extract_references("On page 5 we see the value 12.3 reported.")
    # 'Page' is not in our catalogue so neither '5' nor '12.3' should match.
    # Some patterns might naively match '12.3' as a section number; verify
    # they require a kind keyword.
    assert ("section", "5") not in refs
    assert ("section", "12.3") not in refs
    assert ("chapter", "5") not in refs


def test_extract_preserves_first_seen_order_per_kind() -> None:
    text = "See Section 7.2 then Table 1-1 then Section 3.4 then Table 1-1."
    refs = extract_references(text)
    # section 7.2 first (within section kind), table 1-1 (only once)
    assert refs == [("section", "7.2"), ("section", "3.4"), ("table", "1-1")]


def test_references_in_chunks_dedupes_across_chunks() -> None:
    def _mk(text: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=Chunk(
                chunk_id="x#0",
                doc_id="x",
                page_numbers=[1],
                section_path=None,
                text=text,
                language=Language.EN,
                char_count=len(text),
            ),
            rank=1,
        )

    chunks = [_mk("See §4.1 and Section 5.2."), _mk("Per §4.1 and Table 9-9.")]
    refs = references_in_chunks(chunks)
    assert refs == [("section", "4.1"), ("section", "5.2"), ("table", "9-9")]


def test_ref_to_query() -> None:
    assert _ref_to_query(("section", "22.5.1")) == "section 22.5.1"
    assert _ref_to_query(("chapter", "5")) == "chapter 5"
    assert _ref_to_query(("table", "4-1")) == "table 4-1"
    assert _ref_to_query(("ar_section", "22-5-1")) == "البند 22-5-1"


# ==================== group 2: multihop_search orchestration ====================


def _retrieved(chunk_id: str, text: str, rank: int = 1) -> RetrievedChunk:
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
        dense_score=0.5,
        sparse_score=0.2,
        rerank_score=None,
        rank=rank,
    )


def test_multihop_returns_hop_zero_when_no_references_found() -> None:
    """A chunk with no detectable cross-references stops after hop 0."""
    hop0 = [_retrieved("c#0", "Plain text with no references.", rank=1)]

    with patch("retrieval.multihop.hybrid_search", return_value=hop0) as m:
        out = multihop_search("query", max_hops=3)

    assert len(out) == 1
    assert out[0].source_hop == 0
    # Only one call to hybrid_search (the hop 0 call).
    assert m.call_count == 1


def test_multihop_follows_one_reference() -> None:
    """A chunk that says 'See §22.5.1' triggers a second hop that
    surfaces a new chunk; the union is returned with source_hop set."""
    hop0 = [_retrieved("c#0", "Refer to Section 22.5.1 for the formula.", rank=1)]
    hop1 = [_retrieved("c#1", "Section 22.5.1 — content of the referenced section.", rank=1)]

    with patch("retrieval.multihop.hybrid_search", side_effect=[hop0, hop1]) as m:
        out = multihop_search("query", max_hops=3)

    assert [r.chunk.chunk_id for r in out] == ["c#0", "c#1"]
    assert [r.source_hop for r in out] == [0, 1]
    # Hop-1 chunks must have their non-rerank scores cleared (they were
    # scored against the ref-query, not the original query).
    assert out[1].dense_score is None
    assert out[1].sparse_score is None
    assert m.call_count == 2


def test_multihop_does_not_revisit_seen_chunks() -> None:
    """If a hop-1 ref-query returns the same chunk as hop 0, we drop it."""
    hop0 = [_retrieved("c#0", "Refer to Section 1.1.", rank=1)]
    # Ref-query for "section 1.1" returns the SAME chunk as hop 0 (which
    # can happen since the original chunk mentions the section).
    hop1 = [_retrieved("c#0", "...", rank=1)]

    with patch("retrieval.multihop.hybrid_search", side_effect=[hop0, hop1]):
        out = multihop_search("query", max_hops=3)

    assert len(out) == 1  # no duplicate added
    assert out[0].chunk.chunk_id == "c#0"


def test_multihop_respects_max_hops() -> None:
    """max_hops caps the total number of hops (including hop 0)."""
    # Every hop's chunks contain a new reference; without the cap we'd
    # recurse forever. With max_hops=2 we should call hybrid_search exactly
    # twice (hop 0 + one ref hop).
    hop0 = [_retrieved("c#0", "See §1.1.", rank=1)]
    hop1 = [_retrieved("c#1", "See §2.2.", rank=1)]
    # Spy is allowed extra calls but we'll assert it stopped early.
    hop2 = [_retrieved("c#2", "See §3.3.", rank=1)]

    with patch("retrieval.multihop.hybrid_search", side_effect=[hop0, hop1, hop2]) as m:
        out = multihop_search("query", max_hops=2)

    assert m.call_count == 2  # hop 0 + 1 follow-up
    assert [r.source_hop for r in out] == [0, 1]


def test_multihop_max_hops_one_equals_plain_hybrid() -> None:
    """max_hops=1 means no follow-up — just hybrid."""
    hop0 = [_retrieved("c#0", "See §1.1.", rank=1)]
    with patch("retrieval.multihop.hybrid_search", return_value=hop0) as m:
        out = multihop_search("query", max_hops=1)
    assert m.call_count == 1
    assert [r.source_hop for r in out] == [0]


def test_multihop_invalid_max_hops_raises() -> None:
    with pytest.raises(ValueError, match="max_hops must be >= 1"):
        multihop_search("query", max_hops=0)
