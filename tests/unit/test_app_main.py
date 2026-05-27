"""Unit tests for app.main's pure helpers.

The Streamlit-rendering code isn't unit-tested — it needs a live
Streamlit runtime. Manual smoke + screenshots cover that. What IS
testable: the citation-deduplication helper, which feeds the sources
expander and has a real risk of subtle off-by-one / hashing bugs as
the Citation model evolves.
"""

from __future__ import annotations

from app.main import _LANG_LABEL, _unique_citation_rows
from common.models import Answer, Citation, Claim, Language


def _answer(claims: list[Claim]) -> Answer:
    used = sorted({c.chunk_id for cl in claims for c in cl.citations})
    return Answer(
        question="Q?",
        answer_language=Language.EN,
        claims=claims,
        used_chunks=used,
        hop_count=1,
    )


def _citation(chunk_id: str, doc_id: str, pages: list[int], section: str | None = None) -> Citation:
    return Citation(chunk_id=chunk_id, doc_id=doc_id, page_numbers=pages, section_path=section)


def test_lang_label_covers_every_language_enum() -> None:
    """No Language value should fall through to .value as the chip label —
    every supported enum has a curated short label."""
    for lang in Language:
        assert lang in _LANG_LABEL


def test_unique_citation_rows_empty_answer_yields_no_rows() -> None:
    assert _unique_citation_rows(_answer([])) == []


def test_unique_citation_rows_preserves_first_appearance_order() -> None:
    """Two claims, each citing a distinct chunk in distinct docs."""
    answer = _answer(
        [
            Claim(text="A", citations=[_citation("c1", "fema-x", [10, 11])]),
            Claim(text="B", citations=[_citation("c2", "nist-y", [3])]),
        ]
    )
    rows = _unique_citation_rows(answer)
    assert rows == [
        "- **fema-x** p.10, 11",
        "- **nist-y** p.3",
    ]


def test_unique_citation_rows_dedupes_same_doc_pages_across_claims() -> None:
    """A repeated citation must collapse — the sources expander should
    never show the same chunk twice even if multiple claims cited it."""
    answer = _answer(
        [
            Claim(text="A", citations=[_citation("c1", "fema-x", [10])]),
            Claim(text="B", citations=[_citation("c1-dup", "fema-x", [10])]),
        ]
    )
    rows = _unique_citation_rows(answer)
    assert rows == ["- **fema-x** p.10"]


def test_unique_citation_rows_distinguishes_pages_within_same_doc() -> None:
    """Same doc, different page ranges = different rows."""
    answer = _answer(
        [
            Claim(text="A", citations=[_citation("c1", "fema-x", [10])]),
            Claim(text="B", citations=[_citation("c2", "fema-x", [22])]),
        ]
    )
    rows = _unique_citation_rows(answer)
    assert rows == ["- **fema-x** p.10", "- **fema-x** p.22"]


def test_unique_citation_rows_includes_section_path_when_present() -> None:
    answer = _answer([Claim(text="A", citations=[_citation("c1", "fema-x", [10], "§12.8.1")])])
    rows = _unique_citation_rows(answer)
    assert rows == ["- **fema-x** p.10 — §12.8.1"]


def test_unique_citation_rows_distinguishes_by_section_path() -> None:
    """Same doc + same pages but different section = different chunk =
    distinct row. (Real corpus has this when one page spans two sections.)"""
    answer = _answer(
        [
            Claim(text="A", citations=[_citation("c1", "fema-x", [10], "§12.8.1")]),
            Claim(text="B", citations=[_citation("c2", "fema-x", [10], "§12.8.2")]),
        ]
    )
    rows = _unique_citation_rows(answer)
    assert len(rows) == 2
    assert rows[0].endswith("§12.8.1")
    assert rows[1].endswith("§12.8.2")
