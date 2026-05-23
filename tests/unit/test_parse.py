"""Unit tests for ingest.parse.

Split into three groups:

1. Pure-function tests for the language helpers — no I/O, no PDFs.
2. ``parse_pdf`` tests against tiny PDFs generated on the fly via fpdf2.
3. ``parse_all`` end-to-end tests against a tmp manifest + generated PDFs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from common.errors import LanguageMismatchError, ParseError
from common.language import detect_language
from common.manifest import manifest_path
from common.models import CodeFamily, DocumentMeta, Language, ParsedPage
from ingest import parse as parse_mod
from ingest.parse import (
    _document_majority_language,
    _verify_language,
    parse_all,
    parse_pdf,
    write_jsonl,
)

# --- enough English to comfortably exceed _MIN_CHARS_FOR_DETECTION ---
_EN_TEXT = (
    "Seismic design considerations for new buildings include load combinations, "
    "site characterization, and structural irregularities. This document outlines "
    "the recommended provisions."
)
# Arabic sample (transliterating: "This is text in the Arabic language for testing")
_AR_TEXT = "هذا نص باللغة العربية لأغراض الاختبار. النص طويل بما يكفي لكشف اللغة بشكل موثوق."


# ==================== group 1: pure helpers ====================


def testdetect_language_english() -> None:
    assert detect_language(_EN_TEXT) == Language.EN


def testdetect_language_arabic() -> None:
    assert detect_language(_AR_TEXT) == Language.AR


def testdetect_language_too_short_returns_unknown() -> None:
    assert detect_language("hi") == Language.UNKNOWN
    assert detect_language("   ") == Language.UNKNOWN
    assert detect_language("") == Language.UNKNOWN


def testdetect_language_unsupported_returns_unknown() -> None:
    # French — not in our supported set.
    fr = (
        "Les considérations de conception sismique pour les nouveaux bâtiments "
        "comprennent les combinaisons de charges et la caractérisation du site."
    )
    assert detect_language(fr) == Language.UNKNOWN


def test_document_majority_language_picks_most_common() -> None:
    pages = [
        ParsedPage(doc_id="x", page_number=1, text="a", detected_language=Language.EN),
        ParsedPage(doc_id="x", page_number=2, text="b", detected_language=Language.EN),
        ParsedPage(doc_id="x", page_number=3, text="c", detected_language=Language.AR),
    ]
    assert _document_majority_language(pages) == Language.EN


def test_document_majority_ignores_unknown() -> None:
    pages = [
        ParsedPage(doc_id="x", page_number=1, text="a", detected_language=Language.UNKNOWN),
        ParsedPage(doc_id="x", page_number=2, text="b", detected_language=Language.UNKNOWN),
        ParsedPage(doc_id="x", page_number=3, text="c", detected_language=Language.EN),
    ]
    assert _document_majority_language(pages) == Language.EN


def test_document_majority_all_unknown_returns_unknown() -> None:
    pages = [
        ParsedPage(doc_id="x", page_number=1, text="a", detected_language=Language.UNKNOWN),
    ]
    assert _document_majority_language(pages) == Language.UNKNOWN


def _meta(expected: Language) -> DocumentMeta:
    return DocumentMeta(
        doc_id="d1",
        family=CodeFamily.FEMA,
        code_number="X",
        edition_year=2024,
        variant=None,
        title_en="t",
        title_ar=None,
        source_url="https://example.test/x.pdf",
        expected_language=expected,
        sha256=None,
        license_note="test",
    )


def test_verify_language_matches() -> None:
    pages = [ParsedPage(doc_id="d1", page_number=1, text="t", detected_language=Language.EN)]
    _verify_language(pages, _meta(Language.EN))  # no raise


def test_verify_language_mismatch_raises() -> None:
    pages = [ParsedPage(doc_id="d1", page_number=1, text="t", detected_language=Language.AR)]
    with pytest.raises(LanguageMismatchError) as ei:
        _verify_language(pages, _meta(Language.EN))
    assert ei.value.expected == "en"
    assert ei.value.detected == "ar"


def test_verify_language_mixed_accepts_anything() -> None:
    pages = [
        ParsedPage(doc_id="d1", page_number=1, text="t", detected_language=Language.AR),
        ParsedPage(doc_id="d1", page_number=2, text="t", detected_language=Language.EN),
    ]
    _verify_language(pages, _meta(Language.MIXED))  # no raise


def test_verify_language_undetectable_does_not_raise() -> None:
    """A PDF with no detectable text shouldn't claim a language mismatch —
    we couldn't actually verify either way."""
    pages = [ParsedPage(doc_id="d1", page_number=1, text="", detected_language=Language.UNKNOWN)]
    _verify_language(pages, _meta(Language.EN))  # no raise


# ==================== group 2: parse_pdf against real PDFs ====================


def test_parse_pdf_yields_one_record_per_page(
    make_pdf: Callable[[str, list[str]], Path],
) -> None:
    pdf_path = make_pdf("doc.pdf", [_EN_TEXT, _EN_TEXT + " more content"])
    pages = list(parse_pdf(pdf_path, "doc"))

    assert len(pages) == 2
    assert all(isinstance(p, ParsedPage) for p in pages)
    assert [p.page_number for p in pages] == [1, 2]
    assert all(p.doc_id == "doc" for p in pages)
    assert all(p.detected_language == Language.EN for p in pages)
    assert all(len(p.text) > 0 for p in pages)


def test_parse_pdf_missing_file_raises_parse_error(tmp_path: Path) -> None:
    with pytest.raises(ParseError) as ei:
        list(parse_pdf(tmp_path / "does-not-exist.pdf", "missing"))
    assert "could not open" in ei.value.reason


def test_write_jsonl_round_trips(tmp_path: Path) -> None:
    pages = [
        ParsedPage(doc_id="d", page_number=1, text="hello", detected_language=Language.EN),
        ParsedPage(doc_id="d", page_number=2, text="world", detected_language=Language.EN),
    ]
    out = tmp_path / "out.jsonl"
    n = write_jsonl(pages, out)

    assert n == 2
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    reloaded = [ParsedPage.model_validate_json(line) for line in lines]
    assert reloaded == pages


# ==================== group 3: parse_all end-to-end ====================


def _write_manifest_entry(
    raw_dir: Path,
    *,
    doc_id: str,
    expected_language: str,
    family: str = "fema",
) -> None:
    entry = {
        "doc_id": doc_id,
        "family": family,
        "code_number": "TEST",
        "edition_year": 2024,
        "variant": None,
        "title_en": f"Test {doc_id}",
        "title_ar": None,
        "source_url": "https://example.test/x.pdf",
        "expected_language": expected_language,
        "sha256": None,
        "license_note": "test fixture",
    }
    manifest_path(raw_dir).write_text(json.dumps([entry]), encoding="utf-8")


def test_parse_all_writes_jsonl_for_each_doc(
    raw_dir: Path,
    processed_dir: Path,
    make_pdf: Callable[[str, list[str]], Path],
) -> None:
    _write_manifest_entry(raw_dir, doc_id="doc-a", expected_language="en")
    pdf = make_pdf("source.pdf", [_EN_TEXT, _EN_TEXT])
    (raw_dir / "doc-a.pdf").write_bytes(pdf.read_bytes())

    result = parse_all()

    assert result.all_succeeded
    out = processed_dir / "parsed" / "doc-a.jsonl"
    assert out.is_file()
    records = [ParsedPage.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(records) == 2


def test_parse_all_soft_fails_on_missing_pdf(raw_dir: Path) -> None:
    _write_manifest_entry(raw_dir, doc_id="ghost", expected_language="en")

    result = parse_all()

    assert not result.all_succeeded
    assert result.succeeded == []
    assert len(result.failed) == 1
    failed_id, reason = result.failed[0]
    assert failed_id == "ghost"
    assert "PDF not found" in reason


def test_parse_all_soft_fails_on_language_mismatch(
    raw_dir: Path,
    make_pdf: Callable[[str, list[str]], Path],
) -> None:
    _write_manifest_entry(raw_dir, doc_id="wrong-lang", expected_language="ar")
    pdf = make_pdf("source.pdf", [_EN_TEXT])
    (raw_dir / "wrong-lang.pdf").write_bytes(pdf.read_bytes())

    result = parse_all()

    assert len(result.failed) == 1
    failed_id, reason = result.failed[0]
    assert failed_id == "wrong-lang"
    assert "language mismatch" in reason


def test_parse_all_skips_existing_output_without_force(
    raw_dir: Path,
    processed_dir: Path,
    make_pdf: Callable[[str, list[str]], Path],
) -> None:
    _write_manifest_entry(raw_dir, doc_id="doc-b", expected_language="en")
    pdf = make_pdf("source.pdf", [_EN_TEXT])
    (raw_dir / "doc-b.pdf").write_bytes(pdf.read_bytes())

    # First parse populates the output.
    parse_all()
    out = processed_dir / "parsed" / "doc-b.jsonl"
    first_mtime = out.stat().st_mtime_ns

    # Second parse without --force should NOT touch the file.
    parse_all()
    assert out.stat().st_mtime_ns == first_mtime

    # With force=True it does re-parse and rewrite.
    parse_all(force=True)
    assert out.stat().st_mtime_ns >= first_mtime


def test_parse_all_only_doc_id_filters(
    raw_dir: Path,
    processed_dir: Path,
    make_pdf: Callable[[str, list[str]], Path],
) -> None:
    # Two entries in the manifest, only one PDF on disk.
    entries = [
        {
            "doc_id": "wanted",
            "family": "fema",
            "code_number": "T",
            "edition_year": 2024,
            "variant": None,
            "title_en": "wanted",
            "title_ar": None,
            "source_url": "https://example.test/x.pdf",
            "expected_language": "en",
            "sha256": None,
            "license_note": "test",
        },
        {
            "doc_id": "ignored",
            "family": "fema",
            "code_number": "T",
            "edition_year": 2024,
            "variant": None,
            "title_en": "ignored",
            "title_ar": None,
            "source_url": "https://example.test/x.pdf",
            "expected_language": "en",
            "sha256": None,
            "license_note": "test",
        },
    ]
    manifest_path(raw_dir).write_text(json.dumps(entries), encoding="utf-8")
    pdf = make_pdf("source.pdf", [_EN_TEXT])
    (raw_dir / "wanted.pdf").write_bytes(pdf.read_bytes())

    result = parse_all(only_doc_id="wanted")

    assert result.all_succeeded
    assert len(result.succeeded) == 1
    assert (processed_dir / "parsed" / "wanted.jsonl").is_file()
    # The other doc was never attempted, so no error and no output.
    assert not (processed_dir / "parsed" / "ignored.jsonl").exists()
    assert result.failed == []


# A reference to the module-under-test that ruff won't strip as unused.
assert parse_mod is not None
