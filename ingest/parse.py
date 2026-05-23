"""Parse source PDFs into per-page JSONL records.

For each document listed in the manifest:

1. Open the PDF with pypdfium2.
2. Extract text per page (preserving the 1-indexed page number).
3. Detect language per page with langdetect (seeded for reproducibility).
4. Cross-check the document's majority language against the manifest's
   ``expected_language``; mismatch raises ``LanguageMismatchError``.
5. Stream the results to ``data/processed/parsed/{doc_id}.jsonl``.

Failure policy (matches ingest.fetch):

- ``ParseError`` (corrupt PDF, zero pages) — soft-fail per doc, collected
  into ``ParseResult.failed`` so a bad document doesn't abort the run.
- ``LanguageMismatchError`` — same: surfaced explicitly, never silently
  relabeled.
- Missing PDF on disk — soft-fail with a "run fetch first" hint.

This module performs **only parsing and language verification**. Chunking,
table-aware extraction, OCR for scanned pages, and section breadcrumbing
all live in later stages so each module stays small and independently
testable.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pypdfium2 as pdfium
import typer
from pypdfium2 import PdfiumError

from common.errors import LanguageMismatchError, ParseError
from common.language import detect_language
from common.logging import configure_logging, logger
from common.manifest import load_manifest
from common.models import Language, ParsedPage, doc_id_path
from common.settings import get_settings

if TYPE_CHECKING:
    from common.models import DocumentMeta


app = typer.Typer(add_completion=False, help="Parse PDFs into per-page JSONL records.")


@dataclass
class ParseResult:
    """Summary of a parse_all invocation. Same shape as FetchResult so callers
    can write a single result-handling helper later."""

    succeeded: list[Path] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


# -------------------- pure helpers --------------------


def _document_majority_language(pages: list[ParsedPage]) -> Language:
    """Majority language across pages with a detectable language.
    Returns ``Language.UNKNOWN`` if no page had detectable text."""
    detected = [p.detected_language for p in pages if p.detected_language != Language.UNKNOWN]
    if not detected:
        return Language.UNKNOWN
    return Counter(detected).most_common(1)[0][0]


def _verify_language(pages: list[ParsedPage], meta: DocumentMeta) -> None:
    """Raise ``LanguageMismatchError`` if the document's majority language
    disagrees with the manifest's expected language.

    Accepts ``expected_language == MIXED`` as a wildcard. UNKNOWN majority
    (e.g. a graphics-heavy PDF with little extractable text) is not an
    error — we just couldn't verify either way.
    """
    if meta.expected_language == Language.MIXED:
        return
    majority = _document_majority_language(pages)
    if majority == Language.UNKNOWN:
        return
    if majority != meta.expected_language:
        raise LanguageMismatchError(
            path=meta.doc_id,
            expected=meta.expected_language.value,
            detected=majority.value,
        )


# -------------------- PDF I/O --------------------


def parse_pdf(path: Path, doc_id: str) -> Iterator[ParsedPage]:
    """Yield one ``ParsedPage`` per page in the PDF.

    Raises:
        ParseError: if the PDF cannot be opened or has zero pages.
    """
    try:
        pdf = pdfium.PdfDocument(str(path))
    except (PdfiumError, OSError) as e:
        # PdfiumError covers corrupt / encrypted PDFs; OSError (notably
        # FileNotFoundError) covers a missing path. We surface both as ParseError
        # so callers see a single consistent failure mode.
        raise ParseError(path=str(path), reason=f"could not open PDF: {e}") from e

    n_pages = len(pdf)
    if n_pages == 0:
        raise ParseError(path=str(path), reason="document has zero pages")

    for i in range(n_pages):
        page = pdf[i]
        try:
            text = page.get_textpage().get_text_bounded()
        except PdfiumError as e:
            # A single broken page should not kill the whole document.
            # Log and emit an empty-text page so the page index stays in sync.
            logger.warning(f"{doc_id} page {i + 1}: text extraction failed: {e}")
            text = ""

        yield ParsedPage(
            doc_id=doc_id,
            page_number=i + 1,
            text=text,
            detected_language=detect_language(text),
        )


def write_jsonl(pages: Iterable[ParsedPage], path: Path) -> int:
    """Write pages as JSONL (one ParsedPage.model_dump_json per line).
    Returns the number of records written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(p.model_dump_json() + "\n")
            count += 1
    return count


# -------------------- public entry --------------------


def parse_all(*, force: bool = False, only_doc_id: str | None = None) -> ParseResult:
    """Parse every PDF listed in the manifest.

    Args:
        force: re-parse even if the output JSONL already exists.
        only_doc_id: if set, parse only this doc and skip the rest.
    """
    settings = get_settings()
    docs = load_manifest(settings.raw_dir)
    if not docs:
        logger.warning("manifest is empty; nothing to parse")
        return ParseResult()

    result = ParseResult()

    for meta in docs:
        if only_doc_id is not None and meta.doc_id != only_doc_id:
            continue

        pdf_path = settings.raw_dir / f"{meta.doc_id}.pdf"
        out_path = doc_id_path(settings.processed_dir, "parsed", meta.doc_id)

        if out_path.exists() and not force:
            logger.info(
                f"{meta.doc_id}: parsed output already at {out_path}, skipping "
                f"(use --force to re-parse)"
            )
            result.succeeded.append(out_path)
            continue

        if not pdf_path.is_file():
            reason = (
                f"PDF not found at {pdf_path}; "
                f"run `python -m ingest.fetch` first or drop the file in manually"
            )
            logger.error(f"{meta.doc_id}: {reason}")
            result.failed.append((meta.doc_id, reason))
            continue

        try:
            pages = list(parse_pdf(pdf_path, meta.doc_id))
            _verify_language(pages, meta)
            count = write_jsonl(pages, out_path)
        except (ParseError, LanguageMismatchError) as e:
            logger.error(f"{meta.doc_id}: parse failed — {e}")
            result.failed.append((meta.doc_id, str(e)))
            continue

        majority = _document_majority_language(pages)
        logger.info(
            f"{meta.doc_id}: parsed {count} pages "
            f"(majority language={majority.value}) -> {out_path}"
        )
        result.succeeded.append(out_path)

    return result


# -------------------- CLI --------------------


@app.command()
def main(
    doc_id: str | None = typer.Option(
        None, "--doc-id", help="Parse only this doc_id (default: all in manifest)."
    ),
    force: bool = typer.Option(False, "--force", help="Re-parse even if output JSONL exists."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Parse every PDF in the manifest into data/processed/parsed/."""
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    try:
        result = parse_all(force=force, only_doc_id=doc_id)
    except Exception:  # noqa: BLE001 — top-level CLI catch-all by design (logs traceback, exits 2)
        logger.exception("parse_all aborted on a hard-fail condition")
        sys.exit(2)

    processed_dir = get_settings().processed_dir
    logger.info(
        f"done. {len(result.succeeded)} succeeded, {len(result.failed)} failed "
        f"in {processed_dir}/parsed/"
    )

    if result.failed:
        logger.error("the following documents could not be parsed:")
        for failed_doc_id, reason in result.failed:
            logger.error(f"  - {failed_doc_id}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    app()
