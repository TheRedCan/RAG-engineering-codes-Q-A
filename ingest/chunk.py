"""Split parsed pages into retrieval-ready chunks.

For each document's parsed JSONL (``data/processed/parsed/{doc_id}.jsonl``):

1. Concatenate non-empty pages while tracking each page's char-offset range
   in the concatenated text.
2. Walk a sliding window of ``chunk_target_chars`` with ``chunk_overlap_chars``
   overlap.
3. Snap each window end to the nearest paragraph / sentence / line boundary
   within the last 20% of the window so chunks don't cut mid-sentence.
4. Map each chunk back to the set of page numbers whose offset ranges it
   touches.
5. Re-detect the chunk's language (a chunk could span pages of different
   languages — important once we add bring-your-own Arabic content).
6. Emit ``Chunk`` records to ``data/processed/chunks/{doc_id}.jsonl``.

Out of scope for v0.1 (each is a deliberate later improvement):

- Section heading detection -> ``section_path`` stays None
- Token-based sizing -> char-based for now (BGE-M3's tokenizer would mean
  loading torch at chunk time, which slows iteration)
- Table-aware chunking
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import typer

from common.errors import IngestError
from common.language import detect_language
from common.logging import configure_logging, logger
from common.manifest import load_manifest
from common.models import Chunk, ParsedPage, doc_id_path
from common.settings import get_settings

app = typer.Typer(add_completion=False, help="Chunk parsed PDFs into retrieval units.")


@dataclass
class ChunkResult:
    """Summary of a chunk_all invocation. Same shape as FetchResult and
    ParseResult so a single result-handling helper works across stages."""

    succeeded: list[Path] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


# -------------------- pure helpers --------------------


@dataclass(frozen=True)
class _PageRange:
    """Half-open character range a page occupies in the concatenated document
    text. ``end`` is exclusive."""

    page_number: int
    start: int
    end: int


def _concatenate_pages(pages: list[ParsedPage]) -> tuple[str, list[_PageRange]]:
    """Concatenate non-empty page texts, separated by a blank line, and return
    the concatenated string plus the per-page offset map.

    Empty pages (covers, blank dividers) are silently dropped from the
    concatenation — they contribute no retrievable text and shouldn't pad
    chunk offsets.
    """
    parts: list[str] = []
    ranges: list[_PageRange] = []
    cursor = 0
    separator = "\n\n"

    for p in pages:
        text = p.text
        if not text.strip():
            continue
        start = cursor
        parts.append(text)
        cursor += len(text)
        ranges.append(_PageRange(page_number=p.page_number, start=start, end=cursor))
        # Insert a separator between pages so chunk boundaries align with
        # something more meaningful than a hard concatenation.
        parts.append(separator)
        cursor += len(separator)

    return "".join(parts), ranges


# Boundary characters to snap to, in order of preference. Earlier entries
# produce more semantically coherent splits.
_BOUNDARY_MARKERS: tuple[str, ...] = ("\n\n", ". ", "؟ ", "! ", "؟\n", "!\n", ".\n", "\n")
# Fraction of the target window in which we'll look for a boundary. Outside
# this band we just hard-cut to keep chunk sizes bounded.
_BOUNDARY_SEARCH_FRACTION = 0.2


def _find_snap_boundary(text: str, hard_end: int, soft_window: int) -> int:
    """Return an end-offset <= hard_end that snaps to a natural boundary,
    or ``hard_end`` itself if none is found in the soft window.

    ``soft_window`` is the number of chars before ``hard_end`` we'll search.
    """
    earliest = max(0, hard_end - soft_window)
    region = text[earliest:hard_end]
    best = -1
    for marker in _BOUNDARY_MARKERS:
        idx = region.rfind(marker)
        if idx >= 0:
            best = max(best, earliest + idx + len(marker))
            break  # higher-priority marker wins
    return best if best > 0 else hard_end


def _pages_for_range(ranges: list[_PageRange], start: int, end: int) -> list[int]:
    """Return the page numbers whose offset ranges intersect [start, end).

    Pages are returned in document order. A chunk that spans two pages cites
    both; a chunk fully inside one page cites only that one.
    """
    return [r.page_number for r in ranges if r.start < end and r.end > start]


def _split_to_chunks(
    text: str,
    page_ranges: list[_PageRange],
    *,
    target_chars: int,
    overlap_chars: int,
    min_size_chars: int,
) -> Iterator[tuple[str, list[int]]]:
    """Yield (chunk_text, page_numbers) tuples.

    Invariants enforced:

    - Window never advances by less than 1 char (no infinite loops on tiny text)
    - Trailing chunks shorter than ``min_size_chars`` are dropped to avoid
      polluting the index with stub fragments
    - Page mapping is always non-empty for an emitted chunk (an empty mapping
      would indicate a programming bug, not legitimate output)
    """
    if not text:
        return

    soft_window = max(1, int(target_chars * _BOUNDARY_SEARCH_FRACTION))
    n = len(text)
    start = 0

    while start < n:
        hard_end = min(start + target_chars, n)
        end = _find_snap_boundary(text, hard_end, soft_window) if hard_end < n else n

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= min_size_chars:
            pages = _pages_for_range(page_ranges, start, end)
            if pages:  # defensive: the only way this fails is a bug above
                yield chunk_text, pages

        if end >= n:
            break

        # Advance with overlap. Guard against the overlap pushing us back
        # to or before the previous start (would loop forever on tiny windows).
        next_start = end - overlap_chars
        start = max(start + 1, next_start)


def _make_chunk_id(doc_id: str, index: int) -> str:
    """Stable, debuggable chunk id. Re-chunking the same parsed input
    produces the same ids, so downstream stages that key on chunk_id stay
    coherent across re-runs. Re-parsing or re-chunking with different params
    is expected to invalidate ids — that's correct invalidation."""
    return f"{doc_id}#{index:05d}"


def chunks_from_pages(
    doc_id: str,
    pages: list[ParsedPage],
    *,
    target_chars: int,
    overlap_chars: int,
    min_size_chars: int,
) -> Iterator[Chunk]:
    """Public chunker entry. Produces ``Chunk`` records from a doc's pages."""
    text, ranges = _concatenate_pages(pages)
    for index, (chunk_text, page_numbers) in enumerate(
        _split_to_chunks(
            text,
            ranges,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            min_size_chars=min_size_chars,
        )
    ):
        yield Chunk(
            chunk_id=_make_chunk_id(doc_id, index),
            doc_id=doc_id,
            page_numbers=page_numbers,
            section_path=None,  # populated by a future section-detection pass
            text=chunk_text,
            language=detect_language(chunk_text),
            char_count=len(chunk_text),
        )


# -------------------- I/O --------------------


def _read_parsed(path: Path) -> list[ParsedPage]:
    """Load a parsed JSONL into ``ParsedPage`` records, validating each line."""
    pages: list[ParsedPage] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            pages.append(ParsedPage.model_validate_json(stripped))
    return pages


def write_jsonl(chunks: Iterable[Chunk], path: Path) -> int:
    """Write chunks as JSONL (one Chunk.model_dump_json per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
            count += 1
    return count


# -------------------- public entry --------------------


def chunk_all(*, force: bool = False, only_doc_id: str | None = None) -> ChunkResult:
    """Chunk every parsed document.

    Reads from ``data/processed/parsed/`` and writes to
    ``data/processed/chunks/``. Chunk sizing is controlled by settings
    (``CHUNK_TARGET_CHARS`` etc.) — see ``.env.example``.
    """
    settings = get_settings()
    docs = load_manifest(settings.raw_dir)
    if not docs:
        logger.warning("manifest is empty; nothing to chunk")
        return ChunkResult()

    result = ChunkResult()

    for meta in docs:
        if only_doc_id is not None and meta.doc_id != only_doc_id:
            continue

        parsed_path = doc_id_path(settings.processed_dir, "parsed", meta.doc_id)
        out_path = doc_id_path(settings.processed_dir, "chunks", meta.doc_id)

        if out_path.exists() and not force:
            logger.info(
                f"{meta.doc_id}: chunks already at {out_path}, skipping (use --force to re-chunk)"
            )
            result.succeeded.append(out_path)
            continue

        if not parsed_path.is_file():
            reason = f"parsed JSONL not found at {parsed_path}; run `python -m ingest.parse` first"
            logger.error(f"{meta.doc_id}: {reason}")
            result.failed.append((meta.doc_id, reason))
            continue

        try:
            pages = _read_parsed(parsed_path)
            chunks = list(
                chunks_from_pages(
                    meta.doc_id,
                    pages,
                    target_chars=settings.chunk_target_chars,
                    overlap_chars=settings.chunk_overlap_chars,
                    min_size_chars=settings.chunk_min_size_chars,
                )
            )
            if not chunks:
                raise IngestError(  # noqa: TRY301 — single-call raise, no inner-fn needed
                    f"{meta.doc_id} produced zero chunks from {len(pages)} pages "
                    f"(target_chars={settings.chunk_target_chars})"
                )
            count = write_jsonl(chunks, out_path)
        except (OSError, IngestError, ValueError) as e:
            logger.error(f"{meta.doc_id}: chunk failed — {e}")
            result.failed.append((meta.doc_id, str(e)))
            continue

        lang_counts: dict[str, int] = {}
        for c in chunks:
            lang_counts[c.language.value] = lang_counts.get(c.language.value, 0) + 1
        logger.info(f"{meta.doc_id}: wrote {count} chunks (langs={lang_counts}) -> {out_path}")
        result.succeeded.append(out_path)

    return result


# -------------------- CLI --------------------


@app.command()
def main(
    doc_id: str | None = typer.Option(
        None, "--doc-id", help="Chunk only this doc_id (default: all in manifest)."
    ),
    force: bool = typer.Option(False, "--force", help="Re-chunk even if output exists."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Chunk every parsed document into data/processed/chunks/."""
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    try:
        result = chunk_all(force=force, only_doc_id=doc_id)
    except Exception:  # noqa: BLE001 — top-level CLI catch-all by design (logs traceback, exits 2)
        logger.exception("chunk_all aborted on a hard-fail condition")
        sys.exit(2)

    processed_dir = get_settings().processed_dir
    logger.info(
        f"done. {len(result.succeeded)} succeeded, {len(result.failed)} failed "
        f"in {processed_dir}/chunks/"
    )

    if result.failed:
        logger.error("the following documents could not be chunked:")
        for failed_doc_id, reason in result.failed:
            logger.error(f"  - {failed_doc_id}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    app()
