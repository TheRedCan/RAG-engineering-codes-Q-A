"""Pydantic models describing the data that flows between pipeline stages.

The dataflow is one-way and serialized to JSONL between stages so any stage
can be re-run in isolation against the prior stage's output:

    fetch (manifest.json)      -> data/raw/*.pdf
    parse  (PDF)               -> data/processed/parsed/*.jsonl   [ParsedPage]
    chunk  (ParsedPage)        -> data/processed/chunks/*.jsonl   [Chunk]
    embed  (Chunk)             -> Qdrant points (+ chunks.jsonl unchanged)
    retrieve (query string)    -> list[RetrievedChunk]
    rerank   (RetrievedChunk)  -> list[RetrievedChunk] (re-scored, pruned)
    generate (RetrievedChunk)  -> Answer

If any field needed by a downstream stage is missing, validation fails
**at the boundary**, not deep inside the consumer.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Language(StrEnum):
    """ISO 639-1 codes we support."""

    AR = "ar"
    EN = "en"
    MIXED = "mixed"  # bilingual PDFs (e.g. Arabic body with English code refs) — split downstream
    UNKNOWN = "unknown"


class CodeFamily(StrEnum):
    """Which code/standard family a document belongs to. Extensible later.

    The v0.1 public demo corpus is FEMA + NIST. USACE is included in the
    enum because we tested its hosting and may add documents later; it is
    not currently in the shipped manifest. SBC and ECP are supported as
    bring-your-own: users with legitimate access drop their PDFs into
    data/raw/ and add a manifest entry. Neither code is redistributed by
    this project.
    """

    # === public-domain US gov, freely downloadable, English ===
    FEMA = "fema"
    NIST = "nist"
    USACE = "usace"  # reserved; no docs in v0.1 manifest
    # === bring-your-own (not redistributed) ===
    SBC = "sbc"  # Saudi Building Code
    ECP = "ecp"  # Egyptian Code of Practice
    OTHER = "other"


class DocumentMeta(BaseModel):
    """One entry in the manifest. Identifies a source PDF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(description="Stable slug, e.g. 'fema-p-2082-vol1-2020'.")
    family: CodeFamily
    code_number: str = Field(
        description="Code/standard identifier, e.g. 'P-2082', 'EM-1110-2-2100', 'TN-2209', '304'."
    )
    edition_year: int
    variant: str | None = Field(
        default=None,
        description=(
            "Optional sub-document label. For SBC: 'CR' or 'CC'. "
            "For multi-volume FEMA docs: 'Vol1', 'Vol2'. Leave None for single-volume docs."
        ),
    )
    title_en: str
    title_ar: str | None = None
    source_url: str = Field(description="Official download URL.")
    expected_language: Language
    sha256: str | None = Field(
        default=None,
        description="Expected SHA-256 of the PDF. None means 'pin on first download'.",
    )
    license_note: str = Field(description="Copyright/usage note shown to the user before download.")


class ParsedPage(BaseModel):
    """One page extracted from a source PDF. Stored in JSONL."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page_number: int = Field(ge=1)
    text: str
    detected_language: Language


class Chunk(BaseModel):
    """A retrieval unit. Stored in JSONL and indexed in Qdrant."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Stable hash-based id; unique across corpus.")
    doc_id: str
    page_numbers: list[int] = Field(min_length=1)
    section_path: str | None = Field(default=None, description="Breadcrumb like 'Ch.22 > §22.5.1'.")
    text: str
    language: Language
    char_count: int = Field(ge=1)


class RetrievedChunk(BaseModel):
    """A chunk plus retrieval scores. Output of retrieve and rerank stages."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    rank: int = Field(ge=1)


class Citation(BaseModel):
    """One citation backing a claim in the final answer."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    page_numbers: list[int]
    section_path: str | None = None


class Claim(BaseModel):
    """One factual claim in the answer, with the chunks it is grounded in."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[Citation] = Field(min_length=1)  # enforced: no uncited claims


class Answer(BaseModel):
    """Final structured answer returned to the user / UI."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer_language: Language
    claims: list[Claim]
    used_chunks: list[str] = Field(description="All chunk_ids consulted.")
    hop_count: int = Field(ge=1)


# -------------------- helpers --------------------


def doc_id_path(processed_dir: Path, stage: str, doc_id: str) -> Path:
    """Convention for where each stage writes its per-doc JSONL output."""
    return processed_dir / stage / f"{doc_id}.jsonl"
