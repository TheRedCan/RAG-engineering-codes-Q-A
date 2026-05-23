"""Exception hierarchy for the project.

Design goals:

1. **No silent failures.** Every failure mode in the pipeline has a named
   exception type. Callers handle specific types; bare `except` is banned by ruff.
2. **Layered.** Each pipeline stage has its own base class so callers can catch
   "any ingest failure" or "any retrieval failure" without catching unrelated
   errors.
3. **Carry context.** Every exception carries structured fields (not just a
   string message) so logs and tests can assert on them.
"""

from __future__ import annotations

from dataclasses import dataclass


class EngineeringCodesRagError(Exception):
    """Root of the project's exception hierarchy. Never raise this directly."""


# -------------------- Configuration / environment --------------------


class ConfigError(EngineeringCodesRagError):
    """Invalid or missing configuration (env var, file, CLI arg)."""


class HealthCheckError(EngineeringCodesRagError):
    """An external dependency (Qdrant, Ollama, model file) is not reachable
    or not in the expected state. Raised at startup to fail fast rather than
    at first query."""


# -------------------- Ingest stage --------------------


class IngestError(EngineeringCodesRagError):
    """Base class for any failure inside the ingest pipeline."""


@dataclass
class FetchError(IngestError):
    """A document download from a manifest URL failed."""

    url: str
    status: int | None
    reason: str

    def __str__(self) -> str:
        return f"fetch failed: url={self.url} status={self.status} reason={self.reason}"


@dataclass
class ChecksumMismatchError(IngestError):
    """A downloaded file's SHA-256 did not match the manifest. Treated as a
    hard security failure: the file is deleted and ingestion aborts."""

    path: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return (
            f"checksum mismatch: path={self.path} "
            f"expected={self.expected[:12]}... actual={self.actual[:12]}..."
        )


@dataclass
class ParseError(IngestError):
    """PDF parsing produced no usable output or violated invariants
    (e.g. zero pages, zero text extracted)."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"parse failed: path={self.path} reason={self.reason}"


@dataclass
class LanguageMismatchError(IngestError):
    """A document's detected language disagrees with the manifest. We do not
    silently relabel — the user must update the manifest or remove the file."""

    path: str
    expected: str
    detected: str

    def __str__(self) -> str:
        return (
            f"language mismatch: path={self.path} expected={self.expected} detected={self.detected}"
        )


# -------------------- Retrieval stage --------------------


class RetrievalError(EngineeringCodesRagError):
    """Base class for retrieval-side failures."""


@dataclass
class IndexNotFoundError(RetrievalError):
    """The configured Qdrant collection does not exist yet."""

    collection: str


@dataclass
class EmptyRetrievalError(RetrievalError):
    """Retrieval returned zero candidates for a query. This is not always a
    bug (the corpus may not cover the topic) but the caller must decide how
    to surface it to the user — we do not silently fall back to ungrounded
    generation."""

    query: str


# -------------------- Generation stage --------------------


class GenerationError(EngineeringCodesRagError):
    """Base class for LLM-side failures."""


@dataclass
class LlmUnavailableError(GenerationError):
    """The Ollama server is unreachable or the configured model is not loaded."""

    host: str
    model: str
    reason: str


@dataclass
class CitationVerificationError(GenerationError):
    """A claim in the model's answer could not be grounded in any retrieved
    chunk. The answer is rejected rather than returned with bogus citations."""

    claim: str
    candidate_chunk_ids: list[str]
