"""Embed chunks with BGE-M3 and push them into Qdrant.

For each document's chunks JSONL:

1. Load BGE-M3 once (cached locally at ~/.cache/huggingface/ after first run).
2. Read ``Chunk`` records from ``data/processed/chunks/{doc_id}.jsonl``.
3. Encode in batches, producing for each chunk:
   - a 1024-dim **dense** vector (semantic similarity)
   - a **sparse** vector (learned lexical weights, like a smart BM25)
   ColBERT-style multi-vectors are intentionally OFF in v0.1: they triple
   memory + index size and the gain shows up mostly at the reranker, which
   has its own dedicated model in the rerank stage.
4. Upsert into a Qdrant collection with named vectors ``dense`` + ``sparse``.
5. Per-doc resume: scroll the collection for existing chunk_ids before
   encoding, and skip anything that's already indexed. Re-running embed on
   an already-complete corpus is a no-op (~1 s).

Failure policy:

- Qdrant unreachable at startup -> ``QdrantUnavailableError`` (hard fail).
- Existing collection has a different schema -> ``CollectionSchemaMismatchError``
  (hard fail; user must drop the collection deliberately).
- Per-batch encode / upsert errors -> logged, batch skipped, loop continues;
  the failure is recorded in ``EmbedResult.failed``.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from common.embedder import BgeM3
from common.errors import (
    CollectionSchemaMismatchError,
    EmbedError,
    QdrantUnavailableError,
)
from common.logging import configure_logging, logger
from common.manifest import load_manifest
from common.models import Chunk, doc_id_path
from common.settings import get_settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

# Deterministic UUID namespace for chunk_id -> Qdrant point id.
# Using a fixed UUID5 namespace means the same chunk_id always maps to the
# same point id across machines and re-runs, so upserts are idempotent.
_POINT_ID_NS = uuid.UUID("6b7c1d9e-2c4f-4f6a-9b2d-5e3a8c1d2e6f")

# Names for the two vectors stored per point. Stable strings — changing them
# is a schema break and would require re-embedding.
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"

# Scroll batch size for "what's already indexed" queries.
_SCROLL_BATCH = 1024

app = typer.Typer(add_completion=False, help="Embed chunks into Qdrant.")


@dataclass
class EmbedResult:
    """Same shape as the other ingest-stage results."""

    succeeded: list[str] = field(default_factory=list)  # doc_ids
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)
    # informational counters that help us notice silent regressions
    chunks_indexed: int = 0
    chunks_skipped: int = 0

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


# ==================== Qdrant helpers ====================


def _connect_qdrant() -> QdrantClient:
    """Construct a Qdrant client and run a real handshake. Hard-fails if
    the server is unreachable so we don't waste model load time on a dead
    backend."""
    settings = get_settings()
    # Lazy import keeps unit tests cheap when they don't touch this path.
    from qdrant_client import QdrantClient  # noqa: PLC0415

    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=10)
        client.get_collections()
    except Exception as e:
        raise QdrantUnavailableError(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            reason=str(e),
        ) from e
    return client


def _ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Create the collection if missing; if present, verify the schema we
    rely on. Will not silently migrate a mismatched collection."""
    from qdrant_client.http.models import (  # noqa: PLC0415
        Distance,
        SparseVectorParams,
        VectorParams,
    )

    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        logger.info(f"creating Qdrant collection {name!r} (dense_dim={dense_dim})")
        client.create_collection(
            collection_name=name,
            vectors_config={_DENSE_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)},
            sparse_vectors_config={_SPARSE_NAME: SparseVectorParams()},
        )
        return

    info = client.get_collection(name)
    vectors = info.config.params.vectors or {}
    dense_cfg = vectors.get(_DENSE_NAME) if isinstance(vectors, dict) else None
    if dense_cfg is None:
        raise CollectionSchemaMismatchError(
            collection=name, detail=f"missing named dense vector {_DENSE_NAME!r}"
        )
    if dense_cfg.size != dense_dim:
        raise CollectionSchemaMismatchError(
            collection=name,
            detail=f"dense dim mismatch: stored={dense_cfg.size} expected={dense_dim}",
        )
    sparse = info.config.params.sparse_vectors or {}
    if _SPARSE_NAME not in sparse:
        raise CollectionSchemaMismatchError(
            collection=name, detail=f"missing named sparse vector {_SPARSE_NAME!r}"
        )


def _existing_chunk_ids(client: QdrantClient, collection: str, doc_id: str) -> set[str]:
    """Return the set of chunk_ids already present in the collection for this
    document. Enables granular per-chunk resume on partial re-runs."""
    from qdrant_client.http.models import (  # noqa: PLC0415
        FieldCondition,
        Filter,
        MatchValue,
    )

    found: set[str] = set()
    offset: Any = None
    flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=_SCROLL_BATCH,
            with_payload=["chunk_id"],
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            if p.payload and "chunk_id" in p.payload:
                found.add(str(p.payload["chunk_id"]))
        if offset is None:
            break
    return found


def _point_id_for(chunk_id: str) -> str:
    """Stable, deterministic UUID5 from chunk_id. Same chunk -> same point id."""
    return str(uuid.uuid5(_POINT_ID_NS, chunk_id))


def _upsert_batch(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense_vecs: Iterable[list[float]],
    sparse_vecs: Iterable[dict[int, float]],
) -> None:
    """Upsert one batch into Qdrant. The two vector iterables must align
    with ``chunks`` index-for-index."""
    from qdrant_client.http.models import PointStruct, SparseVector  # noqa: PLC0415

    points: list[PointStruct] = []
    for chunk, dense, sparse_dict in zip(chunks, dense_vecs, sparse_vecs, strict=True):
        # Sparse vectors arrive as {token_id: weight}; Qdrant wants two
        # parallel lists in deterministic order.
        if sparse_dict:
            indices_sorted = sorted(sparse_dict.keys())
            indices = [int(i) for i in indices_sorted]
            values = [float(sparse_dict[i]) for i in indices_sorted]
        else:
            indices, values = [], []
        points.append(
            PointStruct(
                id=_point_id_for(chunk.chunk_id),
                vector={
                    _DENSE_NAME: list(dense),
                    _SPARSE_NAME: SparseVector(indices=indices, values=values),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page_numbers": chunk.page_numbers,
                    "section_path": chunk.section_path,
                    "text": chunk.text,
                    "language": chunk.language.value,
                    "char_count": chunk.char_count,
                },
            )
        )
    client.upsert(collection_name=collection, points=points)


# ==================== embedding model ====================


def _encode_batch(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Encode a batch of chunk texts into (dense_vectors, sparse_dicts).

    Delegates to the shared ``BgeM3`` singleton so the model is loaded
    exactly once per process even when the retrieval stage uses it too.
    """
    return BgeM3.get().encode(texts, max_length=get_settings().embed_max_length, with_sparse=True)


# ==================== I/O ====================


def _read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            chunks.append(Chunk.model_validate_json(stripped))
    return chunks


# ==================== public entry ====================


def _batched(items: list[Chunk], batch_size: int) -> Iterator[list[Chunk]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _embed_one_doc(
    client: QdrantClient,
    collection: str,
    doc_id: str,
    chunks: list[Chunk],
    *,
    batch_size: int,
) -> tuple[int, int, int]:
    """Embed one document and upsert into Qdrant.

    Returns ``(indexed_count, skipped_already_present, failed_batches)``.
    Skipped chunks are those already indexed under this doc_id; their count
    is reported so the caller can include them in the run summary.
    """
    already = _existing_chunk_ids(client, collection, doc_id)
    to_embed = [c for c in chunks if c.chunk_id not in already]

    if not to_embed:
        logger.info(f"{doc_id}: all {len(chunks)} chunks already indexed, skipping")
        return 0, len(already), 0

    logger.info(
        f"{doc_id}: {len(to_embed)} chunks to embed "
        f"({len(already)} already indexed); batch_size={batch_size}"
    )

    indexed = 0
    failed_batches = 0
    for batch in _batched(to_embed, batch_size):
        try:
            dense, sparse = _encode_batch([c.text for c in batch])
            _upsert_batch(client, collection, batch, dense, sparse)
        except Exception as e:  # noqa: BLE001 — batch-level soft fail; captured in caller's result
            logger.exception(f"{doc_id}: batch starting at chunk {batch[0].chunk_id} failed: {e}")
            failed_batches += 1
            continue
        indexed += len(batch)
        if indexed % 100 < batch_size:
            logger.info(f"{doc_id}: {indexed}/{len(to_embed)} embedded")

    return indexed, len(already), failed_batches


def embed_all(*, only_doc_id: str | None = None, reset_collection: bool = False) -> EmbedResult:
    """Embed every chunked document and upsert to Qdrant.

    Args:
        only_doc_id: process only this doc_id.
        reset_collection: drop the collection before starting (destructive!).
    """
    settings = get_settings()
    docs = load_manifest(settings.raw_dir)
    if not docs:
        logger.warning("manifest is empty; nothing to embed")
        return EmbedResult()

    client = _connect_qdrant()

    if reset_collection:
        logger.warning(f"--reset-collection: dropping {settings.qdrant_collection!r}")
        if any(c.name == settings.qdrant_collection for c in client.get_collections().collections):
            client.delete_collection(settings.qdrant_collection)

    _ensure_collection(client, settings.qdrant_collection, settings.embed_dense_dim)

    result = EmbedResult()

    for meta in docs:
        if only_doc_id is not None and meta.doc_id != only_doc_id:
            continue

        chunks_path = doc_id_path(settings.processed_dir, "chunks", meta.doc_id)
        if not chunks_path.is_file():
            reason = f"chunks JSONL not found at {chunks_path}; run `python -m ingest.chunk` first"
            logger.error(f"{meta.doc_id}: {reason}")
            result.failed.append((meta.doc_id, reason))
            continue

        try:
            chunks = _read_chunks(chunks_path)
        except (OSError, ValueError) as e:
            logger.error(f"{meta.doc_id}: could not read chunks: {e}")
            result.failed.append((meta.doc_id, str(e)))
            continue

        if not chunks:
            logger.warning(f"{meta.doc_id}: chunks file is empty, skipping")
            result.succeeded.append(meta.doc_id)
            continue

        indexed, skipped, failed_batches = _embed_one_doc(
            client,
            settings.qdrant_collection,
            meta.doc_id,
            chunks,
            batch_size=settings.embed_batch_size,
        )
        result.chunks_indexed += indexed
        result.chunks_skipped += skipped

        if failed_batches > 0:
            reason = f"{failed_batches} batches failed; indexed {indexed}/{len(chunks) - skipped}"
            logger.error(f"{meta.doc_id}: partial — {reason}")
            result.failed.append((meta.doc_id, reason))
        else:
            logger.info(
                f"{meta.doc_id}: indexed {indexed} new chunks (total {len(chunks)} in index)"
            )
            result.succeeded.append(meta.doc_id)

    return result


# ==================== CLI ====================


@app.command()
def main(
    doc_id: str | None = typer.Option(None, "--doc-id", help="Embed only this doc_id."),
    reset_collection: bool = typer.Option(
        False, "--reset-collection", help="DROP the Qdrant collection before embedding."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Embed every chunked document into Qdrant."""
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    try:
        result = embed_all(only_doc_id=doc_id, reset_collection=reset_collection)
    except (QdrantUnavailableError, CollectionSchemaMismatchError) as e:
        logger.error(f"hard-fail: {e}")
        sys.exit(2)
    except Exception:  # noqa: BLE001 — top-level CLI catch-all; logs traceback, exits 2
        logger.exception("embed_all aborted on a hard-fail condition")
        sys.exit(2)

    logger.info(
        f"done. docs: {len(result.succeeded)} succeeded, {len(result.failed)} failed; "
        f"chunks indexed: {result.chunks_indexed}, already-present skipped: {result.chunks_skipped}"
    )

    if result.failed:
        logger.error("the following documents could not be fully embedded:")
        for failed_doc_id, reason in result.failed:
            logger.error(f"  - {failed_doc_id}: {reason}")
        sys.exit(1)


# Module-attribute reference so ruff doesn't flag `app` as unused
_ = EmbedError  # keeps the import meaningful in case future code uses it directly


if __name__ == "__main__":
    app()
