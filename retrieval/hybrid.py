"""Hybrid retrieval over the Qdrant index.

Given a user query, we run two independent searches against the same
collection — one against the BGE-M3 dense vector (semantic similarity),
one against the BGE-M3 sparse vector (learned-lexical, like a smart BM25).
We fuse the two ranked lists with Reciprocal Rank Fusion (RRF) and return
the merged top-K as ``RetrievedChunk`` records.

Why hybrid (vs dense-only):

- Dense excels at semantic / paraphrase matching ("how do I compute the
  base shear" -> chunks about base-shear procedures).
- Sparse excels at exact-token / identifier matching ("§22.5.1", a
  specific equation number, a code reference). Dense embeddings smear
  these together; the sparse signal preserves them.

RRF rather than weighted-score fusion because the dense (cosine) and
sparse (dot-product) scores aren't on comparable scales. RRF only uses
ranks, which is robust across the two signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.embedder import BgeM3
from common.errors import QdrantUnavailableError
from common.logging import logger
from common.models import Chunk, Language, RetrievedChunk
from common.settings import get_settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import ScoredPoint

# Vector names match what ingest.embed wrote — change here ONLY together
# with that module and a re-embed of the corpus.
_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"

# Max tokens BGE-M3 will encode for the query. Queries are short by
# nature; 512 is generous and keeps query-time latency tight.
_QUERY_MAX_LENGTH = 512

# RRF constant — 60 is the original paper's recommendation and the
# de-facto standard.
_RRF_K = 60


# ==================== pure helpers ====================


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = _RRF_K
) -> list[tuple[str, float]]:
    """Combine multiple ranked id lists via RRF.

    Args:
        rankings: each inner list is one ranked sequence of chunk_ids
            (best first). Same id appearing in multiple lists accumulates
            score from each list it appears in.
        k: smoothing constant. Larger k flattens the score curve.

    Returns:
        ``(chunk_id, score)`` pairs sorted by score descending. Higher score
        = better.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Reconstruct a ``Chunk`` from a Qdrant point payload.

    The payload was written by ``ingest.embed._upsert_batch``; this is its
    inverse. Field shapes are validated by Pydantic so a malformed payload
    will surface immediately, not silently produce wrong-looking output.

    Note: the parameter is typed ``dict[str, Any]`` because Qdrant payloads
    are JSON — the runtime types come from the JSON deserializer, and we
    explicitly coerce each field to the expected Python type at the
    Pydantic boundary below.
    """
    pages_raw = payload["page_numbers"]
    if not isinstance(pages_raw, list):
        raise TypeError(f"page_numbers must be a list, got {type(pages_raw).__name__}")
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        doc_id=str(payload["doc_id"]),
        page_numbers=[int(p) for p in pages_raw],
        section_path=(
            None if payload.get("section_path") is None else str(payload["section_path"])
        ),
        text=str(payload["text"]),
        language=Language(str(payload["language"])),
        char_count=int(payload["char_count"]),
    )


# ==================== Qdrant search ====================


def _connect_qdrant() -> QdrantClient:
    """Build a client and verify the server responds. Same shape as
    ingest.embed._connect_qdrant; duplicated rather than imported to keep
    a one-way module dependency (retrieval must not depend on ingest)."""
    settings = get_settings()
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


def _search_dense(
    client: QdrantClient, collection: str, dense_vec: list[float], limit: int
) -> list[ScoredPoint]:
    resp = client.query_points(
        collection_name=collection,
        query=dense_vec,
        using=_DENSE_NAME,
        limit=limit,
        with_payload=True,
    )
    return list(resp.points)


def _search_sparse(
    client: QdrantClient, collection: str, sparse_dict: dict[int, float], limit: int
) -> list[ScoredPoint]:
    from qdrant_client.http.models import SparseVector  # noqa: PLC0415

    if not sparse_dict:
        return []
    indices_sorted = sorted(sparse_dict.keys())
    sv = SparseVector(
        indices=[int(i) for i in indices_sorted],
        values=[float(sparse_dict[i]) for i in indices_sorted],
    )
    resp = client.query_points(
        collection_name=collection,
        query=sv,
        using=_SPARSE_NAME,
        limit=limit,
        with_payload=True,
    )
    return list(resp.points)


# ==================== public entry ====================


def hybrid_search(query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
    """Run hybrid dense+sparse retrieval against the configured Qdrant
    collection and return the top-K fused results.

    Args:
        query: the user's question. Encoded once.
        top_k: how many results to return after fusion. Defaults to
            ``settings.retrieve_top_k``.

    Returns:
        ``RetrievedChunk`` records with ``dense_score`` and ``sparse_score``
        populated where each chunk appeared in that signal's ranking,
        ``rerank_score`` left None (the rerank stage fills it).
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.retrieve_top_k

    # Per-signal limits: query each signal for slightly more than top_k so
    # RRF has good overlap to work with even when the two signals disagree.
    per_signal_limit = max(top_k, settings.retrieve_top_k)

    # Fail fast on Qdrant BEFORE loading the embedder. The model load takes
    # 5-15 s and ~2 GB of RAM; we don't want to pay that just to discover
    # the index isn't reachable.
    client = _connect_qdrant()
    collection = settings.qdrant_collection

    logger.info(f"hybrid_search: encoding query (len={len(query)} chars)")
    dense_vecs, sparse_dicts = BgeM3.get().encode(
        [query], max_length=_QUERY_MAX_LENGTH, with_sparse=True
    )
    dense_vec, sparse_dict = dense_vecs[0], sparse_dicts[0]

    dense_hits = _search_dense(client, collection, dense_vec, per_signal_limit)
    sparse_hits = _search_sparse(client, collection, sparse_dict, per_signal_limit)

    logger.info(
        f"hybrid_search: dense returned {len(dense_hits)}, sparse returned {len(sparse_hits)}"
    )

    # Index by chunk_id (from payload, not by Qdrant point id, since the
    # payload is what downstream code reads).
    by_chunk_id: dict[str, dict[str, object]] = {}
    for p in dense_hits:
        cid = str(p.payload["chunk_id"]) if p.payload else None
        if cid is None:
            continue
        slot = by_chunk_id.setdefault(
            cid, {"payload": p.payload, "dense_score": None, "sparse_score": None}
        )
        slot["dense_score"] = float(p.score)
    for p in sparse_hits:
        cid = str(p.payload["chunk_id"]) if p.payload else None
        if cid is None:
            continue
        slot = by_chunk_id.setdefault(
            cid, {"payload": p.payload, "dense_score": None, "sparse_score": None}
        )
        slot["sparse_score"] = float(p.score)

    dense_ids = [str(p.payload["chunk_id"]) for p in dense_hits if p.payload]
    sparse_ids = [str(p.payload["chunk_id"]) for p in sparse_hits if p.payload]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids])

    results: list[RetrievedChunk] = []
    for rank, (cid, _rrf_score) in enumerate(fused[:top_k], start=1):
        info = by_chunk_id[cid]
        payload = info["payload"]
        # Narrow type for the type checker; payload is dict[str, object] by
        # construction (we never put anything else into the slot).
        assert isinstance(payload, dict)
        dense_score = info["dense_score"]
        sparse_score = info["sparse_score"]
        assert dense_score is None or isinstance(dense_score, float)
        assert sparse_score is None or isinstance(sparse_score, float)
        results.append(
            RetrievedChunk(
                chunk=chunk_from_payload(payload),
                dense_score=dense_score,
                sparse_score=sparse_score,
                rerank_score=None,
                rank=rank,
            )
        )
    return results
