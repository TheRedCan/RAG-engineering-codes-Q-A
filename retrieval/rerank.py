"""Cross-encoder reranking with BGE-reranker-v2-m3.

Bi-encoders (BGE-M3, what hybrid_search uses) encode the query and each
chunk **independently** into vectors and compare them with cosine — fast
at scale because chunk vectors are precomputed, but they never see the
two pieces of text together.

A cross-encoder takes ``(query, chunk_text)`` as one sequence through the
model, so it can attend across the boundary. The result is a much sharper
relevance score. Cost: we have to run the model once per (query, chunk)
pair at query time, so reranking is only practical on the top-K
candidates from hybrid retrieval, not the whole corpus.

We use the same vendor family for compatibility (BGE-M3 + BGE-reranker)
and because it handles Arabic and English natively, preserving the
bilingual contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.logging import logger
from common.models import RetrievedChunk
from common.settings import get_settings

if TYPE_CHECKING:  # pragma: no cover
    pass


class BgeReranker:
    """Singleton wrapper around FlagEmbedding.FlagReranker."""

    _instance: BgeReranker | None = None

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(f"loading reranker {settings.rerank_model!r} on {settings.rerank_device}")
        from FlagEmbedding import FlagReranker  # noqa: PLC0415

        use_fp16 = settings.rerank_device.startswith("cuda")
        self._model: Any = FlagReranker(
            settings.rerank_model,
            use_fp16=use_fp16,
            devices=[settings.rerank_device],
        )

    @classmethod
    def get(cls) -> BgeReranker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score per (query, document) pair.

        Higher = more relevant. Scores are unbounded; only relative
        ordering matters across pairs from the same call.
        """
        if not pairs:
            return []
        raw = self._model.compute_score([list(p) for p in pairs])
        # FlagReranker returns either a single float (one pair) or a list.
        if isinstance(raw, float | int):
            return [float(raw)]
        return [float(s) for s in raw]


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Rerank candidates against the query, return the new top-K.

    The original ``dense_score`` and ``sparse_score`` are preserved on each
    returned chunk so callers can inspect all three signals. ``rerank_score``
    is filled in. ``rank`` is rewritten to the post-rerank order.

    Args:
        query: the user's question (same one passed to hybrid_search).
        candidates: output of hybrid_search.
        top_k: how many to keep. Defaults to ``settings.rerank_top_k``.
    """
    if not candidates:
        return []

    settings = get_settings()
    if top_k is None:
        top_k = settings.rerank_top_k

    # Cap the candidate pool before cross-encoder scoring. The reranker
    # is the dominant CPU cost; capping keeps latency in budget. The cap
    # is applied AFTER RRF fusion, so sparse-only equation chunks that
    # earned a high RRF score survive.
    if len(candidates) > settings.rerank_input_cap:
        logger.info(
            f"rerank: capping {len(candidates)} candidates to {settings.rerank_input_cap} "
            f"before scoring (keeping top-N by current rank)"
        )
        candidates = candidates[: settings.rerank_input_cap]

    pairs = [(query, c.chunk.text) for c in candidates]
    scores = BgeReranker.get().score(pairs)
    logger.info(
        f"rerank: scored {len(pairs)} candidates (min={min(scores):.3f}, max={max(scores):.3f})"
    )

    # Sort candidates by reranker score, descending.
    scored = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)

    return [
        old.model_copy(update={"rerank_score": s, "rank": new_rank})
        for new_rank, (old, s) in enumerate(scored[:top_k], start=1)
    ]
