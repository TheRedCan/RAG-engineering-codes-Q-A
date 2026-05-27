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

import re
from typing import TYPE_CHECKING, Any

from common.logging import logger
from common.models import RetrievedChunk
from common.settings import get_settings

if TYPE_CHECKING:  # pragma: no cover
    pass


# ==================== direct-question boost ====================
#
# Cross-encoder reranking treats every chunk that mentions the queried
# concept similarly. For "how is Cs calculated?" the model can't tell
# that the user wants the *equation* for Cs (chapter 12 ELF) versus
# *commentary about* Cs (chapter 17 isolation systems). Both score
# highly because both contain "Cs".
#
# Fix: when the query has the shape of a direct-calculation /
# specific-requirement question, give a small additive bonus to chunks
# whose text actually contains concrete markers (equation symbols,
# explicit Eq./Table/§ references). The bonus is small enough that it
# only changes ordering for near-ties — it can't pull an irrelevant
# chunk to the top.

# Patterns that flag a question as wanting concrete content.
_DIRECT_QUESTION_PATTERNS = [
    re.compile(
        r"\bhow\s+(?:is|do\s+(?:i|you|we)|are)\b.*"
        r"\b(?:calculate|compute|determine|derive|find)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+is\s+the\s+(?:equation|formula|value|expression)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhat\s+does\s+(?:section|chapter|§|article)\b", re.IGNORECASE),
    re.compile(
        # Allow up to ~3 intervening words between "the" and the keyword,
        # e.g. "what are the load combinations".
        r"\bwhat\s+(?:are|is)\s+the\s+(?:\S+\s+){0,3}"
        r"(?:requirements?|combinations?|values?|limits?|thresholds?)\b",
        re.IGNORECASE,
    ),
    # Arabic equivalents: "ما هي" (what is) + "كيف" (how)
    re.compile(r"كيف\s+(?:يتم|نحسب|تحسب)"),
    re.compile(r"ما\s+(?:هي|هو)\s+(?:معادلة|قيمة|تركيب|متطلب)"),
]

# Patterns that flag a chunk as containing concrete content. These are
# deliberately permissive within scope — the bonus is small and additive,
# not a hard filter.
_EQUATION_MARKERS = [
    # `Eq. 12.8-2`, `Equation 5-3`
    re.compile(r"\bEq(?:uation)?\.?\s+\d", re.IGNORECASE),
    # `Table 12.8-2`, `Table C17.2-3`
    re.compile(r"\bTable\s+[A-Z]?\d", re.IGNORECASE),
    # `§ 12.8.1`, `Section 12.8.1`
    re.compile(r"§\s*\d|\bSection\s+\d+\.\d+", re.IGNORECASE),
    # Variable-style assignments: `X = 1.25` or `Cs = SDS/(R/Ie)`
    re.compile(r"[A-Za-z][A-Za-z0-9_]{0,8}\s*=\s*[A-Za-z0-9.()/\s+\-*]{2,}"),
]

# Bonus per marker hit, capped. Reranker scores typically span [-10, +5];
# a 1.0 bump can swap near-ties (~0.5 apart) without overwhelming clear
# winners or losers.
_BONUS_PER_MARKER = 0.5
_MAX_BONUS = 1.5


def _is_direct_question(query: str) -> bool:
    """Return True if the query asks for concrete equation/value/requirement content."""
    return any(p.search(query) for p in _DIRECT_QUESTION_PATTERNS)


def _equation_bonus(text: str) -> float:
    """Score-bump for chunks containing equation / table / section markers.

    Each distinct marker pattern that matches contributes
    ``_BONUS_PER_MARKER`` up to ``_MAX_BONUS``. We count *patterns hit*,
    not *match count*, so a chunk packed with `=` signs gets the same
    bonus as one with one equation and one Eq. reference — both signal
    "this chunk has concrete content."
    """
    hits = sum(1 for p in _EQUATION_MARKERS if p.search(text))
    return min(hits * _BONUS_PER_MARKER, _MAX_BONUS)


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

    # Direct-question boost: when the query asks for a calculation /
    # specific requirement, give chunks with concrete content markers
    # (Eq., Table, §, X = ...) a small additive bump so they outrank
    # near-tied narrative chunks at the top of the list. No-op for
    # conceptual queries like "what is soil-structure interaction".
    direct = _is_direct_question(query)
    if direct:
        bonuses = [_equation_bonus(c.chunk.text) for c in candidates]
        final_scores = [s + b for s, b in zip(scores, bonuses, strict=True)]
        n_boosted = sum(1 for b in bonuses if b > 0)
        logger.info(
            f"rerank: direct-question detected; boosted {n_boosted}/{len(candidates)} "
            f"equation-bearing chunks (max bonus={_MAX_BONUS})"
        )
    else:
        final_scores = scores

    # Sort candidates by final score, descending.
    scored = sorted(zip(candidates, final_scores, strict=True), key=lambda x: x[1], reverse=True)

    return [
        old.model_copy(update={"rerank_score": s, "rank": new_rank})
        for new_rank, (old, s) in enumerate(scored[:top_k], start=1)
    ]
