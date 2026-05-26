"""Multi-hop retrieval via deterministic cross-reference following.

Engineering codes are densely cross-referenced. A chunk about "load
combinations" routinely says "see §2.3.6" or "refer to Table 4-1". To
answer a question like "what load combination applies to seismic in
combination with dead load?", the right answer may need both the chunk
the dense search surfaced *and* the chunk it points at.

Approach: deterministic, not LLM-driven.

1. Hop 0: ``hybrid_search`` against the user query.
2. Scan retrieved chunk texts with a regex catalogue for code-style
   cross-references (§22.5.1, Section 22.5.1, Table 4-1, Figure 3.2,
   Eq. (2-3), and Arabic equivalents).
3. For each unique reference token, fire a follow-up hybrid search
   ("section 22.5.1" etc.). The sparse signal in BGE-M3 is excellent at
   exact-token matches, which is what cross-references really are.
4. Union new chunks into the candidate set with ``source_hop`` recording
   the depth. Cap at ``MULTIHOP_MAX_HOPS`` total hops.
5. The single source of truth for final ordering is the cross-encoder
   rerank against the *original* query — multi-hop only widens the
   candidate pool, it never overrides the reranker's verdict.

Why deterministic and not an LLM-driven agent loop:

- References are syntactic, not semantic; regex is the right tool.
- An LLM loop costs another inference per hop and is harder to bound.
- Predictable behavior is easier to test and easier to debug when an
  engineer asks why a particular chunk ended up in their answer.
"""

from __future__ import annotations

import re

from common.logging import logger
from common.models import RetrievedChunk
from common.settings import get_settings
from retrieval.hybrid import hybrid_search

# Per-reference hybrid-search budget. Small because we're aiming at a
# specific section, not a general topic.
_PER_REF_TOP_K = 3

# How many distinct references to chase per hop. Caps the worst-case
# fan-out (some chunks contain a dozen cross-refs and we don't want to
# fire all of them).
_MAX_REFS_PER_HOP = 8


# ==================== reference extraction ====================
#
# Each pattern below uses a named group ``num`` for the section/chapter
# identifier. We collect just the identifier so "Section 22.5.1" and
# "§22.5.1" dedupe to the same reference even though the surface
# text differs.

_SECTION_NUMBER = r"(?P<num>\d+(?:[.\-]\d+)+)"  # 22.5.1, 4-3, 2.1, etc.
_CHAPTER_NUMBER = r"(?P<num>\d+(?:[.\-]\d+)*)"  # also allows bare "5"

_REFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # English section
    (
        "section",
        re.compile(
            rf"(?:§\s*|(?:[Ss]ec(?:tion|\.)?)\s+){_SECTION_NUMBER}\b",
        ),
    ),
    # English chapter (must have explicit word; bare numbers are too noisy)
    (
        "chapter",
        re.compile(
            rf"(?:[Cc]h(?:apter|\.)?)\s+{_CHAPTER_NUMBER}\b",
        ),
    ),
    # === tables and figures ===
    ("table", re.compile(r"[Tt]able\s+(?P<num>\d+(?:[.\-]\d+)*)\b")),
    ("figure", re.compile(r"[Ff]ig(?:ure|\.)?\s+(?P<num>\d+(?:[.\-]\d+)*)\b")),
    # Equations
    (
        "eq",
        re.compile(
            r"[Ee]q(?:uation|\.)?\s*\(?(?P<num>\d+(?:[.\-]\d+)*)\)?",
        ),
    ),
    # Arabic patterns — present so bring-your-own ECP/SBC corpora work
    # without code changes. Patterns target البند (section) and الفصل
    # (chapter). The character class includes Arabic-Indic digits
    # U+0660..U+0669 alongside ASCII digits, so the ruff warnings about
    # those "ambiguous" characters are suppressed per-line: they are
    # intentional and required for actually matching Arabic text.
    (
        "ar_section",
        re.compile(
            r"البند\s+(?P<num>[\d٠-٩]+(?:[.\-][\d٠-٩]+)+)",  # noqa: RUF001
        ),
    ),
    (
        "ar_chapter",
        re.compile(
            r"الفصل\s+(?P<num>[\d٠-٩]+(?:[.\-][\d٠-٩]+)*)",  # noqa: RUF001
        ),
    ),
)


def extract_references(text: str) -> list[tuple[str, str]]:
    """Return a deduped list of (kind, identifier) cross-references.

    ``kind`` is one of ``section / chapter / table / figure / eq / ar_*``;
    ``identifier`` is the numeric part (e.g. ``"22.5.1"``).

    Order is stable: kinds in the catalogue order, identifiers in first-
    seen order within each kind. Dedup is on the (kind, identifier) tuple
    so "§22.5.1" and "Section 22.5.1" collapse to one entry.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for kind, pattern in _REFERENCE_PATTERNS:
        for m in pattern.finditer(text):
            ref = (kind, m.group("num"))
            if ref not in seen:
                seen.add(ref)
                out.append(ref)
    return out


def references_in_chunks(chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
    """Aggregate references across multiple chunks, preserving first-seen
    order and deduping globally."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for c in chunks:
        for ref in extract_references(c.chunk.text):
            if ref not in seen:
                seen.add(ref)
                out.append(ref)
    return out


def _ref_to_query(ref: tuple[str, str]) -> str:
    """Translate a (kind, identifier) reference into a follow-up query
    string. Kept simple — the sparse signal will dominate the match on
    the identifier token; the English/Arabic kind word helps dense."""
    kind, num = ref
    kind_word = {
        "section": "section",
        "chapter": "chapter",
        "table": "table",
        "figure": "figure",
        "eq": "equation",
        "ar_section": "البند",
        "ar_chapter": "الفصل",
    }.get(kind, kind)
    return f"{kind_word} {num}"


# ==================== orchestration ====================


def multihop_search(
    query: str,
    *,
    top_k: int | None = None,
    max_hops: int | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval, then follow cross-references for up to
    ``max_hops`` additional rounds.

    Returns the union of all candidates (direct hits + multi-hop pulls)
    *before* reranking. The caller is expected to feed the result through
    ``retrieval.rerank.rerank`` with the original query so the cross-
    encoder decides the final ordering against the user's actual intent.

    Args:
        query: the user's question.
        top_k: candidate budget for hop 0. Defaults to settings.retrieve_top_k.
        max_hops: total hop count cap (hop 0 included). Defaults to
            settings.multihop_max_hops. ``max_hops=1`` is single-hop
            (equivalent to plain hybrid_search).
    """
    settings = get_settings()
    if max_hops is None:
        max_hops = settings.multihop_max_hops
    if max_hops < 1:
        raise ValueError(f"max_hops must be >= 1, got {max_hops}")

    # Hop 0 — direct search against the user query.
    candidates = hybrid_search(query, top_k=top_k)
    for c in candidates:
        c.source_hop = 0  # explicit, even though Pydantic default already sets it
    seen_chunk_ids: set[str] = {c.chunk.chunk_id for c in candidates}
    logger.info(f"multihop: hop 0 surfaced {len(candidates)} candidates")

    # Refs we've already followed (across all hops) — avoids re-querying the
    # same reference from a later hop's chunks.
    followed: set[tuple[str, str]] = set()
    frontier_chunks = list(candidates)

    for hop in range(1, max_hops):
        refs = [r for r in references_in_chunks(frontier_chunks) if r not in followed]
        if not refs:
            logger.info(f"multihop: hop {hop} — no new references, stopping")
            break

        # Cap fan-out per hop. Take the first N (which preserves first-seen
        # priority across the chunks we just looked at).
        refs = refs[:_MAX_REFS_PER_HOP]
        logger.info(f"multihop: hop {hop} — chasing {len(refs)} references")

        new_this_hop: list[RetrievedChunk] = []
        for ref in refs:
            followed.add(ref)
            ref_query = _ref_to_query(ref)
            for r in hybrid_search(ref_query, top_k=_PER_REF_TOP_K):
                if r.chunk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(r.chunk.chunk_id)
                # The hybrid scores from a ref-query aren't comparable to the
                # user query's scores; null them out so the table/UI doesn't
                # falsely imply they are. The reranker (run later) sets the
                # only score that matters for final ordering.
                r.dense_score = None
                r.sparse_score = None
                r.source_hop = hop
                new_this_hop.append(r)

        if not new_this_hop:
            logger.info(f"multihop: hop {hop} — refs found nothing new, stopping")
            break

        candidates.extend(new_this_hop)
        frontier_chunks = new_this_hop  # only chase refs from newly-added chunks next hop
        logger.info(
            f"multihop: hop {hop} added {len(new_this_hop)} new candidates "
            f"(total now {len(candidates)})"
        )

    return candidates
