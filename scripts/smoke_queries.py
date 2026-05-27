"""Smoke-test the full pipeline against a diverse set of engineer-style queries.

Runs in one Python process so the embedder and reranker are loaded once.
Each query goes through the full retrieval+rerank+LLM stack and the
result is printed compactly (no rich rendering, plain text for log files).

Usage:
    python -m scripts.smoke_queries
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make sibling packages importable when run from repo root via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows default stdout/stderr is cp1252 and chokes on Arabic. Force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from common.logging import configure_logging
from common.settings import get_settings
from generation.answer import answer_question

QUERIES: list[tuple[str, str]] = [
    (
        "equation extraction",
        "How is the seismic design coefficient Cs calculated?",
    ),
    (
        "definition / conceptual",
        "What is soil-structure interaction and when must it be considered?",
    ),
    (
        "specific code section",
        "What does ASCE 7-22 Section 12.8.1 require for the equivalent lateral force procedure?",
    ),
    (
        "different corpus (NIST resilience)",
        "What design considerations affect water and wastewater system resilience?",
    ),
    (
        "bilingual (Arabic question, English corpus)",
        "ما هي تركيبات الأحمال للتصميم الزلزالي؟",
    ),
    (
        "out-of-scope refusal",
        (
            "What is the procedure for designing a steel moment frame in the "
            "Egyptian Code of Practice?"
        ),
    ),
]


def _print_answer(label: str, question: str, answer, dt: float) -> None:
    print("=" * 80)
    print(f"[{label}]")
    print(f"Q: {question}")
    print(
        f"  elapsed: {dt:.1f}s   hops: {answer.hop_count}   "
        f"lang: {answer.answer_language.value}   "
        f"unique_chunks_used: {len(answer.used_chunks)}   "
        f"claim_count: {len(answer.claims)}"
    )
    if not answer.claims:
        print("  (no claims — system refused; corpus did not cover the question)")
    else:
        for i, c in enumerate(answer.claims, start=1):
            cites = "; ".join(
                f"{ct.doc_id} p.{', '.join(map(str, ct.page_numbers))}" for ct in c.citations
            )
            # Compact, ascii-safe-ish text dump
            text = " ".join(c.text.split())
            print(f"  ({i}) {text}")
            print(f"      sources: {cites}")
    print()


def main() -> None:
    # INFO so the smoke log shows which stage produced a refusal
    # (parse failure, verifier rejection, empty retrieval).
    configure_logging(level="INFO", log_dir=get_settings().log_dir)
    print(f"running {len(QUERIES)} queries through the full pipeline")
    print()
    for label, question in QUERIES:
        t0 = time.monotonic()
        try:
            answer = answer_question(question)
        except Exception as e:  # noqa: BLE001 — we want to keep going across queries
            print(f"=== [{label}] FAILED: {type(e).__name__}: {e}")
            continue
        _print_answer(label, question, answer, time.monotonic() - t0)


if __name__ == "__main__":
    main()
