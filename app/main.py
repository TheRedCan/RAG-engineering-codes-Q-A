"""Streamlit chat UI on top of ``generation.answer.answer_question``.

Localhost-only by default (see SECURITY.md). Launch with:

    streamlit run app/main.py --server.address 127.0.0.1

Architecture choices:

- Chat-style transcript (``st.chat_message`` + ``st.chat_input``).
- One ``answer_question`` call per submission. The pipeline is opaque
  to the UI; no token-streaming because Ollama's JSON-mode response
  isn't valid until it's complete anyway.
- Citations rendered AFTER the answer in a collapsible expander,
  deduplicated by (doc_id, page) so multi-cite claims don't repeat.
- Errors raised by the pipeline are surfaced inline (no stack traces);
  Qdrant/Ollama outage becomes a red box with the actionable reason.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import streamlit as st

from common.errors import LlmUnavailableError, QdrantUnavailableError
from common.logging import configure_logging
from common.models import Language
from common.settings import get_settings
from generation.answer import answer_question

if TYPE_CHECKING:  # pragma: no cover
    from common.models import Answer


# Visible language label for the per-answer chip. Kept ASCII-only on
# purpose — no flag emoji — because some Windows terminals (and tools
# that scrape the page) render emoji unreliably.
_LANG_LABEL: dict[Language, str] = {
    Language.AR: "AR",
    Language.EN: "EN",
    Language.MIXED: "mixed",
    Language.UNKNOWN: "unknown",
}


@st.cache_resource
def _bootstrap() -> None:
    """One-time process setup. ``cache_resource`` means this runs once
    per Streamlit server, not per script-rerun."""
    configure_logging(level="INFO", log_dir=get_settings().log_dir)


def _render_answer(answer: Answer, elapsed: float | None = None) -> None:
    """Render an Answer inside the current chat-message container."""
    lang = _LANG_LABEL.get(answer.answer_language, answer.answer_language.value)
    meta_bits = [f"lang: {lang}", f"hops: {answer.hop_count}"]
    if elapsed is not None:
        meta_bits.append(f"{elapsed:.1f}s")
    st.caption(" · ".join(meta_bits))

    if not answer.claims:
        st.warning(
            "The indexed sources do not contain enough information to answer this "
            "question. (No claims could be grounded.)"
        )
        return

    # Numbered claims. Arabic claims get a right-to-left container so
    # punctuation and parentheses render correctly.
    is_rtl = answer.answer_language == Language.AR
    for i, claim in enumerate(answer.claims, start=1):
        if is_rtl:
            st.markdown(
                f"<div dir='rtl' style='text-align: right'>{i}. {claim.text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"{i}. {claim.text}")

    rows = _unique_citation_rows(answer)
    with st.expander(f"Sources ({len(rows)} unique chunk{'s' if len(rows) != 1 else ''})"):
        st.markdown("\n".join(rows))


def _unique_citation_rows(answer: Answer) -> list[str]:
    """Build the citation expander's markdown rows.

    Deduplicates by ``(doc_id, page-tuple, section_path)`` so a claim
    citing the same chunk twice — or two claims citing the same chunk —
    don't double-render. Order is preserved (first appearance wins).
    """
    seen: set[tuple[str, tuple[int, ...], str | None]] = set()
    rows: list[str] = []
    for claim in answer.claims:
        for c in claim.citations:
            key = (c.doc_id, tuple(c.page_numbers), c.section_path)
            if key in seen:
                continue
            seen.add(key)
            pages = ", ".join(map(str, c.page_numbers))
            section = f" — {c.section_path}" if c.section_path else ""
            rows.append(f"- **{c.doc_id}** p.{pages}{section}")
    return rows


def _render_sidebar() -> None:
    """Static info panel — corpus, settings, hard-fail hints."""
    s = get_settings()
    st.sidebar.title("About")
    st.sidebar.markdown(
        "Local RAG over **FEMA / NIST** building-code documents (and any "
        "bring-your-own PDFs in `data/raw/`). Bilingual: ask in English or "
        "Arabic; answers come back in the same language."
    )
    st.sidebar.subheader("Pipeline")
    st.sidebar.markdown(
        f"- retrieve top **{s.retrieve_top_k}** (hybrid: dense + sparse + RRF)\n"
        f"- rerank top **{s.rerank_top_k}** (cross-encoder; input capped at "
        f"**{s.rerank_input_cap}**)\n"
        f"- multi-hop up to **{s.multihop_max_hops}** hops\n"
        f"- generate with **{s.ollama_model}** via Ollama"
    )
    st.sidebar.subheader("Reset")
    if st.sidebar.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()


def _handle_submission(question: str) -> None:
    """Run one Q→A turn. Adds to history; renders the new turn in-place."""
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        t0 = time.monotonic()
        with st.status("Running pipeline...", expanded=False) as status:
            try:
                answer = answer_question(question)
            except QdrantUnavailableError as e:
                status.update(label="Qdrant unreachable", state="error")
                st.error(
                    f"Vector store (Qdrant) is not reachable: {e}. "
                    "Start it with `docker compose -f serving/docker-compose.yml up -d`."
                )
                return
            except LlmUnavailableError as e:
                status.update(label="Ollama unreachable", state="error")
                st.error(
                    f"Local LLM (Ollama) is not reachable or missing the model: {e}. "
                    "Start the Ollama tray app and `ollama pull "
                    f"{get_settings().ollama_model}`."
                )
                return
            elapsed = time.monotonic() - t0
            status.update(
                label=f"Done in {elapsed:.1f}s ({len(answer.claims)} claim"
                f"{'s' if len(answer.claims) != 1 else ''})",
                state="complete",
            )
        _render_answer(answer, elapsed=elapsed)

    st.session_state.history.append((question, answer, elapsed))


def main() -> None:
    st.set_page_config(
        page_title="Engineering Codes RAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _bootstrap()
    _render_sidebar()

    st.title("Engineering Codes Q&A")
    st.caption(
        "Ask in English or Arabic. Answers are grounded in cited sources from the "
        "indexed FEMA / NIST corpus."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    # Replay prior turns from this session.
    for question, answer, elapsed in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            _render_answer(answer, elapsed=elapsed)

    prompt = st.chat_input("Ask a question about the indexed codes…")
    if prompt:
        _handle_submission(prompt)


if __name__ == "__main__":
    main()
