"""End-to-end CLI: retrieve → prompt → LLM → cited Answer.

This is the user-facing entry point of the whole system. It composes the
existing pipeline stages:

1. ``retrieval.multihop.multihop_search`` (or hybrid_search via --no-multihop)
2. ``retrieval.rerank.rerank``
3. ``generation.prompt.build_user_prompt`` + ``llm_response_schema``
4. ``generation.llm.chat`` against Ollama
5. ``generation.prompt.parse_llm_response`` → canonical ``Answer``

Failure policy:
- Qdrant down -> hard fail (QdrantUnavailableError, exit 2)
- Ollama down / model missing -> hard fail (LlmUnavailableError, exit 2)
- LLM produced bad output -> hard fail (LlmOutputError, exit 1) because
  this indicates either a model regression or a schema mismatch and we
  want it loud
- No relevant chunks (claims=[] from LLM) -> NOT a hard fail; the system
  returns an Answer with empty claims and we print "no answer" cleanly
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from common.errors import LlmOutputError, LlmUnavailableError, QdrantUnavailableError
from common.logging import configure_logging, logger
from common.models import Answer, Language
from common.settings import get_settings
from generation.llm import chat, health_check
from generation.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    llm_response_schema,
    parse_llm_response,
)
from retrieval.hybrid import hybrid_search
from retrieval.multihop import multihop_search
from retrieval.rerank import rerank as rerank_chunks

app = typer.Typer(add_completion=False, help="End-to-end Q&A over the indexed engineering codes.")


def answer_question(
    question: str,
    *,
    retrieve_k: int | None = None,
    rerank_k: int | None = None,
    use_multihop: bool = True,
    max_hops: int | None = None,
) -> Answer:
    """End-to-end RAG: retrieve, rerank, generate, parse, return Answer.

    Validates Ollama health BEFORE running retrieval so we don't waste
    the (expensive) embedder load + search if the LLM isn't reachable.
    """
    # Cheapest health check first.
    health_check()

    # Retrieval.
    if use_multihop:
        candidates = multihop_search(question, top_k=retrieve_k, max_hops=max_hops)
    else:
        candidates = hybrid_search(question, top_k=retrieve_k)
    hop_count = max((c.source_hop for c in candidates), default=0) + 1

    if not candidates:
        logger.warning("retrieval returned no chunks; refusing without LLM call")
        # English fallback is fine for an empty-retrieval refusal: there's
        # no source content to detect a language from anyway.
        return Answer(
            question=question,
            answer_language=Language.EN,
            claims=[],
            used_chunks=[],
            hop_count=1,
        )

    # Rerank.
    final_candidates = rerank_chunks(question, candidates, top_k=rerank_k)
    logger.info(f"prompting LLM with {len(final_candidates)} reranked chunks")

    # LLM call.
    user_prompt = build_user_prompt(question, final_candidates)
    raw = chat(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        json_schema=llm_response_schema(),
    )

    # Parse + validate.
    return parse_llm_response(raw, question, final_candidates, hop_count=hop_count)


def _print_answer(answer: Answer, *, console: Console) -> None:
    """Render the structured answer as readable Markdown with citations."""
    console.print(f"\n[bold]Question:[/bold] {answer.question}\n")

    if not answer.claims:
        console.print(
            Panel(
                "The indexed sources do not contain enough information to answer this question.",
                title="[yellow]no answer[/yellow]",
                border_style="yellow",
            )
        )
        return

    md_parts: list[str] = []
    for i, claim in enumerate(answer.claims, start=1):
        cites = ", ".join(
            f"{c.doc_id} p.{', '.join(map(str, c.page_numbers))}" for c in claim.citations
        )
        md_parts.append(f"{i}. {claim.text}  \n   *Sources: {cites}*")
    console.print(Markdown("\n".join(md_parts)))

    console.print(
        f"\n[dim]language={answer.answer_language.value} "
        f"hops={answer.hop_count} unique_chunks_used={len(answer.used_chunks)}[/dim]"
    )


@app.command()
def main(
    question: str = typer.Argument(..., help="The engineer's question."),
    retrieve_k: int | None = typer.Option(None, "--retrieve-k"),
    rerank_k: int | None = typer.Option(None, "--rerank-k"),
    no_multihop: bool = typer.Option(False, "--no-multihop"),
    max_hops: int | None = typer.Option(None, "--max-hops"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Ask a question against the indexed engineering codes."""
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    console = Console()

    try:
        answer = answer_question(
            question,
            retrieve_k=retrieve_k,
            rerank_k=rerank_k,
            use_multihop=not no_multihop,
            max_hops=max_hops,
        )
    except (LlmUnavailableError, QdrantUnavailableError) as e:
        logger.error(f"hard-fail: {e}")
        sys.exit(2)
    except LlmOutputError as e:
        logger.error(f"LLM output unusable: {e}")
        sys.exit(1)

    _print_answer(answer, console=console)


if __name__ == "__main__":
    app()
