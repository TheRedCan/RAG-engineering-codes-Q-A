"""CLI: ``python -m retrieval.query "your question here"``.

Runs hybrid retrieval, optionally reranks, and prints the results in a
human-readable format. No LLM yet — that's the generation stage. This is
the lowest-level interactive surface for sanity-checking the index.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from common.errors import QdrantUnavailableError
from common.logging import configure_logging, logger
from common.models import RetrievedChunk
from common.settings import get_settings
from retrieval.hybrid import hybrid_search
from retrieval.multihop import multihop_search
from retrieval.rerank import rerank as rerank_chunks

app = typer.Typer(add_completion=False, help="Query the engineering-codes index.")

_TEXT_PREVIEW_CHARS = 220
# Show this many leading page numbers before eliding with an ellipsis.
_PAGES_TO_SHOW = 4


def _print_results(query: str, results: list[RetrievedChunk], *, console: Console) -> None:
    """Render results as a compact table. Truncates chunk text to keep
    output skim-able; the full text is in the JSONL on disk if needed."""
    console.print(f"\n[bold]Query:[/bold] {query}\n")
    table = Table(show_lines=True, expand=True)
    table.add_column("#", style="cyan", width=3, no_wrap=True)
    table.add_column("doc / pages", style="green", overflow="fold", max_width=28)
    table.add_column("hop", style="blue", width=3, no_wrap=True)
    table.add_column("dense", style="yellow", width=7, no_wrap=True)
    table.add_column("sparse", style="yellow", width=7, no_wrap=True)
    table.add_column("rerank", style="magenta", width=7, no_wrap=True)
    table.add_column("text preview", overflow="fold")

    for r in results:
        # Compress \r\n and long whitespace runs so previews are dense.
        preview = " ".join(r.chunk.text.split())[:_TEXT_PREVIEW_CHARS]
        pages = ", ".join(str(p) for p in r.chunk.page_numbers[:_PAGES_TO_SHOW])
        if len(r.chunk.page_numbers) > _PAGES_TO_SHOW:
            pages += "…"
        table.add_row(
            str(r.rank),
            f"{r.chunk.doc_id}\np.{pages}",
            str(r.source_hop),
            f"{r.dense_score:.3f}" if r.dense_score is not None else "—",
            f"{r.sparse_score:.3f}" if r.sparse_score is not None else "—",
            f"{r.rerank_score:.3f}" if r.rerank_score is not None else "—",
            preview + ("…" if len(r.chunk.text) > _TEXT_PREVIEW_CHARS else ""),
        )
    console.print(table)


@app.command()
def main(
    query: str = typer.Argument(..., help="The user's question."),
    retrieve_k: int | None = typer.Option(
        None, "--retrieve-k", help="Candidates fetched by hybrid (default from settings)."
    ),
    rerank_k: int | None = typer.Option(
        None, "--rerank-k", help="Final number after reranking (default from settings)."
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip the cross-encoder reranking stage."
    ),
    no_multihop: bool = typer.Option(
        False, "--no-multihop", help="Skip cross-reference following (use plain hybrid)."
    ),
    max_hops: int | None = typer.Option(
        None, "--max-hops", help="Override settings.multihop_max_hops."
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Retrieve the top chunks for a query and print them."""
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    console = Console()

    try:
        if no_multihop:
            candidates = hybrid_search(query, top_k=retrieve_k)
        else:
            candidates = multihop_search(query, top_k=retrieve_k, max_hops=max_hops)
    except QdrantUnavailableError as e:
        logger.error(f"hard-fail: {e}")
        sys.exit(2)

    if not candidates:
        console.print("[yellow]No results.[/yellow]")
        sys.exit(0)

    if no_rerank:
        results = candidates[: (rerank_k or get_settings().rerank_top_k)]
    else:
        results = rerank_chunks(query, candidates, top_k=rerank_k)

    _print_results(query, results, console=console)


if __name__ == "__main__":
    app()
