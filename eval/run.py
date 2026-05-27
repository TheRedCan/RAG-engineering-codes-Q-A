"""Run the eval dataset against the live pipeline; emit summary + JSON.

Usage:
    python -m eval.run                       # full dataset, default paths
    python -m eval.run --ids cs-coefficient  # one or more case ids, comma-sep
    python -m eval.run --no-write            # don't persist a results file

Wall-clock cost: ~75s per query on an RTX 4060 laptop. The full 16-case
set takes ~20 minutes. Not intended to run on every commit; rerun
before releases or after pipeline changes.

Each run writes ``eval/results/{UTC-timestamp}.json`` for trend tracking:
the file contains the full per-case metric breakdown plus the dataset
hash so later comparisons can detect dataset drift.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import typer

from common.logging import configure_logging, logger
from common.settings import get_settings
from eval.dataset import TestCase, load_dataset
from eval.score import ScoredCase, Verdict, score_answer
from generation.answer import answer_question

# Force UTF-8 stdout so Arabic case IDs / questions print on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
_RESULTS_DIR = Path(__file__).parent / "results"

app = typer.Typer(add_completion=False, help="Run the engineering-codes-rag eval set.")


def _filter_cases(cases: list[TestCase], ids: str | None) -> list[TestCase]:
    """Apply --ids filter. Comma-separated, exact match. Unknown ids raise."""
    if not ids:
        return cases
    wanted = {x.strip() for x in ids.split(",") if x.strip()}
    by_id = {c.id: c for c in cases}
    missing = wanted - by_id.keys()
    if missing:
        raise typer.BadParameter(f"unknown case ids: {sorted(missing)}")
    return [by_id[i] for i in wanted]


def _dataset_hash(cases: list[TestCase]) -> str:
    """SHA-256 of the canonical-ordered dataset for run-to-run comparisons."""
    serialized = json.dumps(
        [c.model_dump(mode="json") for c in cases],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _run_one(case: TestCase) -> ScoredCase | str:
    """Execute one case end-to-end. Returns ScoredCase on success, error string on hard fail."""
    t0 = time.monotonic()
    try:
        answer = answer_question(case.question)
    except Exception as e:  # noqa: BLE001 — at this level we treat any pipeline failure as a case error
        return f"{type(e).__name__}: {e}"
    return score_answer(case, answer, time.monotonic() - t0)


def _format_verdict(v: Verdict) -> str:
    return {
        Verdict.PASS: "PASS",
        Verdict.FAIL: "FAIL",
        Verdict.WARN: "WARN",
        Verdict.SKIP: "skip",
    }[v]


def _print_case(case: TestCase, result: ScoredCase | str) -> None:
    """Per-case stdout: id, question, per-metric verdicts, aggregate line."""
    print("=" * 80)
    print(f"[{case.id}]  {case.question}")
    if isinstance(result, str):
        print(f"  PIPELINE ERROR: {result}")
        return
    print(f"  elapsed: {result.elapsed_s:.1f}s")
    for m in result.metrics:
        line = f"  - {m.name:<18} {_format_verdict(m.verdict)}"
        if m.detail:
            line += f"  ({m.detail})"
        print(line)
    status = "PASS" if result.passed else "FAIL"
    extras = []
    if result.fail_count:
        extras.append(f"{result.fail_count} fail")
    if result.warn_count:
        extras.append(f"{result.warn_count} warn")
    suffix = f"  [{', '.join(extras)}]" if extras else ""
    print(f"  => {status}{suffix}")


def _print_summary(results: dict[str, ScoredCase | str]) -> None:
    """Aggregate footer: pass rate, error count, latency stats."""
    total = len(results)
    errored = sum(1 for r in results.values() if isinstance(r, str))
    scored = [r for r in results.values() if isinstance(r, ScoredCase)]
    passed = sum(1 for r in scored if r.passed)
    warned = sum(1 for r in scored if r.warn_count > 0)
    latencies = [r.elapsed_s for r in scored]

    print()
    print("=" * 80)
    print("SUMMARY")
    print("-" * 80)
    pct = 100 * passed / max(1, len(scored))
    print(f"  total cases:      {total}")
    print(f"  pipeline errors:  {errored}")
    print(f"  passed:           {passed} / {len(scored)}  ({pct:.0f}%)")
    print(f"  with warnings:    {warned}")
    if latencies:
        latencies_sorted = sorted(latencies)
        median = latencies_sorted[len(latencies_sorted) // 2]
        print(
            f"  latency:          min={min(latencies):.1f}s  "
            f"median={median:.1f}s  max={max(latencies):.1f}s"
        )


def _write_results(
    results: dict[str, ScoredCase | str],
    cases: list[TestCase],
    out_path: Path,
) -> None:
    """Persist a JSON dump of the run for trend tracking."""
    payload: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset_hash": _dataset_hash(cases),
        "n_cases": len(cases),
        "cases": [],
    }
    cases_out: list[dict[str, object]] = []
    for case_id, result in results.items():
        if isinstance(result, str):
            cases_out.append({"id": case_id, "error": result})
            continue
        cases_out.append(
            {
                "id": case_id,
                "elapsed_s": result.elapsed_s,
                "passed": result.passed,
                "metrics": [
                    {
                        **asdict(m),
                        "verdict": m.verdict.value,
                    }
                    for m in result.metrics
                ],
            }
        )
    payload["cases"] = cases_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresults written to {out_path}")


@app.command()
def main(
    ids: str | None = typer.Option(
        None, "--ids", help="Comma-separated list of case ids to run (default: all)."
    ),
    no_write: bool = typer.Option(
        False, "--no-write", help="Skip writing results file (still prints to stdout)."
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    dataset = _DATASET_PATH
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    cases = _filter_cases(load_dataset(dataset), ids)
    logger.info(f"running {len(cases)} eval case(s)")

    print(f"running {len(cases)} case(s) through the live pipeline")
    print()
    results: dict[str, ScoredCase | str] = {}
    for case in cases:
        result = _run_one(case)
        results[case.id] = result
        _print_case(case, result)

    _print_summary(results)

    if not no_write:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        _write_results(results, cases, _RESULTS_DIR / f"{ts}.json")

    # Exit code: 0 if every case passed (including pipeline-error cases),
    # 1 otherwise. Useful as a pre-release gate or for git bisect.
    any_fail = any(isinstance(r, str) or not r.passed for r in results.values())
    raise typer.Exit(code=1 if any_fail else 0)


if __name__ == "__main__":
    app()
