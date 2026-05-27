"""Eval dataset schema + loader.

A test case captures everything we know up-front about what a *good*
answer looks like. We don't store ground-truth answer text (that would
make every prompt tweak a churn against fragile string matches);
instead we encode the *invariants* that any good answer must satisfy.

Test cases live in ``eval/dataset.jsonl`` (one JSON object per line),
loaded and validated with strict Pydantic. Adding a new case is a
matter of appending one line; the runner picks it up automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from common.errors import ConfigError
from common.models import Language


class TestCase(BaseModel):
    """One labeled query in the eval set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="Stable slug; doubles as the row key in result reports.",
        pattern=r"^[a-z0-9][a-z0-9_-]{1,79}$",
    )
    question: str = Field(min_length=1)
    expected_language: Language = Field(
        description="Language the Answer.answer_language MUST equal."
    )
    expected_behavior: Literal["answer", "refuse"] = Field(
        description=(
            "answer: claims must be non-empty. "
            "refuse: claims must be empty (out-of-scope or unanswerable)."
        )
    )
    must_cite_doc_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Every doc_id listed here must appear in at least one "
            "Citation in the answer. Empty list = no citation requirement."
        ),
    )
    must_mention: list[str] = Field(
        default_factory=list,
        description=(
            "Each substring must appear (case-insensitive) in at least "
            "one claim text. Use for equations, key terms, numeric "
            "values the answer should not omit. Empty list = skip check."
        ),
    )
    must_not_mention: list[str] = Field(
        default_factory=list,
        description=(
            "No substring here may appear in any claim text. Use to "
            "guard against known wrong-direction hallucinations "
            "(e.g. 'Egyptian Code' must not appear when answering from "
            "the US corpus). Empty list = skip check."
        ),
    )
    max_latency_s: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Soft limit. Reported as warning if exceeded — never a "
            "hard fail, since LLM latency is hardware-dependent."
        ),
    )
    notes: str = Field(
        default="",
        description="Free-form explanation for humans reading the dataset.",
    )


def load_dataset(path: Path) -> list[TestCase]:
    """Load and validate a JSONL test-set file.

    Raises:
        ConfigError: file missing, line not valid JSON, schema mismatch,
            or duplicate test-case IDs.
    """
    if not path.is_file():
        raise ConfigError(f"eval dataset not found: {path}")

    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue  # blank line / comment
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}:{line_no}: not valid JSON: {e}") from e
        try:
            case = TestCase.model_validate(data)
        except Exception as e:
            raise ConfigError(f"{path}:{line_no}: schema validation failed: {e}") from e
        if case.id in seen_ids:
            raise ConfigError(f"{path}:{line_no}: duplicate test case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise ConfigError(f"eval dataset is empty: {path}")
    return cases
