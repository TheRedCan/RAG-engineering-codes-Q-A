"""Unit tests for eval.dataset.

The loader must be strict: a malformed line in the dataset is a hard
error rather than a silent skip — eval results are meaningless if a
case quietly dropped out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.errors import ConfigError
from common.models import Language
from eval.dataset import TestCase as EvalCase  # avoid pytest "TestCase" collection
from eval.dataset import load_dataset


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


_CASE_JSON = (
    '{"id": "c1", "question": "Q?", "expected_language": "en", '
    '"expected_behavior": "answer", "must_cite_doc_ids": [], '
    '"must_mention": [], "must_not_mention": []}'
)


def test_loads_minimal_valid_dataset(tmp_path: Path) -> None:
    path = _write(tmp_path, [_CASE_JSON])
    cases = load_dataset(path)
    assert len(cases) == 1
    assert cases[0].id == "c1"
    assert cases[0].expected_language == Language.EN


def test_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            "// this is a comment",
            "",
            _CASE_JSON,
            "  ",
            "// another comment after",
        ],
    )
    cases = load_dataset(path)
    assert len(cases) == 1


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_dataset(tmp_path / "absent.jsonl")


def test_empty_dataset_raises(tmp_path: Path) -> None:
    """A dataset file with only blanks/comments is meaningless — refuse."""
    path = _write(tmp_path, ["// just a comment", ""])
    with pytest.raises(ConfigError, match="empty"):
        load_dataset(path)


def test_invalid_json_raises_with_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, [_CASE_JSON, "this is not json"])
    with pytest.raises(ConfigError, match=r":2: not valid JSON"):
        load_dataset(path)


def test_schema_violation_raises_with_line_number(tmp_path: Path) -> None:
    """Missing required field 'question'."""
    bad = '{"id": "c1", "expected_language": "en", "expected_behavior": "answer"}'
    path = _write(tmp_path, [_CASE_JSON, bad])
    with pytest.raises(ConfigError, match=r":2: schema validation failed"):
        load_dataset(path)


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    path = _write(tmp_path, [_CASE_JSON, _CASE_JSON])
    with pytest.raises(ConfigError, match="duplicate test case id"):
        load_dataset(path)


def test_invalid_id_slug_raises(tmp_path: Path) -> None:
    """ID must match the slug pattern — uppercase / spaces forbidden."""
    bad = _CASE_JSON.replace('"id": "c1"', '"id": "BadID with spaces"')
    path = _write(tmp_path, [bad])
    with pytest.raises(ConfigError, match="schema validation failed"):
        load_dataset(path)


def test_extra_fields_forbidden(tmp_path: Path) -> None:
    """Typo-prone — if someone writes `must_metion` instead of `must_mention`
    the silent default would mean their guard never fires."""
    bad = _CASE_JSON.replace(
        '"must_not_mention": []',
        '"must_not_mention": [], "must_metion": ["typo"]',
    )
    path = _write(tmp_path, [bad])
    with pytest.raises(ConfigError, match="schema validation failed"):
        load_dataset(path)


def test_real_dataset_loads_clean() -> None:
    """The committed dataset must always be loadable. Catches author typos
    when adding new cases."""
    real_path = Path(__file__).parent.parent.parent / "eval" / "dataset.jsonl"
    if not real_path.exists():
        pytest.skip("dataset file not present in this checkout")
    cases = load_dataset(real_path)
    assert len(cases) >= 1
    # Sanity: every required-citation doc_id should look like a real slug.
    for case in cases:
        for doc_id in case.must_cite_doc_ids:
            assert "-" in doc_id, f"case {case.id}: doc_id {doc_id!r} doesn't look slugged"


def test_eval_case_is_frozen() -> None:
    """EvalCase is immutable so the runner can safely share instances
    across goroutines / future async runs."""
    case = EvalCase(
        id="c1",
        question="Q?",
        expected_language=Language.EN,
        expected_behavior="answer",
    )
    with pytest.raises(Exception):  # noqa: B017 — Pydantic's ValidationError on frozen model
        # mypy correctly flags this as a read-only assignment — which is
        # exactly what we want to verify works *at runtime* too.
        case.question = "Q2?"  # type: ignore[misc]
