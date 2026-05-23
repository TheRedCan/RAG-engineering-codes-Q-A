"""Shared pytest fixtures.

Anything used by more than one test module lives here, so test modules
stay focused on their unit-under-test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fpdf import FPDF


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tmp ``data/raw/`` whose location is wired into common.settings.

    Each test gets a fresh directory. The settings singleton is reset so the
    tmp paths are honored, and env vars are stubbed in so any code that reads
    settings during the test sees this directory.
    """
    d = tmp_path / "raw"
    d.mkdir()
    # Local import is intentional so each fixture invocation gets a
    # freshly-resettable module reference.
    from common import settings as settings_mod  # noqa: PLC0415

    monkeypatch.setattr(settings_mod, "_cached", None)
    monkeypatch.setenv("RAW_DIR", str(d))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    return d


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """The processed/ directory paired with the ``raw_dir`` fixture."""
    return tmp_path / "processed"


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[[str, list[str]], Path]:
    """Generate a small PDF in ``tmp_path`` from a list of page bodies.

    Usage:
        path = make_pdf("doc.pdf", ["Page 1 text...", "Page 2 text..."])

    Uses fpdf2; English-only because adding Arabic to fpdf2 needs a font file.
    The parser's Arabic-handling logic is tested separately at the
    ``_detect_language`` level with raw strings.
    """

    def _make(name: str, pages: list[str]) -> Path:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        for body in pages:
            pdf.add_page()
            pdf.set_font("helvetica", size=12)
            pdf.multi_cell(0, 8, text=body)
        path = tmp_path / name
        pdf.output(str(path))
        return path

    return _make
