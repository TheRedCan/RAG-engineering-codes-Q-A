"""Unit tests for ingest.fetch.

Strategy: drive the fetcher against an httpx mock (respx) and a tmp raw_dir,
asserting:

1. Successful download writes a file with the right bytes.
2. SHA-256 is computed and pinned to the lock file.
3. Subsequent run skips download but re-verifies the on-disk hash.
4. Manifest pre-pinned sha256 mismatch -> ChecksumMismatchError, file deleted.
5. Lock-file mismatch on re-download -> ChecksumMismatchError, file deleted.
6. HTTP 404 -> FetchError with the right url + status.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from common.errors import ChecksumMismatchError
from common.manifest import manifest_path
from ingest import fetch as fetch_mod

_PDF_BYTES = b"%PDF-1.4\n%fake pdf content for tests\n%%EOF\n"
_PDF_SHA = hashlib.sha256(_PDF_BYTES).hexdigest()
_URL = "https://example.test/doc.pdf"


def _write_manifest(raw_dir: Path, *, pinned_sha: str | None = None) -> None:
    entry = {
        "doc_id": "test-doc-1",
        "family": "sbc",
        "code_number": "999",
        "edition_year": 2024,
        "variant": "CR",
        "title_en": "Test Doc",
        "title_ar": None,
        "source_url": _URL,
        "expected_language": "en",
        "sha256": pinned_sha,
        "license_note": "test fixture",
    }
    manifest_path(raw_dir).write_text(json.dumps([entry]), encoding="utf-8")


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    # Override settings to point at this tmp dir. Local import is intentional
    # so each fixture invocation gets a freshly-resettable module reference.
    from common import settings as settings_mod  # noqa: PLC0415

    monkeypatch.setattr(settings_mod, "_cached", None)
    monkeypatch.setenv("RAW_DIR", str(d))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    return d


@respx.mock
def test_fetch_writes_file_and_pins_sha(raw_dir: Path) -> None:
    _write_manifest(raw_dir)
    respx.get(_URL).mock(return_value=httpx.Response(200, content=_PDF_BYTES))

    result = fetch_mod.fetch_all()

    assert result.all_succeeded
    assert len(result.succeeded) == 1
    assert result.succeeded[0].read_bytes() == _PDF_BYTES

    lock = json.loads((raw_dir / "manifest.lock.json").read_text())
    assert lock["test-doc-1"] == _PDF_SHA


@respx.mock
def test_second_run_skips_download_but_reverifies(raw_dir: Path) -> None:
    _write_manifest(raw_dir)
    route = respx.get(_URL).mock(return_value=httpx.Response(200, content=_PDF_BYTES))

    fetch_mod.fetch_all()
    fetch_mod.fetch_all()  # second call

    # Network should only be hit once.
    assert route.call_count == 1


@respx.mock
def test_manifest_pinned_mismatch_hard_fails(raw_dir: Path) -> None:
    """Checksum disagreement is treated as a possible tampering signal —
    it propagates out of fetch_all rather than being soft-collected."""
    _write_manifest(raw_dir, pinned_sha="0" * 64)
    respx.get(_URL).mock(return_value=httpx.Response(200, content=_PDF_BYTES))

    with pytest.raises(ChecksumMismatchError) as ei:
        fetch_mod.fetch_all()

    assert ei.value.expected == "0" * 64
    assert ei.value.actual == _PDF_SHA
    # Bad file is removed.
    assert not (raw_dir / "test-doc-1.pdf").exists()


@respx.mock
def test_lock_mismatch_on_redownload_hard_fails(raw_dir: Path) -> None:
    _write_manifest(raw_dir)

    # First successful fetch pins the real hash.
    respx.get(_URL).mock(return_value=httpx.Response(200, content=_PDF_BYTES))
    fetch_mod.fetch_all()

    # Now the server starts returning different bytes.
    different = b"%PDF-1.4\nsomething else\n%%EOF\n"
    respx.get(_URL).mock(return_value=httpx.Response(200, content=different))

    with pytest.raises(ChecksumMismatchError):
        fetch_mod.fetch_all(force=True)

    # The corrupted/new download is removed.
    assert not (raw_dir / "test-doc-1.pdf").exists()


@respx.mock
def test_http_404_is_soft_collected(raw_dir: Path) -> None:
    """Transient HTTP failure on one doc must not abort the whole run.
    The failure is reported via FetchResult.failed instead."""
    _write_manifest(raw_dir)
    respx.get(_URL).mock(return_value=httpx.Response(404, content=b"not found"))

    result = fetch_mod.fetch_all()

    assert not result.all_succeeded
    assert result.succeeded == []
    assert len(result.failed) == 1
    doc_id, reason = result.failed[0]
    assert doc_id == "test-doc-1"
    assert "404" in reason
