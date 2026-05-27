"""Unit tests for common.manifest.append_manifest_entry.

The Streamlit BYO ingest page depends on this helper being atomic and
idempotent-or-loud — it must never silently overwrite an existing
manifest entry, and a crash mid-write must not corrupt the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.errors import ConfigError
from common.manifest import append_manifest_entry, load_manifest, manifest_path
from common.models import CodeFamily, DocumentMeta, Language


def _entry(doc_id: str, *, family: CodeFamily = CodeFamily.OTHER) -> DocumentMeta:
    return DocumentMeta(
        doc_id=doc_id,
        family=family,
        code_number="X-1",
        edition_year=2020,
        variant=None,
        title_en=f"Title for {doc_id}",
        title_ar=None,
        source_url=f"byo://local-upload/{doc_id}",
        expected_language=Language.EN,
        sha256=None,
        license_note="test entry",
    )


def test_append_to_missing_manifest_creates_it(tmp_path: Path) -> None:
    """First-ever BYO upload: there's no manifest.json yet, so the
    helper must create it rather than erroring on missing file."""
    append_manifest_entry(tmp_path, _entry("doc-a"))
    out = load_manifest(tmp_path)
    assert [e.doc_id for e in out] == ["doc-a"]


def test_append_preserves_existing_entries(tmp_path: Path) -> None:
    append_manifest_entry(tmp_path, _entry("doc-a"))
    append_manifest_entry(tmp_path, _entry("doc-b"))
    out = load_manifest(tmp_path)
    assert [e.doc_id for e in out] == ["doc-a", "doc-b"]


def test_append_rejects_duplicate_doc_id(tmp_path: Path) -> None:
    """A clashing doc_id must NEVER silently overwrite — otherwise an
    engineer's UI flow could clobber a hand-curated manifest entry."""
    append_manifest_entry(tmp_path, _entry("doc-a"))
    with pytest.raises(ConfigError, match="already contains doc_id"):
        append_manifest_entry(tmp_path, _entry("doc-a"))


def test_append_is_atomic(tmp_path: Path) -> None:
    """Verify the .tmp + replace pattern — after a successful append the
    .tmp file must NOT remain on disk."""
    append_manifest_entry(tmp_path, _entry("doc-a"))
    leftovers = list(tmp_path.glob("manifest.json.*"))
    assert leftovers == []


def test_append_serializes_with_indented_utf8(tmp_path: Path) -> None:
    """The on-disk manifest must remain human-readable and non-ASCII-safe
    (so Arabic titles round-trip without \\uXXXX escapes)."""
    entry = _entry("doc-ar").model_copy(update={"title_ar": "العنوان"})
    append_manifest_entry(tmp_path, entry)
    text = manifest_path(tmp_path).read_text(encoding="utf-8")
    # Pretty-printed:
    assert "\n  " in text
    # Arabic preserved verbatim:
    assert "العنوان" in text
    # And the file is valid JSON.
    parsed = json.loads(text)
    assert parsed[0]["doc_id"] == "doc-ar"
