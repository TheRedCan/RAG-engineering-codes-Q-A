"""Load and validate the document manifest.

The manifest is the single source of truth for what corpus the system covers.
The machine-readable file is ``data/raw/manifest.json`` — that is what the
fetch script reads and what this module loads. A sibling ``MANIFEST.md`` in
the same directory documents licensing and provenance for human readers; it
is not consumed by code.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from common.errors import ConfigError
from common.models import DocumentMeta

_MANIFEST_FILENAME = "manifest.json"


def manifest_path(raw_dir: Path) -> Path:
    return raw_dir / _MANIFEST_FILENAME


def append_manifest_entry(raw_dir: Path, entry: DocumentMeta) -> None:
    """Add ``entry`` to the manifest, validating uniqueness.

    Used by the Streamlit "Add document" page when an engineer uploads a
    bring-your-own PDF. The file is written atomically (.tmp + replace)
    so a Ctrl-C can't leave the manifest half-written.

    Raises:
        ConfigError: if the manifest is unreadable, malformed, or already
            contains an entry with the same ``doc_id``.
    """
    existing = load_manifest(raw_dir) if manifest_path(raw_dir).exists() else []
    if any(e.doc_id == entry.doc_id for e in existing):
        raise ConfigError(
            f"manifest already contains doc_id={entry.doc_id!r}; "
            "remove the existing entry first or pick a different id"
        )
    existing.append(entry)
    payload = [e.model_dump(mode="json", exclude_none=False) for e in existing]
    path = manifest_path(raw_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_manifest(raw_dir: Path) -> list[DocumentMeta]:
    """Read and validate the manifest. Raises ConfigError on any problem.

    Validation guarantees the caller can rely on:

    - File exists and is valid JSON.
    - Top-level value is a JSON array.
    - Every entry conforms to ``DocumentMeta``.
    - ``doc_id`` values are unique.
    """
    path = manifest_path(raw_dir)
    if not path.is_file():
        raise ConfigError(f"manifest not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"manifest is not valid JSON: {path}: {e}") from e

    if not isinstance(raw, list):
        raise ConfigError(f"manifest top-level must be a JSON array: {path}")

    adapter = TypeAdapter(list[DocumentMeta])
    try:
        entries = adapter.validate_python(raw)
    except Exception as e:  # pydantic ValidationError — re-raise as ConfigError
        raise ConfigError(f"manifest entries failed validation: {e}") from e

    ids = [e.doc_id for e in entries]
    duplicates = {x for x in ids if ids.count(x) > 1}
    if duplicates:
        raise ConfigError(f"manifest has duplicate doc_ids: {sorted(duplicates)}")

    return entries
