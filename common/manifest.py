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
