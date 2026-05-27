"""Streamlit page: upload a bring-your-own PDF and run it through the
full ingest pipeline (fetch-verify → parse → chunk → embed) so it's
queryable from the chat UI.

Design choices:

- We never download for BYO. The file lands in ``data/raw/{doc_id}.pdf``
  before the manifest entry is appended; the fetch stage then just
  hashes + pins it (its existing "file already on disk" code path).
- ``source_url`` is required by ``DocumentMeta``. For BYO we stamp it as
  ``byo://local-upload/{doc_id}`` so the manifest is honest about the
  provenance — re-running fetch on a deleted BYO file will fail loudly
  with a useful error rather than silently no-op.
- Each pipeline stage is invoked with ``only_doc_id=`` so we don't
  re-process the rest of the corpus. The embed stage's resume logic
  (existing chunk_ids) prevents duplicate vectors if the user retries.
- Validation: doc_id must be a slug (lowercase, dashes/underscores).
  A clashing doc_id is rejected at the manifest layer, not silently
  overwritten.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import streamlit as st

from common.errors import (
    ChecksumMismatchError,
    ConfigError,
    EngineeringCodesRagError,
)
from common.logging import logger
from common.manifest import append_manifest_entry, load_manifest
from common.models import CodeFamily, DocumentMeta, Language
from common.settings import get_settings
from ingest.chunk import chunk_all
from ingest.embed import embed_all
from ingest.fetch import fetch_all
from ingest.parse import parse_all

# Slug pattern enforced for doc_id: lowercase letters, digits, dashes,
# underscores. Mirrors the convention used by the public-corpus manifest
# entries (``fema-p-2082-vol1-2020``, etc.) so filenames stay portable
# across operating systems and command-line tools.
_DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,79}$")

# Streamlit's default upload limit is 200 MB. Engineering codes are
# rarely larger, but we surface the cap in the UI so users hitting a
# 600-page commentary aren't left guessing.
_MAX_UPLOAD_MB = 200


def _slugify(stem: str) -> str:
    """Best-effort suggested doc_id from an uploaded filename's stem."""
    s = stem.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:80] or "untitled"


def _validate_doc_id(doc_id: str, raw_dir: Path) -> str | None:
    """Return a user-facing error string, or ``None`` if the id is OK."""
    if not _DOC_ID_RE.match(doc_id):
        return (
            "doc_id must be 2-80 chars, lowercase letters / digits / "
            "dashes / underscores only (e.g. ecp-203-2020)."
        )
    try:
        existing = load_manifest(raw_dir)
    except ConfigError:
        # Manifest missing entirely is fine on first-ever BYO upload.
        return None
    if any(e.doc_id == doc_id for e in existing):
        return f"doc_id={doc_id!r} is already in the manifest."
    return None


def _run_stage(
    label: str,
    runner: Any,
    only_doc_id: str,
    status: Any,
) -> bool:
    """Run a pipeline stage, surface its result in the status box.

    ``runner`` is a zero-arg callable so each stage's call signature
    (fetch_all takes no per-doc kwarg, the others take ``only_doc_id``)
    is normalized at the call site. Returns True iff the stage reported
    the doc_id as succeeded.
    """
    status.update(label=f"{label}…", state="running")
    t0 = time.monotonic()
    try:
        result = runner()
    except EngineeringCodesRagError as e:
        # Any of our typed errors (checksum mismatch, language mismatch,
        # Qdrant down, ...) is shown verbatim — no stack trace.
        status.update(label=f"{label}: failed ({type(e).__name__})", state="error")
        st.error(f"{label} failed: {e}")
        logger.exception(f"{label} failed for {only_doc_id}")
        return False
    except Exception as e:  # noqa: BLE001 — surface anything else as a UI error too
        status.update(label=f"{label}: unexpected error", state="error")
        st.error(f"{label} crashed: {type(e).__name__}: {e}")
        logger.exception(f"{label} crashed for {only_doc_id}")
        return False

    dt = time.monotonic() - t0
    succeeded_ids = {p.stem for p in result.succeeded} if hasattr(result, "succeeded") else set()
    if only_doc_id in succeeded_ids:
        status.update(label=f"{label}: {dt:.1f}s", state="running")
        return True

    # The stage ran but didn't report our doc — typically means a
    # soft-fail collected into result.failed[doc_id, reason].
    failed = getattr(result, "failed", [])
    detail = next((reason for did, reason in failed if did == only_doc_id), "no detail")
    status.update(label=f"{label}: failed", state="error")
    st.error(f"{label} did not succeed for {only_doc_id}: {detail}")
    return False


def _process_pipeline(doc_id: str) -> bool:
    """Run all four ingest stages for one doc_id. Returns overall success.

    fetch_all has no per-doc kwarg — it iterates the whole manifest. For
    a BYO upload that's harmless: the public-corpus files are already on
    disk so it just re-hashes them quickly, then hashes + pins the new
    file. parse / chunk / embed all support ``only_doc_id=`` and are
    scoped to the new doc.
    """
    stages: list[tuple[str, Any]] = [
        ("1/4 verify (hash + pin)", fetch_all),
        ("2/4 parse PDF", lambda: parse_all(only_doc_id=doc_id)),
        ("3/4 chunk text", lambda: chunk_all(only_doc_id=doc_id)),
        ("4/4 embed + upsert to Qdrant", lambda: embed_all(only_doc_id=doc_id)),
    ]
    with st.status("Starting…", expanded=True) as status:
        for label, runner in stages:
            if not _run_stage(label, runner, doc_id, status):
                return False
        status.update(label="Done — document is now queryable.", state="complete")
    return True


def _save_uploaded_file(file_bytes: bytes, dest: Path) -> None:
    """Write atomically so a Ctrl-C / browser-close can't leave a half-written PDF."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(file_bytes)
    tmp.replace(dest)


def _validate_form(
    uploaded: Any,
    doc_id: str,
    title_en: str,
    code_number: str,
    license_note: str,
    raw_dir: Path,
) -> str | None:
    """Return the first user-facing validation error, or ``None`` if OK."""
    if uploaded is None:
        return "Pick a PDF file before submitting."
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if size_mb > _MAX_UPLOAD_MB:
        return f"File is {size_mb:.1f} MB; cap is {_MAX_UPLOAD_MB} MB."
    if err := _validate_doc_id(doc_id, raw_dir):
        return err
    required = {
        "Title (English)": title_en,
        "Code/standard identifier": code_number,
        "License / provenance note": license_note,
    }
    for field_name, value in required.items():
        if not value.strip():
            return f"{field_name} is required."
    return None


def _ingest_submission(
    uploaded: Any,
    doc_id: str,
    family: str,
    code_number: str,
    edition_year: int,
    expected_language: str,
    title_en: str,
    license_note: str,
    raw_dir: Path,
) -> None:
    """Run the full save→manifest→pipeline sequence for one validated submission."""
    dest = raw_dir / f"{doc_id}.pdf"
    if dest.exists():
        st.error(f"{dest} already exists on disk. Remove it manually or pick a different doc_id.")
        return
    _save_uploaded_file(uploaded.getvalue(), dest)
    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    st.toast(f"Wrote {dest.name} ({size_mb:.1f} MB)")

    try:
        entry = DocumentMeta(
            doc_id=doc_id,
            family=CodeFamily(family),
            code_number=code_number.strip(),
            edition_year=int(edition_year),
            variant=None,
            title_en=title_en.strip(),
            title_ar=None,
            source_url=f"byo://local-upload/{doc_id}",
            expected_language=Language(expected_language),
            sha256=None,
            license_note=license_note.strip(),
        )
        append_manifest_entry(raw_dir, entry)
    except ConfigError as e:
        # Roll back the saved file so the user can retry cleanly.
        dest.unlink(missing_ok=True)
        st.error(f"Manifest update failed: {e}")
        return
    st.toast("Manifest updated")

    try:
        ok = _process_pipeline(doc_id)
    except ChecksumMismatchError as e:
        st.error(
            f"Hash verification failed for {doc_id}: {e}. "
            "The file on disk does not match its pinned hash. "
            "This is a security signal — investigate before retrying."
        )
        return

    if ok:
        st.success(
            f"✓ {doc_id} is now in the index. Go to the chat page and ask "
            "a question — it will be retrieved alongside the existing corpus."
        )
    else:
        st.warning(
            "Ingestion did not complete. The PDF is still on disk and the "
            "manifest entry exists; once you fix the issue you can re-run "
            "the failing stage from the command line: "
            f"`python -m ingest.<stage> --only-doc-id {doc_id}`."
        )


def main() -> None:
    st.set_page_config(page_title="Add document — Engineering Codes RAG", layout="wide")
    st.title("Add a document")
    st.caption(
        "Upload a PDF (your own code, standard, or commentary) and ingest it "
        "into the local index. The document becomes queryable from the chat "
        "page as soon as the embed step finishes."
    )

    settings = get_settings()
    raw_dir = settings.raw_dir

    with st.form("ingest_form", clear_on_submit=False):
        uploaded = st.file_uploader(
            "PDF file",
            type=["pdf"],
            help=(
                f"Max {_MAX_UPLOAD_MB} MB. Text-based PDFs only — "
                "scanned-image PDFs are not OCR'd in v0.1."
            ),
        )

        suggested_id = _slugify(Path(uploaded.name).stem) if uploaded else ""

        col1, col2 = st.columns(2)
        with col1:
            doc_id = st.text_input(
                "doc_id (slug)",
                value=suggested_id,
                help="Stable identifier; used as the filename and the citation tag. "
                "Lowercase letters / digits / dashes / underscores.",
            )
            family = st.selectbox(
                "Family",
                options=[f.value for f in CodeFamily],
                index=[f.value for f in CodeFamily].index(CodeFamily.OTHER.value),
                help="Which code family does this document belong to? "
                "Pick OTHER for anything that doesn't fit the named families.",
            )
            code_number = st.text_input(
                "Code/standard identifier",
                placeholder="e.g. ECP 203-2020 or ASCE 41-23",
            )
        with col2:
            title_en = st.text_input(
                "Title (English)",
                placeholder="e.g. Egyptian Code for Concrete Structures",
            )
            edition_year = st.number_input(
                "Edition year", min_value=1900, max_value=2099, value=2020, step=1
            )
            expected_language = st.selectbox(
                "Expected language",
                options=[Language.EN.value, Language.AR.value, Language.MIXED.value],
                help=(
                    "The parse stage cross-checks this against the "
                    "detected language and refuses on mismatch."
                ),
            )
        license_note = st.text_area(
            "License / provenance note",
            placeholder="e.g. Purchased copy, not redistributed. Bring-your-own under "
            "engineer's licence; this PDF stays local on this machine.",
            help="Stored on the manifest entry. Helps you remember the rules later.",
        )
        submitted = st.form_submit_button("Ingest document", type="primary")

    if not submitted:
        return

    err = _validate_form(uploaded, doc_id, title_en, code_number, license_note, raw_dir)
    if err is not None:
        st.error(err)
        return

    _ingest_submission(
        uploaded=uploaded,
        doc_id=doc_id,
        family=family,
        code_number=code_number,
        edition_year=int(edition_year),
        expected_language=expected_language,
        title_en=title_en,
        license_note=license_note,
        raw_dir=raw_dir,
    )


main()
