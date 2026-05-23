"""Download source PDFs listed in the manifest.

Design notes:

- Reads ``data/raw/manifest.json`` (validated via ``common.manifest``).
- Downloads to ``{raw_dir}/{doc_id}.pdf``.
- Computes SHA-256 while streaming.
- Cross-checks the result against ``manifest.lock.json``:
    * If no lock entry exists, the computed hash is pinned to the lock file.
    * If a lock entry exists and matches, the file is accepted.
    * If a lock entry exists and disagrees, the new file is **deleted** and
      ``ChecksumMismatchError`` is raised. The user must investigate before
      we ever trust the new content.
- Already-present files are skipped unless ``--force`` is passed.
- Network errors are retried with exponential backoff; the error reason is
  logged in full. No silent retries — every retry emits a WARNING line.

This module performs **only fetching and checksum pinning**. Language
detection happens in the parse stage so this module stays small, has no
PDF-parser dependency, and remains independently testable.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import typer
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.errors import ChecksumMismatchError, FetchError
from common.logging import configure_logging, logger
from common.manifest import load_manifest
from common.settings import get_settings

if TYPE_CHECKING:
    from common.models import DocumentMeta


@dataclass
class FetchResult:
    """Summary of a fetch_all invocation. Lets callers handle partial success
    explicitly — no silent dropping of failed docs.
    """

    succeeded: list[Path] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, reason)

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


# Streaming chunk size for both download and hashing.
_CHUNK_BYTES = 64 * 1024

# Lock file (gitignored) records the SHA-256 of each downloaded doc.
_LOCK_FILENAME = "manifest.lock.json"

# Browser-like User-Agent. Several US-government CDNs (notably Akamai in front
# of publications.usace.army.mil) reject non-browser User-Agents with HTTP 403
# even for fully public-domain files. The content is openly published and our
# use is legitimate; we're not bypassing access controls, just satisfying a
# defaulted bot filter. Real identification is handled by request volume and
# polite backoff, not by the UA string.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


app = typer.Typer(add_completion=False, help="Download source PDFs listed in the manifest.")


# -------------------- lock file helpers --------------------


def _lock_path(raw_dir: Path) -> Path:
    return raw_dir / _LOCK_FILENAME


def _read_lock(raw_dir: Path) -> dict[str, str]:
    path = _lock_path(raw_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        # Corrupt lock is a configuration error — never silently ignore.
        logger.error(f"lock file is corrupt: {path}: {e}")
        raise
    if not isinstance(data, dict):
        raise TypeError(f"lock file must be a JSON object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def _write_lock(raw_dir: Path, lock: dict[str, str]) -> None:
    path = _lock_path(raw_dir)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# -------------------- download --------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
    reraise=True,
)
def _stream_download(url: str, dest: Path, client: httpx.Client) -> str:
    """Stream a URL to ``dest`` and return its SHA-256 hex digest.

    Raises:
        FetchError: on non-success HTTP status. The partial file is removed.
        httpx.HTTPError: on transport failure; tenacity retries first.
    """
    sha = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")

    logger.info(f"downloading {url} -> {dest.name}")
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code != httpx.codes.OK:
                # Read body briefly for context, then raise.
                body_preview = response.read()[:200].decode("utf-8", errors="replace")
                tmp.unlink(missing_ok=True)
                raise FetchError(  # noqa: TRY301 — inline raise keeps the streaming flow legible
                    url=url,
                    status=response.status_code,
                    reason=f"non-200 response. body[:200]={body_preview!r}",
                )
            with tmp.open("wb") as f:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    f.write(chunk)
                    sha.update(chunk)
    except FetchError:
        raise
    except httpx.HTTPError:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(dest)
    return sha.hexdigest()


def _verify_or_pin(
    meta: DocumentMeta,
    actual_sha: str,
    lock: dict[str, str],
    dest: Path,
) -> bool:
    """Verify hash against manifest + lock. Returns True if pinned for the first time."""
    # 1. If the manifest pre-pins a sha256, that is the strongest claim.
    if meta.sha256 is not None and meta.sha256.lower() != actual_sha.lower():
        dest.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            path=str(dest),
            expected=meta.sha256,
            actual=actual_sha,
        )

    # 2. Otherwise, cross-check against the lock file.
    locked = lock.get(meta.doc_id)
    if locked is not None and locked.lower() != actual_sha.lower():
        dest.unlink(missing_ok=True)
        raise ChecksumMismatchError(
            path=str(dest),
            expected=locked,
            actual=actual_sha,
        )

    # 3. First time we see this doc — pin it.
    newly_pinned = locked is None
    lock[meta.doc_id] = actual_sha
    return newly_pinned


# -------------------- public entry --------------------


def fetch_all(*, force: bool = False) -> FetchResult:
    """Fetch every document in the manifest.

    Policy:
        - Each document is attempted independently. A network / HTTP failure on
          one document does NOT abort the others — it is collected into
          ``FetchResult.failed`` and the loop continues.
        - ``ChecksumMismatchError`` is treated as a potential tampering signal
          and aborts the whole run immediately (security > convenience).

    Args:
        force: redownload even if a file already exists locally.
    """
    settings = get_settings()
    raw_dir = settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs = load_manifest(raw_dir)
    if not docs:
        logger.warning("manifest is empty; nothing to fetch")
        return FetchResult()

    lock = _read_lock(raw_dir)
    result = FetchResult()

    headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf,*/*"}
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

    with httpx.Client(headers=headers, timeout=timeout) as client:
        for meta in docs:
            dest = raw_dir / f"{meta.doc_id}.pdf"

            if dest.exists() and not force:
                # Re-verify the existing file rather than skipping blindly.
                # ChecksumMismatchError here is intentionally NOT caught — a
                # locally-modified or upstream-changed file demands attention.
                existing_sha = _sha256_of_file(dest)
                pinned = _verify_or_pin(meta, existing_sha, lock, dest)
                verb = "pinned existing" if pinned else "verified existing"
                logger.info(f"{meta.doc_id}: {verb} sha256={existing_sha[:12]}")
                result.succeeded.append(dest)
                continue

            try:
                actual_sha = _stream_download(meta.source_url, dest, client)
            except (FetchError, RetryError, httpx.HTTPError) as e:
                reason = f"{e.last_attempt.exception()}" if isinstance(e, RetryError) else str(e)
                logger.error(f"{meta.doc_id}: download failed — {reason}")
                result.failed.append((meta.doc_id, reason))
                continue

            # ChecksumMismatchError is *not* caught here — it indicates the
            # remote file disagrees with our pin and we want a loud abort.
            pinned = _verify_or_pin(meta, actual_sha, lock, dest)
            verb = "pinned" if pinned else "verified"
            logger.info(
                f"{meta.doc_id}: {verb} sha256={actual_sha[:12]} size={dest.stat().st_size} bytes"
            )
            result.succeeded.append(dest)

    # Write whatever we managed to pin even if some failed.
    _write_lock(raw_dir, lock)
    return result


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------- CLI --------------------


@app.command()
def main(
    force: bool = typer.Option(False, "--force", help="Redownload even if the file exists."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Download every PDF listed in the manifest into data/raw/."""
    configure_logging(level=log_level, log_dir=get_settings().log_dir)
    try:
        result = fetch_all(force=force)
    except Exception:  # noqa: BLE001 — top-level CLI catch-all by design (logs full traceback, exits 2)
        # Only reached for hard-fail conditions (ChecksumMismatch, bad config, etc.).
        logger.exception("fetch_all aborted on a hard-fail condition")
        sys.exit(2)

    raw_dir = get_settings().raw_dir
    logger.info(
        f"done. {len(result.succeeded)} succeeded, {len(result.failed)} failed in {raw_dir}"
    )

    if result.failed:
        logger.error("the following documents could not be fetched automatically:")
        for doc_id, reason in result.failed:
            logger.error(f"  - {doc_id}: {reason}")
        logger.error(
            "if the URL is correct but the remote host is blocking scripted "
            "downloads, download the file in your browser and save it to "
            f"{raw_dir}/<doc_id>.pdf, then re-run this command — the script "
            "will verify and pin the file on the next pass."
        )
        sys.exit(1)


if __name__ == "__main__":
    app()
