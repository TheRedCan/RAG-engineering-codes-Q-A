# Document Manifest

This directory is where source PDFs live on your machine. **No PDFs are
committed to this repository.** The fetch script downloads them from official
sources into your local copy of this directory, which is gitignored.

## Machine-readable manifest

The actual list of documents lives in [`manifest.json`](manifest.json) next to
this file. The fetch script (`python -m ingest.fetch`) reads that file; this
Markdown document is for human readers.

## Public corpus (v0.1)

These are **US Federal Government works** (public domain) covering structural
engineering for new buildings, seismic design, concrete stability, and
resilience standards. They are full-text, freely downloadable, and legally
redistributable — but we still keep them out of the repo to avoid bloat and to
let users always pull the latest official versions.

| Doc ID | Source | Topic | Approx pages |
|---|---|---|---|
| `fema-p-2082-vol1-2020` | FEMA P-2082 Vol 1 (2020) | NEHRP Seismic Provisions + Commentary | ~600 |
| `fema-nehrp-design-examples-vol1-2020` | FEMA NEHRP 2020 Design Examples Vol 1 | Worked seismic design examples + flow charts | ~700 |
| `nist-tn-2209-2022` | NIST TN 2209 (2022) | Assessment of Resilience in Building Codes | ~230 |

## Bring-your-own corpus (Egyptian / Saudi codes)

The Egyptian Code of Practice (ECP) and the Saudi Building Code (SBC) are
**not redistributed by this project** — ECP is sold by the Egyptian HBRC and
the freely-published SBC content is gated behind a DRM viewer on sbc.gov.sa.

If you have legitimate access (purchase, employer, university licence), you
can extend the corpus locally:

1. Drop the PDF(s) into this directory.
2. Add an entry to `manifest.json`. Use `"family": "ecp"` or `"family": "sbc"`.
3. Either pre-pin the `sha256` to whatever the file's SHA-256 is, or leave it
   `null` and the fetch script will pin it on first verification pass.
4. Re-run the ingest pipeline. The Arabic-aware parsing and embedding work
   the same way for any PDF you supply.

Do not commit these PDFs — `.gitignore` already excludes them.

## Checksum lock

The first time you fetch a document, its SHA-256 is recorded in
`manifest.lock.json` (gitignored). Subsequent fetches verify against the lock
file. If a remote PDF changes content (e.g. an erratum is published) the
verification fails loudly and the file is deleted — you must consciously
re-pin by clearing the lock entry. This catches both upstream tampering and
silent content drift.
