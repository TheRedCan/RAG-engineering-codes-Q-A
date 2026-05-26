# engineering-codes-rag

Local bilingual (Arabic / English) Retrieval-Augmented Generation system for
building-code and technical engineering documents. Designed for a working
civil / architectural engineer to query technical content and get cited,
multi-source answers.

The **public demo corpus** consists of three full-text, public-domain US
Federal Government documents covering NEHRP seismic design provisions
(FEMA P-2082 Vol 1), worked seismic design examples (FEMA NEHRP Design
Examples Vol 1), and a survey of resilience in building codes
(NIST TN 2209). These are downloaded into your local `data/raw/` directory
by the fetch script and are not redistributed by this repo. See
[`data/raw/MANIFEST.md`](data/raw/MANIFEST.md).

The **Egyptian Code of Practice (ECP)** and **Saudi Building Code (SBC)** are
supported as a **bring-your-own** corpus: the ingestion pipeline (including
Arabic OCR and bilingual embeddings) handles them if you place legally-obtained
PDFs into `data/raw/` and add a manifest entry. The project does not
redistribute either code. See [`NOTICE.md`](NOTICE.md).

> Status: **v0.0.1 — ingest pipeline working end-to-end.** Stages `fetch`,
> `parse`, `chunk`, and `embed` are implemented and tested. `retrieve` and
> `generate` (with multi-hop + citation verifier) are next.

## What the system does

- Fetches the public-domain demo PDFs from fema.gov and nvlpubs.nist.gov
  with streaming SHA-256 verification. Supports bring-your-own corpora
  (drop PDFs into `data/raw/` and add a manifest entry).
- Parses each PDF (Arabic + English, including tables and cross-references).
- Chunks section-aware and indexes hybrid (dense + sparse) vectors in Qdrant.
- Retrieves with hybrid + reranking, follows code cross-references for
  multi-hop questions, and generates a structured answer with **enforced
  citations** (every claim must be grounded in a retrieved chunk).
- Runs **fully locally** — no outbound network calls at query time.

## Architecture

```
fetch  -> parse -> chunk -> embed -> [Qdrant index]
                                          v
                              hybrid retrieve + rerank
                                          v
                               multi-hop reference-following loop
                                          v
                                 generate (Qwen2.5-7B via Ollama)
                                          v
                                citation grounding verifier
                                          v
                                       Answer
```

Each stage is an independent Python module under its own package
(`ingest/`, `retrieval/`, `generation/`, `eval/`, `app/`) with its own CLI
entrypoint and its own tests. Stages communicate via JSONL files on disk
under `data/processed/`, so any stage can be re-run in isolation against the
prior stage's output. Stage I/O schemas live in [`common/models.py`](common/models.py).

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Qwen2.5-7B-Instruct (Q4_K_M GGUF) via **Ollama** | Strong Arabic + English on 8 GB VRAM |
| Embeddings | **BGE-M3** (CPU) | Multilingual; dense + sparse + ColBERT from one model |
| Reranker | **bge-reranker-v2-m3** (CPU) | Same family, multilingual |
| Vector store | **Qdrant** (Docker) | Native hybrid dense+sparse support |
| Validation | **Pydantic v2** | Strict stage I/O contracts |
| Logging | **loguru** | Captures tracebacks; structured JSON on disk |
| Lint / type | **ruff** + **mypy --strict** | Enforces no-silent-failure rules |

## Engineering rules baked into the code

These are not just guidelines — they are enforced by tooling.

1. **No silent failures.** `BLE001` and `TRY` ruff rules ban bare `except:`
   and `except Exception: pass`. Every failure mode has a named exception in
   [`common/errors.py`](common/errors.py).
2. **Strict validation at stage boundaries.** Every Pydantic model uses
   `extra="forbid"`; invalid input fails immediately, never silently.
3. **Warnings are errors.** `pytest` treats warnings as errors, so a
   deprecation never sneaks in unnoticed.
4. **Strict typing.** `mypy --strict` is mandatory.
5. **Health checks at startup**, not at first query. The app refuses to start
   if Qdrant or Ollama is unreachable.
6. **Checksums on every downloaded PDF.** The fetch script pins SHA-256 to
   `data/raw/manifest.lock.json` and deletes any file that fails verification.

## Security model

See [`SECURITY.md`](SECURITY.md). Short version: this is for single-user or
**trusted LAN** use. There is no built-in authentication. Do not expose to the
public internet without a reverse proxy. PDF parsing is an attack surface —
only ingest PDFs from sources you trust.

## Getting started

> Requires Python 3.11 or 3.12. (Project does not yet support 3.13.)

```powershell
# 1. Clone and enter
git clone <your-fork-url>
cd "RAGS project"

# 2. Create an env (conda recommended since you already have miniconda)
conda create -n engineering-codes-rag python=3.12 -y
conda activate engineering-codes-rag

# 3. Install with dev + ingest extras for now
pip install -e ".[dev,ingest]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy env template (optional)
cp .env.example .env

# 6. Fetch the corpus
python -m ingest.fetch
```

The fetch step downloads ~30 MB of PDFs into `data/raw/` from fema.gov and
nvlpubs.nist.gov. These files are gitignored — they will never be committed.
At least one of the FEMA URLs is currently served behind a CDN that rejects
scripted clients; if that happens the fetch script will tell you exactly
which file to download manually in a browser. Re-running fetch after the
manual download verifies and pins it like any other.

## Running tests

```powershell
pytest -m "not integration"      # fast unit tests
pytest                           # all tests (requires Qdrant + Ollama)
```

## Repo layout

```
.
├─ common/                # shared infra: errors, logging, models, settings
├─ ingest/                # fetch -> parse -> chunk -> embed
├─ retrieval/             # hybrid + rerank + multi-hop
├─ generation/            # prompting, citation, grounding verification
├─ eval/                  # test sets + RAGAS harness
├─ app/                   # Streamlit UI
├─ serving/               # docker-compose for Qdrant + Ollama
├─ scripts/               # one-off operational scripts
├─ data/
│  ├─ raw/                # source PDFs (gitignored) + MANIFEST.md + manifest.json
│  └─ processed/          # parsed/chunked JSONL (gitignored)
├─ index/                 # Qdrant persistent storage (gitignored)
├─ models/                # local model cache (gitignored)
├─ tests/
│  ├─ unit/
│  └─ integration/
└─ pyproject.toml
```

## License

Code: Apache-2.0 (see [`LICENSE`](LICENSE)). Third-party content (codes,
models) is not redistributed; see [`NOTICE.md`](NOTICE.md).
