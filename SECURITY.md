# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in this repository, please **do not** open
a public GitHub issue. Instead, email the maintainer privately:

- **Contact**: omarbassel2020@gmail.com
- **Subject prefix**: `[SECURITY] engineering-codes-rag`

Please include:

1. A description of the vulnerability and its potential impact.
2. Steps to reproduce.
3. Affected versions / commit SHA.
4. Any suggested mitigations.

You can expect an acknowledgement within 7 days. A fix timeline will be agreed
on a per-issue basis depending on severity.

## Scope

In scope:

- Code in this repository.
- Default configuration files committed to this repository.
- The fetch and ingestion pipelines.

Out of scope:

- Vulnerabilities in upstream dependencies (report those to their respective
  maintainers; we will track via Dependabot).
- Vulnerabilities in the LLM model weights themselves.
- The content of any third-party documents the user chooses to ingest.

## Security model assumptions

This project is designed for **local single-user use or trusted LAN deployment**.
The following are explicitly out of scope and not protected against:

- **Authentication / authorization**: there is none. Do not expose the service
  to the public internet without a reverse proxy that adds authentication.
- **Multi-tenant isolation**: all ingested documents are queryable by anyone
  who can reach the service.
- **Malicious input documents**: ingest only PDFs you trust. PDF parsing is a
  known attack surface. JavaScript and embedded file execution are disabled in
  the parser, but a determined attacker with control over input PDFs could
  still target parser vulnerabilities. Source PDFs only from official, trusted
  channels (e.g. fema.gov, nvlpubs.nist.gov, or your verified internal copy
  of a code document).
- **Prompt injection from documents**: retrieved chunks are wrapped in
  delimited tags in the LLM prompt and the system prompt instructs the model
  to treat retrieved content as data, not instructions. This is best-effort
  mitigation, not a guarantee.

## What this project does to reduce risk

- No outbound network calls at query time (fully local inference).
- No telemetry.
- Secrets scanning via gitleaks pre-commit hook.
- Dependency vulnerability scanning via Dependabot and `pip-audit` in CI.
- Static analysis via CodeQL in CI.
- All third-party PDF URLs are declared in `data/raw/manifest.json` (the
  machine-readable manifest; `MANIFEST.md` is the human-readable companion).
- The first verified download of each file pins its SHA-256 to
  `data/raw/manifest.lock.json`; subsequent runs verify against the lock.
- Fetch script fails loudly and deletes the file on any checksum mismatch.
