# Notices and Attributions

## This software

Copyright 2026 Omar Bassel
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## Third-party content NOT included in this repository

This project is a Retrieval-Augmented Generation (RAG) system. The source
documents themselves are **not** distributed with this repository, regardless
of their licensing. The fetch script downloads them into your local
gitignored `data/raw/` directory.

### Public-domain documents (fetched automatically by the fetch script)

These are works of the US Federal Government and therefore **public domain
in the United States** under 17 U.S.C. § 105. Users in other jurisdictions
should confirm their own legal status before redistributing.

- **FEMA P-2082 Vol 1 (NEHRP Recommended Seismic Provisions, 2020)** —
  US Federal Emergency Management Agency. Hosted at fema.gov.
- **FEMA NEHRP 2020 Design Examples, Training Materials and Design Flow
  Charts — Volume 1** — US Federal Emergency Management Agency. Hosted at
  fema.gov.
- **NIST Technical Note 2209 (Assessment of Resilience in Codes, 2022)** —
  US National Institute of Standards and Technology. Hosted at nvlpubs.nist.gov.

### Bring-your-own (NOT fetched by this project)

#### Saudi Building Code (SBC)

The Saudi Building Code is the intellectual property of the Saudi Building Code
National Committee (sbc.gov.sa). While the SBC site advertises "free browsing,"
the full-text PDFs are gated behind a DRM viewer (view.protectedpdf.com) and
some volumes are sold directly. This repository:

- Does **not** redistribute any SBC PDF files.
- Does **not** provide a fetch path for SBC content.
- Will process SBC PDFs if you obtain them legitimately and place them in
  your local `data/raw/` directory.

#### Egyptian Code of Practice (ECP)

The Egyptian Code of Practice is the intellectual property of the Housing and
Building National Research Center (HBRC) of Egypt. ECP documents are sold by
HBRC and are **not** distributed by this project. Users with legitimate access
may place ECP PDFs in their local `data/raw/` directory; the ingestion
pipeline will process them. Do not commit such PDFs.

### Models

- **Qwen2.5-7B-Instruct** — Apache-2.0 (Alibaba Cloud)
- **BAAI/bge-m3** — MIT (Beijing Academy of Artificial Intelligence)
- **BAAI/bge-reranker-v2-m3** — Apache-2.0 (Beijing Academy of Artificial Intelligence)

Models are not redistributed; they are fetched from their original publishers
(Hugging Face / Ollama registry) at install time.
