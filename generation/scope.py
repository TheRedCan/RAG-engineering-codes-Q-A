"""Scope verifier — final safety net for out-of-scope answers.

This is *not* a grounding check (each claim against its cited chunk).
We tried that earlier and it over-rejected good paraphrased answers
(see the removed generation/verify.py). This module asks a much
narrower question:

    "Does this answer actually address the user's SPECIFIC question?"

Three concrete failure modes we want to catch:

1. **Jurisdiction mismatch.** User asks about the Egyptian Code of
   Practice (or any specific code body not in our corpus); model
   produces a tangential answer from US FEMA documents.
2. **Made-up source.** User cites a specific paper / author / report
   that isn't in the corpus; model fabricates an answer "from" it.
3. **Made-up section.** User references a section number that doesn't
   exist (e.g. ASCE §99.99.99); model invents content for it.

Design choices to avoid the last verifier's over-rejection trap:

- **Default to PASS** on uncertainty, on malformed verifier output, or
  on LLM error. The cost of an extra refusal is bigger than the cost
  of an undetected scope drift.
- **One LLM call** with structured JSON output (Ollama's `format` mode).
- **Only refuse on explicit mismatch.** The verifier must point to a
  specific trigger; vague gut-feelings of "this seems off" get PASS.
- **No-op for empty answers.** If the model already refused (claims=[]),
  there's nothing to verify.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from common.logging import logger
from common.models import Answer
from generation.llm import chat

_SCOPE_SYSTEM = (
    "You are a scope-verification utility. You decide whether an answer "
    "is on-topic for the user's question.\n\n"
    "You return REFUSE only in these explicit cases:\n"
    "1. The user named a specific code body, jurisdiction, or standard "
    "(e.g. 'Egyptian Code', 'Saudi Building Code', 'British Standard') "
    "AND the answer's citations are from a DIFFERENT code body (FEMA, "
    "ASCE, NIST) without acknowledging the mismatch.\n"
    "2. The user named a specific paper, author, report, or document "
    "(e.g. 'the Smith 2024 paper', 'Report X-123') AND the answer's "
    "citations don't include that source. Generic answers about the "
    "topic don't satisfy a source-specific question.\n"
    "3. The user referenced a specific code section by number (e.g. "
    "'ASCE 7-22 Section 99.99.99', '§14.5.3.2') AND no claim or "
    "citation in the answer mentions that section. A non-existent or "
    "out-of-scope section yields a fabricated answer.\n\n"
    "You return PASS in EVERY other case. In particular:\n"
    "- Paraphrasing, summarization, and reformatting are fine.\n"
    "- Different writing style than the question is fine.\n"
    "- Citations from a different but RELATED document in the same "
    "code body are fine (e.g. ASCE 7 question answered with content "
    "from a FEMA NEHRP commentary).\n"
    "- If you can't identify a clear, explicit mismatch in one of the "
    "three categories above, return PASS.\n"
    "- If in doubt, return PASS. False refusal costs the user a real "
    "answer; false acceptance only costs a slightly off-topic reply.\n\n"
    "Output strictly the JSON schema you are given."
)


class _ScopeVerdict(BaseModel):
    """Structured verifier output. ``reason`` is one short sentence for logs."""

    model_config = ConfigDict(extra="forbid")

    verdict: str = Field(pattern=r"^(pass|refuse)$")
    reason: str = Field(min_length=1, max_length=500)


def _schema() -> dict[str, Any]:
    return _ScopeVerdict.model_json_schema()


def _build_user_prompt(question: str, answer: Answer) -> str:
    """Compose the verifier's user message: question + answer text + citation summary."""
    claim_text = "\n".join(f"- {c.text}" for c in answer.claims)
    cited_docs = sorted({c.doc_id for cl in answer.claims for c in cl.citations})
    cited_sections = sorted(
        {c.section_path or "" for cl in answer.claims for c in cl.citations}
    )
    cited_sections = [s for s in cited_sections if s]
    section_line = (
        f"Cited sections: {cited_sections}" if cited_sections else "Cited sections: none"
    )
    return (
        f"User question:\n{question}\n\n"
        f"Answer claims:\n{claim_text}\n\n"
        f"Cited documents: {cited_docs}\n"
        f"{section_line}\n\n"
        "Verdict (pass / refuse) per your rules:"
    )


def verify_scope(question: str, answer: Answer) -> Answer:
    """Return ``answer`` if it's in-scope, or an empty-claims refusal if not.

    Defaults to PASS (returning ``answer`` unchanged) on:
    - empty answer (nothing to verify)
    - LLM errors / timeouts
    - malformed verifier output

    This is intentional: under uncertainty, a slightly off-topic answer
    is better than nuking a useful one. The verifier only refuses on
    explicit triggers it can articulate.
    """
    if not answer.claims:
        return answer

    try:
        raw = chat(
            system=_SCOPE_SYSTEM,
            user=_build_user_prompt(question, answer),
            json_schema=_schema(),
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 — verifier failure must not break the pipeline
        logger.warning(f"scope verifier LLM call failed: {e}; defaulting to PASS")
        return answer

    try:
        data = json.loads(raw)
        verdict = _ScopeVerdict.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"scope verifier output malformed: {e}; defaulting to PASS")
        return answer

    if verdict.verdict == "pass":
        logger.info(f"scope verifier PASS: {verdict.reason}")
        return answer

    logger.info(f"scope verifier REFUSE: {verdict.reason}")
    return Answer(
        question=answer.question,
        answer_language=answer.answer_language,
        claims=[],
        used_chunks=[],
        hop_count=answer.hop_count,
    )
