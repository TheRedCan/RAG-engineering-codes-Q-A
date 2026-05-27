"""Query translation for cross-lingual retrieval.

The corpus we index is overwhelmingly English (FEMA / NIST). BGE-M3 is
multilingual and *can* match an Arabic query against English chunks, but
the cross-lingual signal is much weaker than monolingual — empirically,
Arabic queries that should hit ASCE 7 load-combo equation chunks land
on pointer-style chapter-intro chunks instead.

Cheapest fix that preserves answer language: LLM-translate the query
to English before retrieval, then pass the *original* question to the
answer LLM so its language-detection rule (system prompt #7) still
fires and the user gets Arabic prose back.

Public surface:

    translate_query_for_retrieval(question) -> (retrieval_query, detected_language)

If detection returns EN or UNKNOWN, we pass the original question through
unchanged — no extra LLM call. Only AR (and any future supported non-EN
language) triggers translation.
"""

from __future__ import annotations

from common.errors import LlmOutputError
from common.language import detect_language
from common.logging import logger
from common.models import Language
from generation.llm import chat

_TRANSLATE_SYSTEM = (
    "You are a translation utility for an engineering-codes retrieval system. "
    "Translate the user's question into clear, technical English suitable for "
    "querying a corpus of building-code and structural-engineering documents. "
    "Preserve code identifiers (ASCE 7, FEMA P-2082, §12.8.1, etc.), equation "
    "symbols, and numerical values exactly as-is. Output ONLY the translated "
    "English question — no preamble, no quotes, no explanation."
)

# Hard cap on translation output. A reasonable English rendering of a
# realistic engineer's question is well under this; anything longer is
# almost certainly the model adding commentary despite the instruction.
_MAX_TRANSLATION_CHARS = 1000


def translate_query_for_retrieval(question: str) -> tuple[str, Language]:
    """Return ``(retrieval_query, detected_language)``.

    - English / unknown input → pass-through, no LLM call.
    - Arabic input → LLM translation to English; original language returned
      so callers can keep the original question for the answer-language
      pass.

    Raises:
        LlmOutputError: if translation produced empty or implausibly long
            output. We refuse to silently retrieve on a bad translation.
    """
    lang = detect_language(question)
    if lang in (Language.EN, Language.UNKNOWN):
        return question, lang

    logger.info(f"translating non-EN query (lang={lang.value}) to English for retrieval")
    raw = chat(system=_TRANSLATE_SYSTEM, user=question, temperature=0.0)
    translated = raw.strip()
    # Some models wrap the answer in quotes despite the instruction.
    # `2` is the minimum length to have a matching opening + closing quote.
    _MIN_QUOTED_LEN = 2
    if (
        len(translated) >= _MIN_QUOTED_LEN
        and translated[0] == translated[-1]
        and translated[0] in ('"', "'")
    ):
        translated = translated[1:-1].strip()

    if not translated:
        raise LlmOutputError(reason="translation produced empty output", raw_output=raw)
    if len(translated) > _MAX_TRANSLATION_CHARS:
        raise LlmOutputError(
            reason=(
                f"translation exceeded {_MAX_TRANSLATION_CHARS} chars "
                f"(got {len(translated)}); likely commentary, not translation"
            ),
            raw_output=raw,
        )

    logger.info(f"translation: {len(question)}c {lang.value} -> {len(translated)}c en")
    return translated, lang
