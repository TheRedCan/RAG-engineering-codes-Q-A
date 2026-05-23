"""Language detection helpers shared by every stage that classifies text.

We constrain the output to the project's supported set (Arabic, English,
UNKNOWN) on purpose: silently accepting a third language would let bad
documents creep into the index without anyone noticing.
"""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

from common.models import Language

# langdetect's default behaviour is non-deterministic. Seed it so the same
# input produces the same label across runs and across processes — important
# for reproducible chunking and for stable tests.
DetectorFactory.seed = 0

# Pages or chunks shorter than this cannot be reliably classified.
# 30 chars is short enough to admit table cells and captions, long enough
# that langdetect can settle on a confident answer for prose.
MIN_CHARS_FOR_DETECTION = 30

# langdetect emits ISO 639-1 codes. We map only the languages we support.
# Anything else (fr, de, zh, ...) returns UNKNOWN, which the parser uses
# as a signal to refuse the document rather than silently accept it.
_LANGDETECT_TO_ENUM: dict[str, Language] = {
    "ar": Language.AR,
    "en": Language.EN,
}


def detect_language(text: str) -> Language:
    """Detect language from text.

    Returns ``Language.UNKNOWN`` for inputs too short to classify reliably,
    inputs langdetect cannot handle, or languages outside our supported set.
    """
    if len(text.strip()) < MIN_CHARS_FOR_DETECTION:
        return Language.UNKNOWN
    try:
        code = detect(text)
    except LangDetectException:
        return Language.UNKNOWN
    return _LANGDETECT_TO_ENUM.get(code, Language.UNKNOWN)
