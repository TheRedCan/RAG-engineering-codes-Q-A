"""Unit tests for generation.translate.

The LLM call is mocked; we only verify the wiring rules:

- English / unknown queries → pass-through, no LLM call.
- Arabic query → LLM is invoked; quotes and outer whitespace stripped.
- Empty / oversized LLM output → LlmOutputError.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from common.errors import LlmOutputError
from common.models import Language
from generation.translate import (
    _MAX_TRANSLATION_CHARS,
    translate_query_for_retrieval,
)


def test_english_query_passes_through_without_llm_call() -> None:
    """English input must not waste an LLM call on translation."""
    with patch("generation.translate.chat") as chat_mock:
        out_query, lang = translate_query_for_retrieval(
            "How is the seismic design coefficient Cs calculated under ASCE 7?"
        )

    chat_mock.assert_not_called()
    assert out_query.startswith("How is the seismic")
    assert lang == Language.EN


def test_unknown_language_passes_through_without_llm_call() -> None:
    """Short / unclassifiable inputs must not be translated; we have no
    confident source language to translate FROM."""
    with patch("generation.translate.chat") as chat_mock:
        out_query, lang = translate_query_for_retrieval("???")

    chat_mock.assert_not_called()
    assert out_query == "???"
    assert lang == Language.UNKNOWN


def test_arabic_query_is_translated_and_lang_returned() -> None:
    """Non-EN goes through the LLM; we return the cleaned English text
    and the ORIGINAL language so the caller knows to keep the original
    question for the answer pass."""
    arabic = "ما هي تركيبات الأحمال للتصميم الزلزالي في الكود الأمريكي ASCE 7-22؟"
    with patch(
        "generation.translate.chat",
        return_value="  What are the load combinations for seismic design in ASCE 7-22?  ",
    ) as chat_mock:
        out_query, lang = translate_query_for_retrieval(arabic)

    chat_mock.assert_called_once()
    assert out_query == "What are the load combinations for seismic design in ASCE 7-22?"
    assert lang == Language.AR


def test_translation_strips_surrounding_quotes() -> None:
    """Some small models wrap output in quotes despite the instruction."""
    arabic = "ما هي تركيبات الأحمال للتصميم الزلزالي في الكود الأمريكي؟"
    with patch("generation.translate.chat", return_value='"What are load combos?"'):
        out_query, _ = translate_query_for_retrieval(arabic)

    assert out_query == "What are load combos?"


def test_empty_translation_raises_llm_output_error() -> None:
    arabic = "ما هي تركيبات الأحمال للتصميم الزلزالي في الكود الأمريكي؟"
    with (
        patch("generation.translate.chat", return_value="   "),
        pytest.raises(LlmOutputError),
    ):
        translate_query_for_retrieval(arabic)


def test_oversized_translation_raises_llm_output_error() -> None:
    """Defends against the model adding commentary instead of translating."""
    arabic = "ما هي تركيبات الأحمال للتصميم الزلزالي في الكود الأمريكي؟"
    bloated = "x" * (_MAX_TRANSLATION_CHARS + 1)
    with (
        patch("generation.translate.chat", return_value=bloated),
        pytest.raises(LlmOutputError),
    ):
        translate_query_for_retrieval(arabic)
