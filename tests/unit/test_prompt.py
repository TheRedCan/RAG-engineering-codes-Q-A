"""Unit tests for generation.prompt."""

from __future__ import annotations

import json

import pytest

from common.errors import LlmOutputError
from common.models import Chunk, Language, RetrievedChunk
from generation.prompt import (
    LlmDraftAnswer,
    build_user_prompt,
    llm_response_schema,
    parse_llm_response,
)


def _make_retrieved(
    chunk_id: str = "doc#00001",
    doc_id: str = "fema-x",
    page_numbers: list[int] | None = None,
    text: str = "Some chunk content.",
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_numbers=page_numbers or [1, 2],
            section_path=None,
            text=text,
            language=Language.EN,
            char_count=len(text),
        ),
        rank=rank,
    )


# ==================== prompt building ====================


def test_build_user_prompt_includes_indices_and_doc_metadata() -> None:
    chunks = [
        _make_retrieved(chunk_id="a", text="alpha text", page_numbers=[1]),
        _make_retrieved(chunk_id="b", text="beta text", page_numbers=[7, 8]),
    ]
    out = build_user_prompt("what is X?", chunks)

    assert "what is X?" in out
    assert 'index="1"' in out
    assert 'index="2"' in out
    assert "alpha text" in out
    assert "beta text" in out
    assert 'pages="7, 8"' in out


def test_build_user_prompt_caps_chunk_count() -> None:
    """We never send more than _MAX_CHUNKS_IN_PROMPT chunks (default 8)."""
    chunks = [_make_retrieved(chunk_id=f"c{i}", text=f"t{i}", rank=i + 1) for i in range(20)]
    out = build_user_prompt("q", chunks)
    assert 'index="8"' in out
    assert 'index="9"' not in out


def test_llm_response_schema_is_valid_json_schema() -> None:
    schema = llm_response_schema()
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "claims" in schema.get("properties", {})


# ==================== response parsing ====================


_TWO_CHUNKS = [
    _make_retrieved(chunk_id="alpha", doc_id="doc-1", page_numbers=[5]),
    _make_retrieved(chunk_id="beta", doc_id="doc-2", page_numbers=[10, 11]),
]


def test_parse_resolves_indices_to_full_citations() -> None:
    raw = json.dumps(
        {
            "answer_language": "en",
            "claims": [
                {"text": "Claim one.", "cites": [1]},
                {"text": "Claim two.", "cites": [1, 2]},
            ],
        }
    )
    answer = parse_llm_response(raw, "Q?", _TWO_CHUNKS, hop_count=2)

    assert answer.question == "Q?"
    assert answer.answer_language == Language.EN
    assert answer.hop_count == 2
    assert len(answer.claims) == 2
    # Index 1 -> chunk "alpha"; index 2 -> chunk "beta"
    assert answer.claims[0].citations[0].chunk_id == "alpha"
    assert answer.claims[0].citations[0].doc_id == "doc-1"
    assert answer.claims[0].citations[0].page_numbers == [5]
    assert {c.chunk_id for c in answer.claims[1].citations} == {"alpha", "beta"}
    # used_chunks union, sorted
    assert answer.used_chunks == ["alpha", "beta"]


def test_parse_empty_claims_is_a_valid_refusal() -> None:
    """An empty claims list represents the LLM's structured 'I can't answer
    from these sources' — must NOT raise."""
    raw = json.dumps({"answer_language": "en", "claims": []})
    answer = parse_llm_response(raw, "Q?", _TWO_CHUNKS, hop_count=1)
    assert answer.claims == []
    assert answer.used_chunks == []


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(LlmOutputError, match="not valid JSON"):
        parse_llm_response("not json", "Q?", _TWO_CHUNKS, hop_count=1)


def test_parse_rejects_missing_required_field() -> None:
    raw = json.dumps({"claims": []})  # missing answer_language
    with pytest.raises(LlmOutputError, match="schema validation failed"):
        parse_llm_response(raw, "Q?", _TWO_CHUNKS, hop_count=1)


def test_parse_rejects_out_of_range_citation() -> None:
    """LLM cited chunk index 5 but we only sent 2 chunks. Must fail loudly
    — silently dropping the citation would hide a model error."""
    raw = json.dumps({"answer_language": "en", "claims": [{"text": "x", "cites": [5]}]})
    with pytest.raises(LlmOutputError, match=r"out of range|only 2 chunks were provided"):
        parse_llm_response(raw, "Q?", _TWO_CHUNKS, hop_count=1)


def test_parse_rejects_claim_with_no_citations() -> None:
    """LlmCitation schema enforces min_length=1 on cites."""
    raw = json.dumps({"answer_language": "en", "claims": [{"text": "x", "cites": []}]})
    with pytest.raises(LlmOutputError):
        parse_llm_response(raw, "Q?", _TWO_CHUNKS, hop_count=1)


def test_draft_schema_round_trips() -> None:
    """The draft schema should accept the canonical shape we ask of the LLM."""
    obj = LlmDraftAnswer.model_validate(
        {"answer_language": "ar", "claims": [{"text": "نص", "cites": [1]}]}
    )
    assert obj.answer_language == Language.AR
    assert obj.claims[0].text == "نص"
