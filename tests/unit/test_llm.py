"""Unit tests for generation.llm.

Tests the wire format against a mocked httpx — no real Ollama needed.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from common.errors import LlmUnavailableError
from generation.llm import chat, health_check


def _tags_url() -> str:
    return "http://127.0.0.1:11434/api/tags"


def _chat_url() -> str:
    return "http://127.0.0.1:11434/api/chat"


# ==================== health_check ====================


@respx.mock
def test_health_check_passes_when_model_present() -> None:
    respx.get(_tags_url()).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen2.5:7b-instruct-q4_K_M"},
                    {"name": "other:tag"},
                ]
            },
        )
    )
    health_check()  # no raise


@respx.mock
def test_health_check_fails_when_model_missing() -> None:
    respx.get(_tags_url()).mock(
        return_value=httpx.Response(200, json={"models": [{"name": "other:tag"}]})
    )
    with pytest.raises(LlmUnavailableError) as ei:
        health_check()
    assert "not found on Ollama" in ei.value.reason
    assert "ollama pull" in ei.value.reason  # actionable instruction


@respx.mock
def test_health_check_fails_when_server_unreachable() -> None:
    respx.get(_tags_url()).mock(side_effect=httpx.ConnectError("nope"))
    with pytest.raises(LlmUnavailableError) as ei:
        health_check()
    assert "could not reach" in ei.value.reason


@respx.mock
def test_health_check_fails_on_5xx() -> None:
    respx.get(_tags_url()).mock(return_value=httpx.Response(500, text="oh no"))
    with pytest.raises(LlmUnavailableError):
        health_check()


# ==================== chat ====================


@respx.mock
def test_chat_round_trip_returns_assistant_content() -> None:
    expected = '{"answer_language": "en", "claims": []}'
    route = respx.post(_chat_url()).mock(
        return_value=httpx.Response(
            200, json={"message": {"role": "assistant", "content": expected}}
        )
    )

    import json as _json  # noqa: PLC0415

    result = chat(system="sys", user="user", json_schema={"type": "object"})

    assert result == expected
    # Verify the wire-format details we depend on. Parse JSON rather than
    # substring-match so we don't break on whitespace/encoding variants.
    sent = route.calls.last.request
    body = _json.loads(sent.read().decode("utf-8"))
    assert body["stream"] is False
    assert "model" in body
    assert "format" in body  # json_schema field was included
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]


@respx.mock
def test_chat_omits_format_when_no_schema() -> None:
    route = respx.post(_chat_url()).mock(
        return_value=httpx.Response(200, json={"message": {"content": "ok"}})
    )

    import json as _json  # noqa: PLC0415

    chat(system="s", user="u")

    body = _json.loads(route.calls.last.request.read().decode("utf-8"))
    assert "format" not in body


@respx.mock
def test_chat_wraps_transport_error() -> None:
    respx.post(_chat_url()).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LlmUnavailableError) as ei:
        chat(system="s", user="u")
    assert "chat request failed" in ei.value.reason


@respx.mock
def test_chat_rejects_malformed_response_shape() -> None:
    """A response missing the expected 'message.content' shape must fail
    loudly rather than silently return an empty string."""
    respx.post(_chat_url()).mock(
        return_value=httpx.Response(200, json={"message": {"content": 12345}})
    )
    with pytest.raises(LlmUnavailableError) as ei:
        chat(system="s", user="u")
    assert "unexpected response shape" in ei.value.reason
