"""Thin Ollama client.

Single-purpose: talk to a local Ollama server via its HTTP API. Health
check at startup so we fail fast if the server is down or the configured
model isn't pulled. Chat completions with optional JSON-schema enforcement.

We talk to ``/api/chat`` (not ``/api/generate``) so the conversation
contract — system message + user message — is explicit in the wire
format. ``stream: false`` because we collect the full response before
parsing JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from common.errors import LlmUnavailableError
from common.logging import logger
from common.settings import get_settings

if TYPE_CHECKING:  # pragma: no cover
    pass

# Ollama's chat / completion endpoints can be slow on CPU. A 5-min upper
# bound keeps test/dev runs honest while leaving room for legitimately
# long generations.
_REQUEST_TIMEOUT_SECONDS = 300.0


def _base_url() -> str:
    s = get_settings()
    return f"http://{s.ollama_host}:{s.ollama_port}"


def health_check() -> None:
    """Verify Ollama is up AND the configured model is available.

    Raises:
        LlmUnavailableError: if the server is unreachable, returns an
        error, or the configured model is not in the model list.
    """
    settings = get_settings()
    url = f"{_base_url()}/api/tags"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        raise LlmUnavailableError(
            host=settings.ollama_host,
            model=settings.ollama_model,
            reason=f"could not reach {url}: {e}",
        ) from e

    models = {m.get("name") for m in payload.get("models", [])}
    if settings.ollama_model not in models:
        raise LlmUnavailableError(
            host=settings.ollama_host,
            model=settings.ollama_model,
            reason=(
                f"model {settings.ollama_model!r} not found on Ollama; "
                f"pull it with: ollama pull {settings.ollama_model}. "
                f"Available: {sorted(models)!r}"
            ),
        )


def chat(
    *,
    system: str,
    user: str,
    json_schema: dict[str, Any] | None = None,
    temperature: float = 0.1,
) -> str:
    """Send a single-turn chat to Ollama and return the assistant message.

    Args:
        system: system prompt content.
        user: user message content.
        json_schema: if given, passed as Ollama's ``format`` field to force
            the model to emit JSON conforming to that schema. Ollama
            validates this server-side.
        temperature: low default (0.1) — we want deterministic, faithful
            answers, not creative writing.
    """
    settings = get_settings()
    url = f"{_base_url()}/api/chat"
    body: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            # Penalize token repetition (default in Ollama is 1.1). The 1.3
            # value here discourages the LLM from looping on the same
            # claim template under strict JSON mode — observed on small
            # models that "lock onto" one chunk and repeat it.
            "repeat_penalty": 1.3,
        },
    }
    if json_schema is not None:
        body["format"] = json_schema

    logger.info(
        f"ollama chat: model={settings.ollama_model} "
        f"system={len(system)}c user={len(user)}c json={json_schema is not None}"
    )
    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        raise LlmUnavailableError(
            host=settings.ollama_host,
            model=settings.ollama_model,
            reason=f"chat request failed: {e}",
        ) from e

    # /api/chat returns {"message": {"role": "assistant", "content": "..."}, ...}
    msg = payload.get("message", {})
    content = msg.get("content", "")
    if not isinstance(content, str):
        raise LlmUnavailableError(
            host=settings.ollama_host,
            model=settings.ollama_model,
            reason=f"unexpected response shape: {payload!r}",
        )
    return content
