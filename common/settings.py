"""Centralised, validated configuration.

Reads from environment variables (and a local ``.env``). Pydantic enforces
types and ranges at startup — invalid config fails immediately instead of
silently misbehaving deep in a pipeline stage.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, IPvAnyAddress
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Project-wide settings. Singleton accessed via ``get_settings()``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # logging
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR)$")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # storage
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")

    # qdrant
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_collection: str = "engineering_codes_chunks"

    # ollama
    ollama_host: str = "127.0.0.1"
    ollama_port: int = Field(default=11434, ge=1, le=65535)
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"

    # embeddings / reranker
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_device: str = Field(default="cpu", pattern=r"^(cpu|cuda|cuda:\d+)$")
    rerank_device: str = Field(default="cpu", pattern=r"^(cpu|cuda|cuda:\d+)$")

    # chunking tuning. Character-based for v0.1 (no tokenizer dep);
    # ~4 chars/token in English so 2048 ≈ 512 tokens, a common RAG sweet spot.
    chunk_target_chars: int = Field(default=2048, ge=100, le=10000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=2000)
    chunk_min_size_chars: int = Field(default=100, ge=1)

    # retrieval tuning
    retrieve_top_k: int = Field(default=30, ge=1, le=200)
    rerank_top_k: int = Field(default=6, ge=1, le=50)
    multihop_max_hops: int = Field(default=3, ge=1, le=6)

    # app exposure (kept on localhost by default — see SECURITY.md)
    app_host: IPvAnyAddress = Field(default="127.0.0.1")  # type: ignore[assignment]
    app_port: int = Field(default=8501, ge=1, le=65535)


_cached: Settings | None = None


def get_settings() -> Settings:
    """Return the validated settings singleton. Loads on first call."""
    global _cached  # noqa: PLW0603 — module-level singleton is the intended pattern here
    if _cached is None:
        _cached = Settings()
    return _cached
