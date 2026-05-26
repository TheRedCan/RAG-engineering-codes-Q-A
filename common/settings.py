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

    # embed stage. BGE-M3's dense output is always 1024-dim. Batch size on
    # CPU trades memory for throughput; 16-32 is a good range on 16 GB RAM.
    embed_dense_dim: int = Field(default=1024, ge=1, le=4096)
    embed_batch_size: int = Field(default=16, ge=1, le=256)
    # tokens, BGE-M3 caps at 8192. Default 1024 is well above our typical
    # ~480-token chunk while keeping attention activations small enough for
    # CPU runs on a 16 GB machine (attention memory is O(seq²)).
    embed_max_length: int = Field(default=1024, ge=128, le=8192)

    # retrieval tuning. retrieve_top_k=50 is wide enough to include
    # sparse-only-dominant chunks (e.g. equation listings whose tokens
    # don't match the dense embedding of a natural-language query but
    # which sparse retrieval catches). rerank_top_k=8 gives the LLM
    # room to draw both reference text AND substantive content without
    # blowing the context window.
    retrieve_top_k: int = Field(default=50, ge=1, le=200)
    rerank_top_k: int = Field(default=8, ge=1, le=50)
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
