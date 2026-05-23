"""Ingest stage: fetch -> parse -> chunk -> embed.

Each submodule is independently runnable via its CLI entry point and writes
JSONL output that the next submodule consumes. See common.models for the
schemas.
"""
