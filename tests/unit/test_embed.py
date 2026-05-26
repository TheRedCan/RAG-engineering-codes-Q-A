"""Unit tests for ingest.embed.

We mock both BGE-M3 and the Qdrant client so tests stay fast and offline.
Integration against real Qdrant + real BGE-M3 happens in the actual run,
which would be too slow for CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.errors import CollectionSchemaMismatchError, QdrantUnavailableError
from common.manifest import manifest_path
from common.models import Chunk, Language
from ingest import embed as embed_mod
from ingest.embed import (
    _DENSE_NAME,
    _SPARSE_NAME,
    _batched,
    _ensure_collection,
    _existing_chunk_ids,
    _point_id_for,
    embed_all,
)

# ==================== pure helpers ====================


def test_point_id_for_is_deterministic() -> None:
    assert _point_id_for("abc#00000") == _point_id_for("abc#00000")
    assert _point_id_for("abc#00000") != _point_id_for("abc#00001")


def test_batched_yields_expected_groups() -> None:
    items = list(range(10))
    batches = list(_batched(items, 3))  # type: ignore[arg-type]
    assert [len(b) for b in batches] == [3, 3, 3, 1]


# ==================== Qdrant schema helpers ====================


def _mock_collections(names: list[str]) -> MagicMock:
    """Build a mock get_collections() return: namespace with .collections list.

    ``name`` is a reserved kwarg on MagicMock (sets the mock's identifier),
    so we must assign .name after construction for the test code to read it
    as a real string attribute.
    """
    cols = []
    for n in names:
        c = MagicMock()
        c.name = n
        cols.append(c)
    return MagicMock(collections=cols)


def _mock_get_collection(dense_dim: int, *, with_sparse: bool = True) -> MagicMock:
    """Mock client.get_collection(name) -> CollectionInfo with the named-vector layout."""
    dense_cfg = MagicMock(size=dense_dim)
    vectors = {_DENSE_NAME: dense_cfg}
    sparse_vectors: dict[str, Any] = {_SPARSE_NAME: MagicMock()} if with_sparse else {}
    return MagicMock(
        config=MagicMock(
            params=MagicMock(vectors=vectors, sparse_vectors=sparse_vectors),
        )
    )


def test_ensure_collection_creates_when_missing() -> None:
    client = MagicMock()
    client.get_collections.return_value = _mock_collections([])

    _ensure_collection(client, "ec_chunks", dense_dim=1024)

    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "ec_chunks"
    assert _DENSE_NAME in kwargs["vectors_config"]
    assert kwargs["vectors_config"][_DENSE_NAME].size == 1024
    assert _SPARSE_NAME in kwargs["sparse_vectors_config"]


def test_ensure_collection_accepts_matching_schema() -> None:
    client = MagicMock()
    client.get_collections.return_value = _mock_collections(["ec_chunks"])
    client.get_collection.return_value = _mock_get_collection(dense_dim=1024)

    _ensure_collection(client, "ec_chunks", dense_dim=1024)

    client.create_collection.assert_not_called()


def test_ensure_collection_rejects_wrong_dense_dim() -> None:
    client = MagicMock()
    client.get_collections.return_value = _mock_collections(["ec_chunks"])
    client.get_collection.return_value = _mock_get_collection(dense_dim=384)

    with pytest.raises(CollectionSchemaMismatchError) as ei:
        _ensure_collection(client, "ec_chunks", dense_dim=1024)
    assert "dense dim mismatch" in ei.value.detail


def test_ensure_collection_rejects_missing_sparse() -> None:
    client = MagicMock()
    client.get_collections.return_value = _mock_collections(["ec_chunks"])
    client.get_collection.return_value = _mock_get_collection(dense_dim=1024, with_sparse=False)

    with pytest.raises(CollectionSchemaMismatchError) as ei:
        _ensure_collection(client, "ec_chunks", dense_dim=1024)
    assert "sparse" in ei.value.detail


def test_existing_chunk_ids_collects_across_pages() -> None:
    """The scroll loop must iterate until offset is None and de-dup chunk_ids."""
    client = MagicMock()
    page1 = ([MagicMock(payload={"chunk_id": f"d#{i:05d}"}) for i in range(3)], "next-offset")
    page2 = ([MagicMock(payload={"chunk_id": f"d#{i:05d}"}) for i in range(3, 5)], None)
    client.scroll.side_effect = [page1, page2]

    found = _existing_chunk_ids(client, "ec_chunks", "d")

    assert found == {f"d#{i:05d}" for i in range(5)}
    assert client.scroll.call_count == 2


# ==================== embed_all end-to-end ====================


def _write_manifest_entry(raw_dir: Path, doc_id: str) -> None:
    entry = {
        "doc_id": doc_id,
        "family": "fema",
        "code_number": "T",
        "edition_year": 2024,
        "variant": None,
        "title_en": "t",
        "title_ar": None,
        "source_url": "https://example.test/x.pdf",
        "expected_language": "en",
        "sha256": None,
        "license_note": "test",
    }
    manifest_path(raw_dir).write_text(json.dumps([entry]), encoding="utf-8")


def _write_chunks(processed_dir: Path, doc_id: str, n: int) -> None:
    out = processed_dir / "chunks" / f"{doc_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(n):
            c = Chunk(
                chunk_id=f"{doc_id}#{i:05d}",
                doc_id=doc_id,
                page_numbers=[i + 1],
                section_path=None,
                text=f"chunk {i} text body",
                language=Language.EN,
                char_count=20,
            )
            f.write(c.model_dump_json() + "\n")


def _fake_encode_batch(texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Deterministic fake encoder: dense vec of fixed length, sparse dict per chunk."""
    dim = 1024
    return (
        [[float(i % 7) / 7.0] * dim for i, _ in enumerate(texts)],
        [{1: 0.5, 2: 0.3} for _ in texts],
    )


def test_embed_all_hard_fails_when_qdrant_unreachable(raw_dir: Path) -> None:
    _write_manifest_entry(raw_dir, "x")

    # _connect_qdrant catches any exception from the constructor OR the
    # handshake call and wraps it as QdrantUnavailableError. We trigger
    # via the constructor here.
    with (
        patch("qdrant_client.QdrantClient", side_effect=ConnectionRefusedError("nope")),
        pytest.raises(QdrantUnavailableError),
    ):
        embed_all()


def test_embed_all_indexes_chunks_and_skips_already_present(
    raw_dir: Path, processed_dir: Path
) -> None:
    _write_manifest_entry(raw_dir, "d1")
    _write_chunks(processed_dir, "d1", n=5)

    client = MagicMock()
    client.get_collections.return_value = _mock_collections(["engineering_codes_chunks"])
    client.get_collection.return_value = _mock_get_collection(dense_dim=1024)
    # Pretend chunks d1#00000 and d1#00001 are already indexed.
    client.scroll.return_value = (
        [
            MagicMock(payload={"chunk_id": "d1#00000"}),
            MagicMock(payload={"chunk_id": "d1#00001"}),
        ],
        None,
    )

    with (
        patch("qdrant_client.QdrantClient", return_value=client),
        patch.object(embed_mod, "_encode_batch", side_effect=_fake_encode_batch),
    ):
        result = embed_all()

    assert result.all_succeeded
    assert result.chunks_indexed == 3  # 5 total - 2 already present
    assert result.chunks_skipped == 2
    # One upsert call (3 chunks fit in default batch size of 16).
    assert client.upsert.call_count == 1
    upsert_kwargs = client.upsert.call_args.kwargs
    assert len(upsert_kwargs["points"]) == 3


def test_embed_all_soft_fails_on_missing_chunks_file(raw_dir: Path) -> None:
    _write_manifest_entry(raw_dir, "ghost")

    client = MagicMock()
    client.get_collections.return_value = _mock_collections(["engineering_codes_chunks"])
    client.get_collection.return_value = _mock_get_collection(dense_dim=1024)

    with patch("qdrant_client.QdrantClient", return_value=client):
        result = embed_all()

    assert not result.all_succeeded
    assert result.failed[0][0] == "ghost"
    assert "chunks JSONL not found" in result.failed[0][1]


def test_embed_all_collects_partial_failure(raw_dir: Path, processed_dir: Path) -> None:
    """If one batch upsert raises, the doc is reported as partial — not silently dropped."""
    _write_manifest_entry(raw_dir, "d2")
    _write_chunks(processed_dir, "d2", n=2)  # one batch

    client = MagicMock()
    client.get_collections.return_value = _mock_collections(["engineering_codes_chunks"])
    client.get_collection.return_value = _mock_get_collection(dense_dim=1024)
    client.scroll.return_value = ([], None)
    client.upsert.side_effect = RuntimeError("simulated qdrant outage")

    with (
        patch("qdrant_client.QdrantClient", return_value=client),
        patch.object(embed_mod, "_encode_batch", side_effect=_fake_encode_batch),
    ):
        result = embed_all()

    assert not result.all_succeeded
    assert result.chunks_indexed == 0
    assert "1 batches failed" in result.failed[0][1]
