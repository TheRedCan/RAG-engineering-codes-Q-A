"""Shared BGE-M3 loader.

The embedder is used by two stages with slightly different needs:
- ingest.embed encodes corpus chunks in batches at indexing time
- retrieval.hybrid encodes a single user query at query time

Both go through the same singleton so loading the ~2 GB model is paid
exactly once per process. The actual import of FlagEmbedding (which
pulls in torch) is deferred to first ``get()`` so tests that mock the
encoder don't pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.logging import logger
from common.settings import get_settings

if TYPE_CHECKING:  # pragma: no cover
    pass


class BgeM3:
    """Singleton wrapper around FlagEmbedding.BGEM3FlagModel.

    Usage:
        dense_vecs, sparse_dicts = BgeM3.get().encode(["text"], max_length=512)
    """

    _instance: BgeM3 | None = None

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(f"loading embedding model {settings.embed_model!r} on {settings.embed_device}")
        # Lazy import keeps tests cheap; torch is heavy.
        from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

        # use_fp16 only makes sense on CUDA; CPU stays FP32.
        use_fp16 = settings.embed_device.startswith("cuda")
        self._model: Any = BGEM3FlagModel(
            settings.embed_model,
            use_fp16=use_fp16,
            devices=[settings.embed_device],
        )

    @classmethod
    def get(cls) -> BgeM3:
        """Return the process-wide singleton, loading it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Clear the singleton so tests can inject a mock instance."""
        cls._instance = None

    def encode(
        self,
        texts: list[str],
        *,
        max_length: int,
        with_sparse: bool = True,
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """Encode texts.

        Returns:
            (dense_vecs, sparse_dicts) — both lists are aligned with ``texts``.
            Sparse dicts map token-id (int) -> learned weight (float).
            When ``with_sparse=False``, the sparse list contains empty dicts.
        """
        out = self._model.encode(
            sentences=texts,
            batch_size=len(texts),
            max_length=max_length,
            return_dense=True,
            return_sparse=with_sparse,
            return_colbert_vecs=False,
        )
        dense = [list(map(float, v)) for v in out["dense_vecs"]]
        if with_sparse:
            sparse = [{int(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]
        else:
            sparse = [{} for _ in texts]
        return dense, sparse


__all__ = ["BgeM3"]
