"""Retrieval stage: hybrid (dense + sparse) search, reranking, and the
multi-hop reference-following loop.

Not yet implemented. The stage will read ``Chunk`` records from the embed
stage's Qdrant index and emit ``list[RetrievedChunk]`` for the generator.
"""
