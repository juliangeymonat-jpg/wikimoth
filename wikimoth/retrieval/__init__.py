# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""WikiMoth retrieval core — self-contained, zero external dependency.

Vendored from MOTHRAG (Julian's own Apache-2.0 code) so WikiMoth installs and runs
with **no ``mothrag`` dependency**:

- :class:`Chunk` — the retrievable unit (note-identity + text + metadata).
- :func:`simple_chunk` — sentence-aware ~400-token chunker.
- :func:`_slugify` — note-identity normaliser (``[[Bar Baz]]`` ↔ ``Bar_Baz.md``).
- :class:`GraphRetriever` — wikilink-graph multi-hop retrieval (pure stdlib).

The default WikiMoth retriever is :class:`GraphRetriever`; the BM25-seeded
:class:`wikimoth.hybrid.HybridRetriever` builds on it.
"""

from wikimoth.retrieval.chunk import Chunk
from wikimoth.retrieval.chunking import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    simple_chunk,
)
from wikimoth.retrieval.graph import GraphRetriever, _slugify

__all__ = [
    "Chunk",
    "simple_chunk",
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "GraphRetriever",
    "_slugify",
]
