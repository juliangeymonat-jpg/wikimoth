"""WikiMoth — deterministic, token-minimal, reproducible memory for Claude/agents.

Pipeline: a vendored ``GraphRetriever(source="wikilinks")`` pulls the relevant
note-chain from a ``[[wikilink]]`` markdown vault → a compaction stage →
a Claude reader. The public surface is the :class:`MemoryRAG` pipeline plus its
pluggable stages (compactor, reader).

Importing this package is free of network/API and has ZERO runtime
dependencies: the wikilink graph retriever, chunker and pipeline are all
vendored under :mod:`wikimoth.retrieval` (Apache-2.0, adapting the MOTHRAG
paper's chunking convention). There is no ``mothrag`` package dependency.
"""

from wikimoth.compaction import Compactor, HeadroomCompactor, NoOpCompactor
from wikimoth.pipeline import MemoryRAG
from wikimoth.reader import ClaudeReader, EchoReader, Reader
from wikimoth.tokens import count_passage_tokens, count_tokens, token_backend

__version__ = "0.2.1"

__all__ = [
    "MemoryRAG",
    "Compactor",
    "NoOpCompactor",
    "HeadroomCompactor",
    "Reader",
    "EchoReader",
    "ClaudeReader",
    "count_tokens",
    "count_passage_tokens",
    "token_backend",
    "__version__",
]
