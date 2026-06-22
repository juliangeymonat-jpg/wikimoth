"""WikiMoth — deterministic, token-minimal, reproducible memory for Claude/agents.

Pipeline: MOTHRAG ``GraphRetriever(source="wikilinks")`` pulls the relevant
note-chain from a ``[[wikilink]]`` markdown vault → a compaction stage →
a Claude reader. The public surface is the :class:`MemoryRAG` pipeline plus its
pluggable stages (compactor, reader).

Importing this package is free of network/API and of the MOTHRAG dependency:
MOTHRAG is imported lazily, only when a vault is indexed with the default
retriever (or when :class:`ClaudeReader` is constructed).
"""

from wikimoth.compaction import Compactor, HeadroomCompactor, NoOpCompactor
from wikimoth.pipeline import MemoryRAG
from wikimoth.reader import ClaudeReader, EchoReader, Reader
from wikimoth.tokens import count_passage_tokens, count_tokens, token_backend

__version__ = "0.1.0"

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
