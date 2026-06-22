# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""simple_chunk — sentence-aware fixed-size chunker (vendored from MOTHRAG).

Byte-for-byte the same algorithm as ``mothrag.core.api._simple_chunk`` so WikiMoth
chunks a note identically to MOTHRAG's ingest, vendored so WikiMoth needs no
``mothrag`` install. Pure stdlib (``re``).
"""

from __future__ import annotations

import re

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50


def simple_chunk(
    text: str,
    *,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Sentence-aware fixed-size chunker.

    Splits on sentence boundaries (.!?). Greedily packs sentences into chunks of
    ~``chunk_size_tokens`` whitespace tokens with ``chunk_overlap_tokens`` overlap
    between adjacent chunks. Tokens = whitespace-separated runs (approximate, not
    a GPT tokenizer; good enough for chunking heuristics).
    """
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        s_tokens = len(sent.split())
        if cur_tokens + s_tokens > chunk_size_tokens and cur:
            chunks.append(" ".join(cur))
            overlap: list[str] = []
            o_tokens = 0
            for back in reversed(cur):
                b_tokens = len(back.split())
                if o_tokens + b_tokens > chunk_overlap_tokens:
                    break
                overlap.insert(0, back)
                o_tokens += b_tokens
            cur = overlap
            cur_tokens = o_tokens
        cur.append(sent)
        cur_tokens += s_tokens
    if cur:
        chunks.append(" ".join(cur))
    return chunks


__all__ = ["simple_chunk", "CHUNK_SIZE_TOKENS", "CHUNK_OVERLAP_TOKENS"]
