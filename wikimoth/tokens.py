# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Token counting — the unit WikiMoth optimises.

WikiMoth's whole value proposition is *tokens fed to the reader*, so we need a
single, consistent counter shared by the pipeline and the benchmark harness.

Preference order:

1. ``tiktoken`` (``cl100k_base``) if installed — the most representative count
   for a real LLM reader. Cached encoder so repeated calls are cheap.
2. otherwise a dependency-free estimate of ``len(text) // 4`` (the common
   ~4-chars-per-token rule of thumb).

The exact backend used is reported via :func:`token_backend` so a benchmark
run records *how* it counted. Counts must be comparable *within* a run (same
backend for every arm), which they are by construction.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoder():
    """Return a cached tiktoken encoder, or ``None`` if tiktoken is absent."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 — any failure → estimate fallback
        return None


def token_backend() -> str:
    """Name of the active counting backend (``"tiktoken"`` or ``"estimate"``)."""
    return "tiktoken" if _encoder() is not None else "estimate"


def count_tokens(text: str) -> int:
    """Token count for a single string."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return len(text) // _CHARS_PER_TOKEN


def count_passage_tokens(passages: Iterable[str]) -> int:
    """Total token count across an iterable of passage strings."""
    return sum(count_tokens(p) for p in passages)


__all__ = ["count_tokens", "count_passage_tokens", "token_backend"]
