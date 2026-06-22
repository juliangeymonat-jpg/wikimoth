# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Real flat baselines for the benchmark — BM25 (sparse) and dense (semantic).

The clean/realistic corpora compare graph traversal against retrieval that has
no notion of links. To make that comparison credible (not "GraphRetriever with
traversal disabled"), this module ships two *standard* flat retrievers that
satisfy MOTHRAG's ``Retriever`` Protocol (``index`` / ``retrieve`` / ``__len__``)
so they drop straight into :class:`wikimoth.MemoryRAG` / the harness:

- :class:`BM25Retriever` — Okapi BM25 (``rank_bm25``). The canonical sparse
  baseline. Like any lexical method it cannot reach a note that shares no terms
  with the query.
- :class:`STDenseRetriever` — sentence-transformers bi-encoder cosine
  (``all-MiniLM-L6-v2`` by default). The honest *semantic* adversary: it can
  reach a note that is topically related without sharing words. $0 inference,
  no API key (local model); import-guarded so the package needs neither dep
  until a baseline is actually constructed.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

# Set before any torch import: avoids a known OpenMP duplicate-runtime crash on
# Windows (the same conflict that segfaults the `sentence_transformers` package
# on this stack — we use transformers+torch directly to sidestep it).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


class BM25Retriever:
    """Okapi BM25 sparse retriever (``rank_bm25``). Real flat sparse baseline."""

    name = "bm25"

    def __init__(self) -> None:
        self._chunks: list[Any] = []
        self._bm25 = None

    def index(self, chunks: Sequence[Any]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover
            raise ImportError("BM25Retriever requires `rank_bm25` (pip install rank_bm25).") from e
        self._chunks = list(chunks)
        corpus = [_tokenize(getattr(c, "text", "") or "") for c in self._chunks]
        # BM25Okapi needs a non-empty doc per row; substitute a sentinel token.
        corpus = [toks or ["\x00empty"] for toks in corpus]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, question: str, *, top_k: int = 10) -> list[Any]:
        if not self._chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        out: list[Any] = []
        for i in order:
            c = self._chunks[i]
            try:
                c.score = float(scores[i])
            except (AttributeError, TypeError):
                pass
            out.append(c)
        return out

    def __len__(self) -> int:
        return len(self._chunks)


class STDenseRetriever:
    """Dense semantic retriever — MiniLM bi-encoder cosine, via transformers.

    The honest *semantic* baseline: it can reach a note topically related to the
    query without sharing words. Built directly on ``transformers`` + ``torch``
    (mean pooling + L2 normalize — exactly what sentence-transformers does for
    this model) to sidestep a local segfault in the ``sentence_transformers``
    package. Local model → $0 inference, no API key. Model + tokenizer are
    loaded lazily in ``__init__`` (first use downloads ~80MB from the HF hub).
    """

    name = "dense_st"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 *, batch_size: int = 64, max_length: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "STDenseRetriever requires `transformers` + `torch`."
            ) from e
        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._chunks: list[Any] = []
        self._emb = None

    def _encode(self, texts: list[str]):
        import numpy as np
        torch = self._torch
        out_chunks = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self._tok(batch, padding=True, truncation=True,
                            max_length=self.max_length, return_tensors="pt")
            with torch.no_grad():
                model_out = self._model(**enc)
            tok_emb = model_out.last_hidden_state            # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (tok_emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = summed / counts                            # mean pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out_chunks.append(emb.cpu().numpy())
        return np.vstack(out_chunks) if out_chunks else np.zeros((0, 1))

    def index(self, chunks: Sequence[Any]) -> None:
        self._chunks = list(chunks)
        texts = [getattr(c, "text", "") or "" for c in self._chunks]
        self._emb = self._encode(texts)

    def retrieve(self, question: str, *, top_k: int = 10) -> list[Any]:
        if not self._chunks or self._emb is None:
            return []
        q = self._encode([question])[0]
        scores = self._emb @ q  # cosine (vectors are unit-normalized)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:top_k]
        out: list[Any] = []
        for i in order:
            c = self._chunks[i]
            try:
                c.score = float(scores[i])
            except (AttributeError, TypeError):
                pass
            out.append(c)
        return out

    def __len__(self) -> int:
        return len(self._chunks)


__all__ = ["BM25Retriever", "STDenseRetriever"]
