# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""HybridRetriever — BM25-seeded graph traversal (WikiMoth v2).

The real-vault dogfood showed the honest trade-off: on direct-lookup queries a
strong flat retriever (BM25/dense) beats graph-pure retrieval (the answer note
shares words with the query), while graph traversal is the only thing that
reaches a *connected* answer note that shares no words (the multi-hop case). The
hybrid takes both:

1. **Seed** with BM25 (a proper TF-IDF-style ranker, far stronger than the
   graph's crude set-overlap seeding) — so a direct-lookup answer note is the
   top seed and is returned.
2. **Traverse** the ``[[wikilink]]`` graph from a SMALL number of those seeds
   (``seed_top_k``, default 1 = the single best match) — so a link-only answer
   note is reached via BFS. Seeding narrowly is what keeps sibling/lexical
   decoys from flooding the reserve with the wrong chain.
3. **Blend, BM25-primary** (``graph_reserve``): return BM25's top
   ``(top_k - graph_reserve)`` notes first — NEVER displace a strong direct hit
   — then fill the last ``graph_reserve`` slots with graph-traversal-only notes
   (the link-only multi-hop reach BM25 can't see). This keeps real-vault parity
   with BM25 (it never loses a direct hit) while adding multi-hop. ``graph_reserve``
   is the knob: 0 = pure BM25, higher = more multi-hop reach (at some
   direct-lookup cost on dense vaults — measured Pareto trade-off).

Satisfies the ``Retriever`` Protocol (``index`` / ``retrieve`` / ``__len__``) so
it drops into :class:`wikimoth.MemoryRAG`. Pure-Python + ``rank_bm25``; the
wikilink adjacency is reused from WikiMoth's vendored
:class:`wikimoth.retrieval.GraphRetriever`.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Sequence

_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


class HybridRetriever:
    """BM25-seeded wikilink-graph traversal. Best of flat + multi-hop."""

    name = "hybrid"

    def __init__(
        self,
        *,
        source: str = "wikilinks",
        max_hops: int = 2,
        seed_top_k: int = 1,
        undirected: bool = True,
        max_hub_degree: int | None = None,
        graph_reserve: int = 2,
    ) -> None:
        self.source = source
        self.max_hops = int(max_hops)
        self.seed_top_k = int(seed_top_k)
        self.undirected = bool(undirected)
        self.max_hub_degree = max_hub_degree
        # BM25-PRIMARY blend: return BM25's top (top_k - graph_reserve) notes
        # (never lose a strong direct hit), then fill the last `graph_reserve`
        # slots with graph-traversal-only notes (the link-only multi-hop reach
        # BM25 can't see). graph_reserve=0 → pure BM25; =top_k → graph-only.
        self.graph_reserve = int(graph_reserve)
        self._chunks: list[Any] = []
        self._bm25 = None
        self._graph = None

    def index(self, chunks: Sequence[Any]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover
            raise ImportError("HybridRetriever requires `rank_bm25`.") from e
        from wikimoth.retrieval import GraphRetriever

        self._chunks = list(chunks)
        corpus = [_tokenize(getattr(c, "text", "") or "") for c in self._chunks]
        corpus = [toks or ["\x00empty"] for toks in corpus]
        self._bm25 = BM25Okapi(corpus)
        # Reuse the wikilink adjacency (we only read ._adj; not modified).
        self._graph = GraphRetriever(
            source=self.source, max_hops=self.max_hops, undirected=self.undirected,
            max_hub_degree=self.max_hub_degree,
        )
        self._graph.index(self._chunks)

    def retrieve(self, question: str, *, top_k: int = 10) -> list[Any]:
        if not self._chunks or self._bm25 is None or self._graph is None:
            return []
        from wikimoth.retrieval import _slugify

        scores = self._bm25.get_scores(_tokenize(question))
        mx = max(scores) if len(scores) else 0.0
        norm = [(s / mx) if mx > 0 else 0.0 for s in scores]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

        seeds = [i for i in order if scores[i] > 0][: self.seed_top_k] or order[:1]

        # BFS over the wikilink adjacency from the BM25 seeds, carrying the
        # seed's bm25 strength decayed by hop (provenance) to rank graph-only
        # notes — the link-only multi-hop reach BM25 alone can't see.
        adj = self._graph._adj
        hop: dict[int, int] = {i: 0 for i in seeds}
        val: dict[int, float] = {i: norm[i] for i in seeds}
        frontier: deque[int] = deque(seeds)
        while frontier:
            i = frontier.popleft()
            d = hop[i]
            if d >= self.max_hops:
                continue
            for j in adj.get(i, ()):
                if j not in hop or hop[j] > d + 1:
                    hop[j] = d + 1
                    val[j] = val.get(i, 0.0) * 0.5
                    frontier.append(j)
                elif hop[j] == d + 1:
                    val[j] = max(val.get(j, 0.0), val.get(i, 0.0) * 0.5)

        def slug_of(i: int) -> str:
            return _slugify(getattr(self._chunks[i], "doc_id", "") or "")

        reserve = max(0, min(self.graph_reserve, top_k))
        n_primary = top_k - reserve

        picked: list[Any] = []
        seen: set[str] = set()

        def add(i: int, score: float) -> None:
            s = slug_of(i)
            if not s or s in seen:
                return
            seen.add(s)
            c = self._chunks[i]
            try:
                c.score = float(score)
            except (AttributeError, TypeError):
                pass
            picked.append(c)

        # 1) BM25-primary: top distinct notes (never displace a strong direct hit).
        for i in order:
            if len(picked) >= n_primary:
                break
            if scores[i] <= 0 and picked:
                break  # don't pad the primary block with zero-score noise
            add(i, norm[i] + 1.0)

        # 2) reserve slots: graph-traversal-only notes (not already picked),
        #    ranked by hop then provenance value.
        graph_only = [i for i in hop if slug_of(i) not in seen]
        graph_only.sort(key=lambda i: (hop[i], -val.get(i, 0.0), i))
        for i in graph_only:
            if len(picked) >= top_k:
                break
            add(i, val.get(i, 0.0) + 1.0 / (2 ** hop[i]))

        # 3) backfill from the rest of the BM25 order if still short.
        for i in order:
            if len(picked) >= top_k:
                break
            add(i, norm[i])

        return picked[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


__all__ = ["HybridRetriever"]
