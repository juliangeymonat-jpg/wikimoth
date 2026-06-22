# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""GraphRetriever -- substrate-agnostic graph retrieval over a folder of notes
whose edges are *human-authored* (no embeddings, no LLM, no network).

Vendored into WikiMoth from MOTHRAG (``mothrag.core.retrieval.graph``) so WikiMoth is
self-contained and pip-installable with zero ``mothrag`` dependency. Treats a
corpus of markdown notes as a graph where

- **nodes** are passages (chunks), and
- **edges** are ``[[wikilink]]`` references between notes (Obsidian-style / the
  Claude memory vault). Edges are authored by a human, so the graph needs *zero*
  embeddings and *zero* external dependencies.

This gives the multi-hop property a lexical-only (or even a dense) retriever
misses: a passage that is **not** lexically similar to the question but **is**
reachable via a ``[[wikilink]]`` from a lexically-matching passage still gets
retrieved. Pure Python: ``re`` + ``collections``.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Sequence


# A ``[[wikilink]]`` token. Handles the three Obsidian forms:
#   [[Target]]            -> Target
#   [[Target|alias text]] -> Target   (display alias dropped)
#   [[Target#heading]]    -> Target   (heading anchor dropped)
_WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:[#\|][^\]]*)?\]\]")

# Tokeniser shared by note-identity slugify and lexical seeding.
_WORD_RE = re.compile(r"\w+")


def _slugify(name: str) -> str:
    """Normalise a note title / link target to a comparable identity key.

    Lowercase, collapse internal whitespace and ``_``/``-`` runs to single
    spaces, drop a trailing ``.md``/``.markdown`` extension, and strip any
    leading directory components. ``"Foo/Bar Baz.md"`` and ``"bar-baz"`` both
    normalise to ``"bar baz"`` so a ``[[Bar Baz]]`` link resolves to the
    ``Bar_Baz.md`` note regardless of surface punctuation.
    """
    if not name:
        return ""
    s = str(name).strip()
    s = s.replace("\\", "/")
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    low = s.lower()
    for ext in (".markdown", ".md"):
        if low.endswith(ext):
            s = s[: -len(ext)]
            break
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(text: str) -> list[str]:
    """Lowercase ``\\w+`` tokeniser (dep-free)."""
    return _WORD_RE.findall(text.lower())


class GraphRetriever:
    """Wikilink-graph retriever satisfying the ``Retriever`` Protocol (duck-typed).

    Parameters
    ----------
    source
        Edge source. Only ``"wikilinks"`` is implemented; the constructor rejects
        anything else so a typo fails loudly rather than producing an edge-free graph.
    max_hops
        Maximum BFS depth from the lexical seeds along wikilink edges (default 2).
    seed_top_k
        How many lexical seeds to expand from. ``None`` (default) means "use the
        ``top_k`` passed to :meth:`retrieve`". Seed narrowly (small value) under
        heavy lexical noise so decoys don't crowd the BFS-reached chain.
    undirected
        Traverse wikilink edges in both directions when ``True`` (default).
    source_opts
        Reserved for future edge sources; ignored by ``"wikilinks"``.
    """

    name = "graph_wikilinks"

    def __init__(
        self,
        *,
        source: str = "wikilinks",
        max_hops: int = 2,
        seed_top_k: int | None = None,
        undirected: bool = True,
        max_hub_degree: int | None = None,
        rank_mode: str = "flat",
        **source_opts,
    ) -> None:
        if source != "wikilinks":
            raise ValueError(
                f"GraphRetriever source must be 'wikilinks' (only source "
                f"implemented), got {source!r}."
            )
        if rank_mode not in ("flat", "provenance"):
            raise ValueError(f"rank_mode must be 'flat' or 'provenance', got {rank_mode!r}.")
        self.source = source
        self.max_hops = int(max_hops)
        self.seed_top_k = seed_top_k
        self.undirected = bool(undirected)
        # Hub taming: a note linking to MORE than this many distinct notes (an
        # index/TOC like MEMORY.md) is demoted from the graph — its edges are
        # dropped so BFS doesn't flood through it. None = keep all edges.
        self.max_hub_degree = max_hub_degree
        # Ranking: "flat" = lexical + proximity (every reached node competes
        # equally); "provenance" = origin-seed lexical × proximity (a node
        # reached from a STRONG seed outranks one reached incidentally / via a
        # hub) — fixes the STAGE-4 flood where a genuinely-linked answer is
        # outscored by hundreds of incidentally-reached chunks.
        self.rank_mode = rank_mode
        self.source_opts = dict(source_opts)

        self._chunks: list = []
        self._note_to_chunks: dict[str, set[int]] = {}
        self._adj: dict[int, set[int]] = {}
        self._hub_slugs: set[str] = set()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence) -> None:
        """Build the wikilink graph over ``chunks`` (idempotent rebuild)."""
        self._chunks = list(chunks)
        self._note_to_chunks = {}
        self._adj = {i: set() for i in range(len(self._chunks))}

        for i, c in enumerate(self._chunks):
            slug = self._note_identity(c)
            if slug:
                self._note_to_chunks.setdefault(slug, set()).add(i)

        # Hub detection: a note whose distinct *resolved* link targets exceed
        # max_hub_degree is an index/TOC (e.g. MEMORY.md → ~every note). Its
        # edges flood BFS, so we demote it: drop edges FROM and INTO it.
        self._hub_slugs = set()
        if self.max_hub_degree is not None:
            targets_per_note: dict[str, set[str]] = {}
            for i, c in enumerate(self._chunks):
                src = self._note_identity(c)
                if not src:
                    continue
                for raw in _WIKILINK_RE.findall(getattr(c, "text", "") or ""):
                    t = _slugify(raw)
                    if t and t != src and t in self._note_to_chunks:
                        targets_per_note.setdefault(src, set()).add(t)
            self._hub_slugs = {
                s for s, ts in targets_per_note.items() if len(ts) > self.max_hub_degree
            }

        for i, c in enumerate(self._chunks):
            if self._note_identity(c) in self._hub_slugs:
                continue  # hub: no traversable edges out of it
            text = getattr(c, "text", "") or ""
            for raw_target in _WIKILINK_RE.findall(text):
                target = _slugify(raw_target)
                if not target or target in self._hub_slugs:
                    continue  # dangling, or don't link INTO a demoted hub
                targets = self._note_to_chunks.get(target)
                if not targets:
                    continue  # dangling link: target note not in corpus.
                for j in targets:
                    if j == i:
                        continue
                    self._adj[i].add(j)
                    if self.undirected:
                        self._adj[j].add(i)

    @staticmethod
    def _note_identity(chunk) -> str:
        """Derive a chunk's note slug from its identity fields."""
        meta = getattr(chunk, "metadata", None) or {}
        candidates = [
            meta.get("title"),
            meta.get("filename"),
            meta.get("source"),
            getattr(chunk, "doc_id", "") or "",
        ]
        cid = getattr(chunk, "chunk_id", "") or ""
        if cid:
            candidates.append(cid.split("#", 1)[0])
        for cand in candidates:
            slug = _slugify(cand or "")
            if slug:
                return slug
        return ""

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str, *, top_k: int = 10) -> list:
        """Return up to ``top_k`` chunks: lexical seeds + wikilink expansion."""
        return [c for c, _hop in self._retrieve_ranked(question, top_k=top_k)]

    def retrieve_with_hops(self, question: str, *, top_k: int = 10) -> list:
        """Like :meth:`retrieve` but returns ``[(chunk, hop_distance), ...]``."""
        return self._retrieve_ranked(question, top_k=top_k)

    def _retrieve_ranked(self, question: str, *, top_k: int) -> list:
        """Shared core: returns ``[(chunk, hop_distance), ...]`` ranked."""
        if not self._chunks:
            return []

        q_tokens = set(_tokenize(question))

        lex: dict[int, float] = {}
        for i, c in enumerate(self._chunks):
            s = self._lexical_score(q_tokens, getattr(c, "text", "") or "")
            if s > 0.0:
                lex[i] = s

        n_seeds = self.seed_top_k if self.seed_top_k is not None else top_k
        n_seeds = max(1, int(n_seeds))
        seeds = sorted(lex, key=lambda i: lex[i], reverse=True)[:n_seeds]

        hop: dict[int, int] = {i: 0 for i in seeds}
        # Provenance value: the origin seed's lexical score carried along the
        # BFS tree, decayed 0.5 per hop (seeds keep their own lexical). Used by
        # rank_mode="provenance" so a node reached from a STRONG seed outranks
        # one reached incidentally / via a hub.
        value: dict[int, float] = {i: lex.get(i, 0.0) for i in seeds}
        frontier: deque[int] = deque(seeds)
        while frontier:
            i = frontier.popleft()
            d = hop[i]
            if d >= self.max_hops:
                continue
            for j in self._adj.get(i, ()):  # neighbours
                if j not in hop or hop[j] > d + 1:
                    hop[j] = d + 1
                    value[j] = value.get(i, 0.0) * 0.5
                    frontier.append(j)
                elif hop[j] == d + 1:
                    value[j] = max(value.get(j, 0.0), value.get(i, 0.0) * 0.5)

        scored: list[tuple[float, int]] = []
        for i in hop:
            if self.rank_mode == "provenance":
                score = value.get(i, 0.0)
            else:
                score = lex.get(i, 0.0) + self._proximity_bonus(hop[i])
            scored.append((score, i))

        scored.sort(key=lambda t: (-t[0], t[1]))

        out: list = []
        for score, i in scored[:top_k]:
            c = self._chunks[i]
            try:
                c.score = float(score)
            except (AttributeError, TypeError):
                pass
            out.append((c, hop[i]))
        return out

    @staticmethod
    def _lexical_score(q_tokens: set, text: str) -> float:
        """Dep-free token-overlap score in [0, 1] (fraction of query tokens present)."""
        if not q_tokens:
            return 0.0
        doc_tokens = set(_tokenize(text))
        if not doc_tokens:
            return 0.0
        overlap = q_tokens & doc_tokens
        return len(overlap) / len(q_tokens)

    @staticmethod
    def _proximity_bonus(hop_distance: int) -> float:
        """Traversal proximity bonus, decaying with hop distance (hop0=1, hop1=.5, ...)."""
        h = max(0, int(hop_distance))
        return 1.0 / (2 ** h)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def note_identity(self, chunk) -> str:
        """Public alias of :meth:`_note_identity` for demos / debugging."""
        return self._note_identity(chunk)

    def edge_count(self) -> int:
        """Total number of directed adjacency edges in the graph."""
        return sum(len(v) for v in self._adj.values())


__all__ = ["GraphRetriever", "_slugify"]
