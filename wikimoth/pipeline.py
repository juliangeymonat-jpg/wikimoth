# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MemoryRAG — the WikiMoth pipeline: retrieve → compact → read.

Wires three pluggable stages over a folder of ``[[wikilink]]`` markdown notes
(an Obsidian-style vault, or Claude's own memory folder):

1. **retrieve** — MOTHRAG's :class:`GraphRetriever(source="wikilinks")` walks
   the human-authored wikilink graph to pull the relevant *note-chain* for a
   question (multi-hop, no embeddings, no LLM, no network).
2. **compact** — an optional :class:`~wikimoth.compaction.Compactor` shrinks the
   retrieved passages before the (paid) reader. Default is the identity
   :class:`NoOpCompactor`.
3. **read** — a :class:`~wikimoth.reader.Reader` answers from the passages.
   Default is the API-free :class:`EchoReader`, so the whole pipeline runs end
   to end with zero cost; swap in :class:`ClaudeReader` for a real Claude
   answer.

Every stage is constructor-injectable, so the pipeline is both testable
(synthetic vault + Echo reader, no API) and pluggable (drop in any retriever /
compactor / reader satisfying the Protocols).

Token discipline (the value prop)
---------------------------------
:meth:`MemoryRAG.index` splits each note into ~400-token chunks with ~50-token
overlap (MOTHRAG's own ``_simple_chunk`` convention) instead of loading a whole
note as one blob. Retrieval then returns the relevant *chunks*, not whole
files, so a single fat note (e.g. a 30k-token ``MEMORY.md`` table-of-contents)
no longer floods the reader. Each chunk keeps its source-note identity
(``doc_id`` = note slug, ``chunk_id`` = ``f"{slug}#chunk{i}"``), so the
wikilink graph still connects across chunked notes — the multi-hop property is
preserved at chunk granularity.

Pure-navigation hubs (``index_notes`` / ``exclude_content``)
------------------------------------------------------------
A note like ``MEMORY.md`` is a *table of contents*: it lexically matches every
query and links to everything, so as retrievable content it would dominate the
candidate set while carrying almost no answer-bearing text. ``index`` can treat
such notes as **graph edges only**: their ``[[links]]`` still build edges (so
they connect the dots), but their own chunks are excluded from the candidate
set. ``exclude_content`` defaults to ``("MEMORY.md",)``.

Self-contained: ``GraphRetriever``, ``Chunk`` and ``simple_chunk`` are vendored
in :mod:`wikimoth.retrieval` (zero ``mothrag`` dependency).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from wikimoth.compaction import Compactor, NoOpCompactor
from wikimoth.reader import EchoReader, Reader
from wikimoth.tokens import count_passage_tokens, token_backend


def _parse_iso_date(s: Any) -> date | None:
    """Parse a full ISO date string (``YYYY-MM-DD``); fail-open to ``None``."""
    if not isinstance(s, str):
        return None
    s = s.strip()[:10]
    if len(s) != 10:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None

# MOTHRAG's ingest convention (reused, not reinvented): notes are split into
# ~CHUNK_SIZE_TOKENS-token chunks with CHUNK_OVERLAP_TOKENS overlap.
# See mothrag.core.api._DEFAULTS ("chunk_size_tokens": 400,
# "chunk_overlap_tokens": 50) and mothrag.core.api.ingest (api.py:762).
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# Notes whose own chunks are excluded from retrieval by default (their wikilinks
# still build graph edges). MEMORY.md is the canonical Claude-memory hub: a pure
# table-of-contents that lexically matches every query.
DEFAULT_EXCLUDE_CONTENT: tuple[str, ...] = ("MEMORY.md",)


def _slugify_note(name: str) -> str:
    """Note-identity slug (delegates to the vendored ``GraphRetriever._slugify``)."""
    from wikimoth.retrieval.graph import _slugify

    return _slugify(name)


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into ~400-token chunks (50 overlap) via MOTHRAG's chunker.

    Reuses ``mothrag.core.api._simple_chunk`` (sentence-aware, whitespace-token
    sizing) so WikiMoth's chunking is identical to MOTHRAG's ingest path. Falls
    back to a small deterministic word-window splitter only if that symbol is
    unavailable, so the pipeline never silently degrades to whole-note chunks.
    """
    from wikimoth.retrieval.chunking import simple_chunk

    return simple_chunk(
        text,
        chunk_size_tokens=CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
    )


def _fallback_word_chunk(
    text: str,
    *,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Deterministic word-window splitter (only used if the lab chunker moves)."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size_tokens:
        return [" ".join(words)]
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    out: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_tokens]
        if window:
            out.append(" ".join(window))
        if start + chunk_size_tokens >= len(words):
            break
    return out


def _make_chunk(
    *,
    text: str,
    note_slug: str,
    idx: int,
    filename: str,
    source: str,
    superseded: bool = False,
    valid_from: str | None = None,
    valid_to: str | None = None,
) -> Any:
    """Build a MOTHRAG ``Chunk`` with chunk-level, note-preserving identity.

    ``doc_id`` is the note slug and ``chunk_id`` is ``f"{slug}#chunk{idx}"`` —
    exactly the shape ``GraphRetriever._note_identity`` strips back to a note,
    so every chunk of a note shares one graph node-identity and wikilinks still
    resolve at chunk granularity. The bitemporal/supersession fields are stamped
    on EVERY chunk of a note (not just chunk 0) so the as-of filter is correct for
    multi-chunk notes.
    """
    from wikimoth.retrieval.chunk import Chunk

    return Chunk(
        text=text,
        doc_id=note_slug,
        chunk_id=f"{note_slug}#chunk{idx}",
        metadata={
            "source": source,
            "filename": filename,
            "note_slug": note_slug,
            "superseded": superseded,
            "valid_from": valid_from,
            "valid_to": valid_to,
        },
    )


def _load_vault_chunks(
    vault_dir: str | Path,
    *,
    exclude_content: Iterable[str] = DEFAULT_EXCLUDE_CONTENT,
) -> list[Any]:
    """Load ``.md`` notes under ``vault_dir`` as ~400-token MOTHRAG ``Chunk`` s.

    Each note is split into ~``CHUNK_SIZE_TOKENS``-token chunks (overlap
    ``CHUNK_OVERLAP_TOKENS``) via MOTHRAG's ``_simple_chunk``, so a fat note no
    longer enters the reader whole. Every chunk keeps its source-note identity
    (``doc_id`` = note slug, ``chunk_id`` = ``f"{slug}#chunk{i}"``) so the
    wikilink graph still connects across chunks of the same note.

    Notes whose *filename* matches ``exclude_content`` (default ``MEMORY.md``)
    are kept in the graph as **edges only**: their chunks stay in the graph
    (so their ``[[links]]`` build edges and they can serve as BFS waypoints)
    but are flagged ``metadata['exclude_content']`` so the pipeline drops them
    from the retrieval candidate set (see :class:`MemoryRAG`). The ``Chunk``
    import is local so importing this module needs nothing from MOTHRAG until a
    vault is actually indexed.
    """
    vault = Path(vault_dir)
    if not vault.exists():
        raise FileNotFoundError(f"vault_dir not found: {vault}")

    from wikimoth.frontmatter import parse_frontmatter

    exclude_slugs = {_slugify_note(n) for n in exclude_content}

    chunks: list[Any] = []
    for p in sorted(vault.rglob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        note_slug = _slugify_note(p.name)
        source = str(p)

        excluded = note_slug in exclude_slugs
        fm = parse_frontmatter(text)
        superseded = bool(fm.get("superseded_by")) or fm.get("status") == "superseded"
        valid_from = fm.get("valid_from")
        valid_to = fm.get("valid_to")

        parts = _chunk_text(text)
        if not parts:
            # Empty / whitespace-only note: keep a single empty chunk so its
            # identity still exists as a (possibly link-target) graph node.
            parts = [text]
        for i, ctext in enumerate(parts):
            c = _make_chunk(
                text=ctext,
                note_slug=note_slug,
                idx=i,
                filename=p.name,
                source=source,
                superseded=superseded,
                valid_from=valid_from,
                valid_to=valid_to,
            )
            if excluded:
                # Edges-only hub: its chunks stay in the GRAPH (full text, so
                # its [[links]] build edges AND it can be a BFS waypoint), but
                # are flagged so the pipeline drops them from the returned
                # candidate set — its (e.g. 30k-token TOC) body never reaches
                # the reader, while still connecting the dots.
                c.metadata["exclude_content"] = True
            chunks.append(c)
    return chunks


def _default_retriever() -> Any:
    """Construct the default ``GraphRetriever(source='wikilinks')``.

    Imported lazily so ``import wikimoth`` and ``MemoryRAG(retriever=...)`` work
    with an injected retriever even when MOTHRAG is not on the path.
    """
    from wikimoth.retrieval.graph import GraphRetriever

    return GraphRetriever(source="wikilinks", max_hops=2)


class MemoryRAG:
    """Deterministic, token-minimal memory over a ``[[wikilink]]`` vault.

    Parameters
    ----------
    retriever
        Anything satisfying MOTHRAG's ``Retriever`` Protocol
        (``index`` / ``retrieve`` / ``__len__``). Default:
        ``GraphRetriever(source="wikilinks", max_hops=2)`` (built lazily).
    compactor
        A :class:`~wikimoth.compaction.Compactor`. Default:
        :class:`NoOpCompactor` (identity).
    reader
        A :class:`~wikimoth.reader.Reader`. Default: :class:`EchoReader`
        (no API, deterministic) — so ``answer()`` is free until you inject a
        real reader.
    exclude_content
        Filenames whose chunks are **graph-edges-only**: their ``[[links]]``
        build edges, but their own chunks are dropped from the retrieval
        candidate set. Default :data:`DEFAULT_EXCLUDE_CONTENT` (``MEMORY.md``).
        Pass ``()`` to keep every note's content retrievable. Used by
        :meth:`index` (the on-disk path); :meth:`index_chunks` honours a
        ``metadata['exclude_content']`` flag on pre-built chunks instead.
    """

    def __init__(
        self,
        *,
        retriever: Any | None = None,
        compactor: Compactor | None = None,
        reader: Reader | None = None,
        exclude_content: Iterable[str] = DEFAULT_EXCLUDE_CONTENT,
        as_of: date | None = None,
        show_superseded: bool = False,
    ) -> None:
        self.retriever = retriever if retriever is not None else _default_retriever()
        self.compactor = compactor if compactor is not None else NoOpCompactor()
        self.reader = reader if reader is not None else EchoReader()
        self.exclude_content = tuple(exclude_content)
        self._exclude_slugs = {_slugify_note(n) for n in self.exclude_content}
        self._indexed = False
        # Supersession-aware / bitemporal retrieval (mutable per request):
        #   as_of=None           -> current view: drop superseded + expired bodies
        #   as_of=<date D>       -> time-travel: show what was valid at D
        #   show_superseded=True -> forensic: keep everything
        self.as_of = as_of
        self.show_superseded = show_superseded
        self._n_droppable = 0

    # ------------------------------------------------------------------
    # Excluded-content bookkeeping
    # ------------------------------------------------------------------

    def _is_excluded(self, chunk: Any) -> bool:
        """True if ``chunk`` belongs to an edges-only (excluded) note."""
        meta = getattr(chunk, "metadata", None) or {}
        if meta.get("exclude_content"):
            return True
        slug = meta.get("note_slug") or _slugify_note(
            getattr(chunk, "doc_id", "") or ""
        )
        return slug in self._exclude_slugs

    def _is_visible(self, chunk: Any) -> bool:
        """Supersession-/as-of-aware visibility of a chunk's body in the result.

        The chunk stays in the GRAPH either way (so its ``[[NEW]]`` edge is live and
        a query hitting a superseded note hops one edge to the current one); this
        only controls whether its BODY reaches the reader. Fail-open: an unparseable
        date never hides a chunk.
        """
        if self.show_superseded:
            return True
        meta = getattr(chunk, "metadata", None) or {}
        vf = _parse_iso_date(meta.get("valid_from"))
        vt = _parse_iso_date(meta.get("valid_to"))
        # The reference instant: today for the current view, or the as-of date for
        # time-travel. A fact is valid at T iff valid_from <= T < valid_to.
        t = self.as_of if self.as_of is not None else date.today()
        valid_at_t = (vf is None or vf <= t) and (vt is None or t < vt)
        if self.as_of is not None:
            # time-travel: only the validity window matters (the supersession flag
            # is a "now" concept; at T the replacement may not have happened yet).
            return valid_at_t
        # current view: valid right now AND not superseded.
        return valid_at_t and not meta.get("superseded")

    def _keep(self, chunk: Any) -> bool:
        """A chunk's body reaches the reader iff it is neither excluded nor hidden."""
        return not self._is_excluded(chunk) and self._is_visible(chunk)

    def _count_droppable(self, chunks: Sequence[Any]) -> int:
        """Over-fetch headroom: how many chunks the keep-filter drops in the current
        view. Counts only chunks actually hidden NOW (excluded, superseded, or
        outside today's validity window) rather than every dated note, so a vault
        of currently-valid dated notes does not inflate every query's fetch."""
        today = date.today()
        n = 0
        for c in chunks:
            meta = getattr(c, "metadata", None) or {}
            if self._is_excluded(c) or meta.get("superseded"):
                n += 1
                continue
            vf = _parse_iso_date(meta.get("valid_from"))
            vt = _parse_iso_date(meta.get("valid_to"))
            valid_now = (vf is None or vf <= today) and (vt is None or today < vt)
            if not valid_now:
                n += 1
        return n

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, vault_dir: str | Path) -> "MemoryRAG":
        """Chunk ``.md`` notes under ``vault_dir`` and build the wikilink graph.

        Each note is split into ~400-token chunks (50 overlap) via MOTHRAG's
        chunker, keeping per-chunk note identity so the graph still connects
        across chunks. Notes in ``self.exclude_content`` become edges-only.
        Returns ``self`` for chaining. Pure-python and network-free for the
        default ``GraphRetriever``.
        """
        chunks = _load_vault_chunks(vault_dir, exclude_content=self.exclude_content)
        self.retriever.index(chunks)
        self._n_droppable = self._count_droppable(chunks)
        self._indexed = True
        return self

    def index_chunks(self, chunks: Sequence[Any]) -> "MemoryRAG":
        """Index pre-built chunks directly (used by tests / in-memory vaults).

        Excluded-content handling for this path is driven by the per-chunk
        ``metadata['exclude_content']`` flag (and ``self.exclude_content`` by
        note slug); the chunks are passed to the retriever verbatim so its
        graph still sees the excluded notes' edges.
        """
        chunks = list(chunks)
        self.retriever.index(chunks)
        self._n_droppable = self._count_droppable(chunks)
        self._indexed = True
        return self

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = 8) -> tuple[list[Any], int]:
        """Return ``(chunks, token_count)`` for ``question``.

        Excluded-content (edges-only) chunks are dropped from the result, so the
        returned chunks are answer-bearing content only. To keep ``top_k``
        content chunks despite the drop, retrieval over-fetches before
        filtering. ``token_count`` is the number of tokens the retained
        passages would feed to the reader (the quantity WikiMoth minimises).
        """
        chunks = self._retrieve_filtered(question, top_k=top_k)
        token_count = count_passage_tokens(getattr(c, "text", "") or "" for c in chunks)
        return chunks, token_count

    def retrieve_with_hops(
        self, question: str, top_k: int = 8
    ) -> list[tuple[Any, int]]:
        """``[(chunk, hop_distance), ...]`` with excluded-content dropped.

        Falls back to hop ``-1`` (unknown) for retrievers without
        ``retrieve_with_hops`` (e.g. a plain dense retriever).
        """
        fn = getattr(self.retriever, "retrieve_with_hops", None)
        if callable(fn):
            fetch = self._fetch_k(top_k)
            pairs = fn(question, top_k=fetch)
            kept = [(c, hop) for c, hop in pairs if self._keep(c)]
            return kept[:top_k]
        chunks = self._retrieve_filtered(question, top_k=top_k)
        return [(c, -1) for c in chunks]

    def _fetch_k(self, top_k: int) -> int:
        """Over-fetch budget so dropping hidden/excluded chunks still yields ``top_k``.

        Headroom = the number of chunks the keep-filter could drop (excluded hubs +
        superseded + dated). With a plain vault this is ``top_k``; an as-of /
        supersession filter that drops many chunks no longer starves the result.
        """
        return top_k + self._n_droppable

    def _retrieve_filtered(self, question: str, *, top_k: int) -> list[Any]:
        """Retrieve, drop excluded/hidden chunks, trim to ``top_k``."""
        fetch = self._fetch_k(top_k)
        chunks = self.retriever.retrieve(question, top_k=fetch)
        return [c for c in chunks if self._keep(c)][:top_k]

    # ------------------------------------------------------------------
    # Answer (retrieve → compact → read)
    # ------------------------------------------------------------------

    def answer(self, question: str, top_k: int = 8) -> str:
        """Full pipeline: retrieve → compact → read. Returns the answer text.

        With the defaults (Graph retriever + NoOp compactor + Echo reader) this
        runs end to end with **zero** API calls.
        """
        result = self.answer_verbose(question, top_k=top_k)
        return result["answer"]

    def answer_verbose(self, question: str, top_k: int = 8) -> dict[str, Any]:
        """Like :meth:`answer` but returns provenance + token accounting.

        Keys: ``answer``, ``passages`` (post-compaction), ``chunks`` (raw
        retrieved, excluded-content already dropped), ``tokens_retrieved``,
        ``tokens_fed_to_reader``, ``token_backend``, ``n_passages``.
        """
        chunks = self._retrieve_filtered(question, top_k=top_k)
        raw_passages = [getattr(c, "text", "") or "" for c in chunks]
        tokens_retrieved = count_passage_tokens(raw_passages)

        passages = self.compactor.compact(raw_passages)
        tokens_fed = count_passage_tokens(passages)

        answer = self.reader.read(question, passages)
        return {
            "answer": answer,
            "passages": passages,
            "chunks": chunks,
            "tokens_retrieved": tokens_retrieved,
            "tokens_fed_to_reader": tokens_fed,
            "token_backend": token_backend(),
            "n_passages": len(passages),
        }

    def __len__(self) -> int:
        return len(self.retriever)


__all__ = ["MemoryRAG", "CHUNK_SIZE_TOKENS", "CHUNK_OVERLAP_TOKENS",
           "DEFAULT_EXCLUDE_CONTENT"]
