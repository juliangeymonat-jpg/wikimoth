# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Dogfood: WikiMoth retrieval over Claude's OWN memory vault. Zero API cost.

Indexes the real Claude memory folder (an ``MEMORY.md`` index plus topic notes
that cross-reference each other with ``[[wikilinks]]``) via
:class:`wikimoth.MemoryRAG`, then runs a few *connect-the-dots* questions and
prints, for each:

    * the retrieved chunk names (note ``#chunk``),
    * their hop distance (0 = lexical seed; >=1 = reached via ``[[wikilink]]``),
    * the per-chunk and per-question token count fed to the reader.

Notes are chunked into ~400-token pieces (the chunking convention), so retrieval
returns the relevant *chunks*, not whole files: a single fat note (the
~30k-token ``MEMORY.md`` table of contents) no longer floods the reader. The
pure-navigation ``MEMORY.md`` hub is indexed as **graph edges only** (its
``[[links]]`` build edges; its content is excluded from candidates).

**Retrieval only**: the reader is never called, so there is **no API call and
no cost**. ``GraphRetriever(source="wikilinks")`` (vendored in
``wikimoth.retrieval``) is pure-Python (no embeddings, no network), so this is
fast even over the whole vault.

Run (WikiMoth is self-contained, just put it on the path):

    PYTHONPATH=. python examples/dogfood_memory.py [VAULT_DIR]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from wikimoth import MemoryRAG
from wikimoth.tokens import count_tokens, token_backend

# Point this at any folder of [[wikilink]] markdown notes (an Obsidian vault, or
# Claude Code's own memory folder): pass it as the 1st arg, or set WIKIMOTH_VAULT.
_ENV_VAULT = os.environ.get("WIKIMOTH_VAULT")
DEFAULT_VAULT = Path(_ENV_VAULT) if _ENV_VAULT else None

# Seed from only the few strongest lexical matches, then let the wikilink graph
# expand: this leaves slots for hop>=1 notes so the connect-the-dots property
# is visible (vs seed_top_k == top_k, which fills every slot with raw lexical
# hits and never exercises the graph).
SEED_TOP_K = 3
MAX_HOPS = 2


def _build_retriever():
    from wikimoth.retrieval import GraphRetriever

    return GraphRetriever(
        source="wikilinks", max_hops=MAX_HOPS, seed_top_k=SEED_TOP_K
    )

# Generic connect-the-dots questions; replace with ones that fit your own vault.
QUESTIONS = [
    "what connects this project to its main dependency?",
    "what architectural decision was made, and why?",
    "which rule governs how the code repos are organised?",
]


def chunk_name(rag: MemoryRAG, chunk) -> str:
    """Human-readable ``note #chunkN`` label for a retrieved chunk."""
    # GraphRetriever exposes note_identity(); fall back to the filename.
    fn = getattr(rag.retriever, "note_identity", None)
    slug = fn(chunk) if callable(fn) else ""
    if not slug:
        meta = getattr(chunk, "metadata", None) or {}
        slug = meta.get("filename") or getattr(chunk, "doc_id", "?")
    cid = getattr(chunk, "chunk_id", "") or ""
    suffix = ""
    if "#" in cid:
        suffix = " #" + cid.split("#", 1)[1]
    return f"{slug}{suffix}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    vault = Path(args[0]) if args else DEFAULT_VAULT
    if vault is None:
        print("[dogfood] pass a folder of [[wikilink]] markdown as the 1st arg (or set WIKIMOTH_VAULT).")
        return
    if not vault.exists():
        print(f"[dogfood] vault not found: {vault}")
        print("[dogfood] pass a folder of [[wikilink]] markdown as the 1st arg.")
        return

    # Seed-limited GraphRetriever so the wikilink hops are actually exercised.
    rag = MemoryRAG(retriever=_build_retriever())   # NoOp / Echo defaults
    rag.index(vault)            # retrieval only; reader never invoked

    print(f"[dogfood] vault          : {vault}")
    print(f"[dogfood] chunks indexed : {len(rag)}  (~400-token chunks, 50 overlap)")
    excl = getattr(rag, "exclude_content", ())
    if excl:
        print(f"[dogfood] edges-only hubs: {', '.join(excl)} (content excluded)")
    edge_count = getattr(rag.retriever, "edge_count", None)
    if callable(edge_count):
        print(f"[dogfood] wikilink edges : {edge_count()}")
    print(f"[dogfood] token backend  : {token_backend()}")
    print()

    top_k = 8
    for qi, question in enumerate(QUESTIONS, 1):
        results = rag.retrieve_with_hops(question, top_k=top_k)
        _, total_tokens = rag.retrieve(question, top_k=top_k)

        print(f"=== Q{qi}: {question}")
        print(f"    tokens fed to reader (top_k={top_k}): {total_tokens}")
        print("    retrieved chunks (rank. note #chunk  [hop, tokens]):")
        for rank, (c, hop) in enumerate(results, 1):
            name = chunk_name(rag, c)
            tag = "seed" if hop == 0 else f"{hop}-hop link"
            n_tok = count_tokens(getattr(c, "text", "") or "")
            print(f"      {rank:>2}. {name:<66} [hop={hop} {tag:<11} tokens={n_tok}]")
        print()


if __name__ == "__main__":
    main()
