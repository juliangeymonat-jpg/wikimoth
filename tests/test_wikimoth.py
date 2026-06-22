# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""WikiMoth tests — synthetic in-memory vault, NO paid API calls.

WikiMoth is self-contained; just put it on the path (provides GraphRetriever + Chunk):

    PYTHONPATH=. pytest -q

Every test uses the API-free EchoReader / NoOp + Headroom-fallback compactors,
so nothing here touches the network, Claude, or any paid service.
"""

from __future__ import annotations

import pytest

from wikimoth.retrieval import Chunk  # noqa: E402

from wikimoth import (  # noqa: E402
    EchoReader,
    HeadroomCompactor,
    MemoryRAG,
    NoOpCompactor,
    count_tokens,
)
from wikimoth.benchmark.harness import (  # noqa: E402
    FourArmHarness,
    Question,
    oracle_retrieval_loss,
)
from wikimoth.pipeline import (  # noqa: E402
    CHUNK_SIZE_TOKENS,
    _chunk_text,
    _load_vault_chunks,
    _slugify_note,
)


# ---------------------------------------------------------------------------
# Synthetic in-memory vault
# ---------------------------------------------------------------------------
#
# The multi-hop trap: the question asks about "the capital of the kingdom of
# Eldoria". The note that NAMES the capital ("Brightport") never repeats the
# query tokens "capital"/"Eldoria" — it is reachable ONLY by following a
# [[wikilink]] from the lexically-matching Eldoria note. A lexical retriever
# alone would miss it; the wikilink graph connects the dots.

VAULT = {
    "Eldoria.md": (
        "Eldoria kingdom capital. The seat of Eldoria's crown lives in "
        "[[Brightport]]. See also [[Geography]]."
    ),
    # Deliberately shares ZERO tokens with the QUESTION so it can ONLY be
    # reached by following the [[Brightport]] wikilink from the Eldoria note.
    "Brightport.md": (
        "Brightport, a bustling harbour founded in third age, hosts royal mint."
    ),
    "Geography.md": (
        "Northern reaches stay mountainous, cold, fjorded."
    ),
    "Cooking.md": (
        "Recipe fish stew: simmer broth, add herbs, serve hot."
    ),
}

QUESTION = "Where is Eldoria's capital and crown seat?"
GOLD_NOTE = "Brightport.md"  # the answer-bearing note (reached via wikilink)

# Only the single strongest lexical match (Eldoria) seeds; Brightport (zero
# query-token overlap) must therefore arrive purely via the wikilink hop.
SEED_TOP_K = 1


def _graph_retriever(max_hops: int = 2):
    from wikimoth.retrieval import GraphRetriever

    return GraphRetriever(
        source="wikilinks", max_hops=max_hops, seed_top_k=SEED_TOP_K
    )


def _make_chunks(vault: dict[str, str]) -> list[Chunk]:
    """One Chunk per synthetic note; filename carries note identity."""
    chunks = []
    for fname, text in vault.items():
        chunks.append(
            Chunk(
                text=text,
                doc_id=fname,
                chunk_id=f"{fname}#chunk0",
                metadata={"source": fname, "filename": fname},
            )
        )
    return chunks


@pytest.fixture
def chunks():
    return _make_chunks(VAULT)


@pytest.fixture
def rag(chunks):
    r = MemoryRAG(
        retriever=_graph_retriever(),
        compactor=NoOpCompactor(),
        reader=EchoReader(),
    )
    r.index_chunks(chunks)
    return r


# ---------------------------------------------------------------------------
# Retrieval: the multi-hop property
# ---------------------------------------------------------------------------

def test_multihop_reaches_lexically_dissimilar_note(rag):
    """Brightport (no query tokens) surfaces via [[wikilink]] from Eldoria."""
    chunks, tokens = rag.retrieve(QUESTION, top_k=8)
    names = {c.doc_id for c in chunks}
    assert "Eldoria.md" in names, "lexical seed must be retrieved"
    assert GOLD_NOTE in names, (
        "answer-bearing note must be reached via the wikilink graph, "
        f"got {names}"
    )
    assert tokens > 0


def test_retrieve_returns_token_count(rag):
    chunks, tokens = rag.retrieve(QUESTION, top_k=8)
    assert isinstance(tokens, int)
    assert tokens > 0


def test_lexical_only_baseline_misses_the_hop(chunks):
    """max_hops=0 (lexical only) should NOT reach Brightport — proves the hop matters."""
    lex_only = _graph_retriever(max_hops=0)
    lex_only.index(chunks)
    names = {c.doc_id for c in lex_only.retrieve(QUESTION, top_k=8)}
    assert "Eldoria.md" in names
    assert GOLD_NOTE not in names, (
        "lexical-only retrieval should miss the linked note; if it doesn't, "
        "the multi-hop test is not actually exercising the graph"
    )


def test_retrieve_with_hops_reports_distance(rag):
    pairs = rag.retrieve_with_hops(QUESTION, top_k=8)
    by_doc = {c.doc_id: hop for c, hop in pairs}
    assert by_doc.get("Eldoria.md") == 0, "lexical seed is hop 0"
    assert by_doc.get(GOLD_NOTE, -1) >= 1, "Brightport is reached >=1 hop away"


# ---------------------------------------------------------------------------
# Compactors
# ---------------------------------------------------------------------------

def test_noop_compactor_is_identity():
    passages = ["alpha", "beta", "gamma"]
    out = NoOpCompactor().compact(passages)
    assert out == passages
    assert out is not passages  # returns a fresh list


def test_headroom_compactor_falls_back_to_noop_when_absent():
    """headroom is not installed in this env → graceful NoOp fallback."""
    c = HeadroomCompactor()
    assert c.available is False, "headroom should be absent in the test env"
    passages = ["one", "two"]
    assert c.compact(passages) == passages  # identity fallback


def test_headroom_strict_raises_when_absent():
    with pytest.raises(ImportError):
        HeadroomCompactor(strict=True)


# ---------------------------------------------------------------------------
# Reader (API-free)
# ---------------------------------------------------------------------------

def test_echo_reader_runs_without_api():
    out = EchoReader().read("q?", ["passage one", "passage two"])
    assert "passages=2" in out
    assert "q?" in out


def test_echo_reader_empty_passages():
    out = EchoReader().read("q?", [])
    assert "no passages" in out


# ---------------------------------------------------------------------------
# End-to-end pipeline (no API)
# ---------------------------------------------------------------------------

def test_answer_runs_end_to_end_with_echo(rag):
    ans = rag.answer(QUESTION, top_k=8)
    assert isinstance(ans, str)
    assert ans  # non-empty
    assert "[echo]" in ans  # proves the EchoReader (not a real API) answered


def test_answer_verbose_token_accounting(rag):
    v = rag.answer_verbose(QUESTION, top_k=8)
    assert v["tokens_fed_to_reader"] > 0
    assert v["tokens_retrieved"] == v["tokens_fed_to_reader"]  # NoOp → equal
    assert v["n_passages"] >= 2
    assert v["token_backend"] in ("tiktoken", "estimate")


def test_injectable_stages_default(chunks):
    """Defaults must be GraphRetriever / NoOp / Echo and run free."""
    r = MemoryRAG()  # all defaults
    r.index_chunks(chunks)
    assert r.compactor.name == "noop"
    assert r.reader.name == "echo"
    assert len(r) == len(VAULT)


# ---------------------------------------------------------------------------
# Benchmark harness (no API)
# ---------------------------------------------------------------------------

def test_harness_dump_vs_deterministic_token_savings(tmp_path):
    """The deterministic arm must feed FEWER tokens than the dump arm."""
    for fname, text in VAULT.items():
        (tmp_path / fname).write_text(text, encoding="utf-8")
    # When loaded from disk, doc_id is the full path; build gold from that.
    gold_paths = [str(tmp_path / GOLD_NOTE)]

    h = FourArmHarness(tmp_path, retriever=_graph_retriever(), top_k=8)
    q = Question(text=QUESTION, gold_doc_ids=gold_paths)

    dump = h.run_arm("dump", q)
    det = h.run_arm("deterministic", q)

    assert dump.tokens_fed_to_reader >= det.tokens_fed_to_reader
    assert dump.retrieval_recall_at_k == 1.0  # dump trivially contains gold
    # deterministic must actually retrieve the gold note (multi-hop) → recall 1
    assert det.retrieval_recall_at_k == 1.0


def test_harness_agentic_arm_is_stub(tmp_path):
    for fname, text in VAULT.items():
        (tmp_path / fname).write_text(text, encoding="utf-8")
    h = FourArmHarness(tmp_path)
    with pytest.raises(NotImplementedError):
        h.run_arm("agentic", Question(text=QUESTION))


def test_harness_compacted_arm_notes_headroom_fallback(tmp_path):
    for fname, text in VAULT.items():
        (tmp_path / fname).write_text(text, encoding="utf-8")
    h = FourArmHarness(tmp_path)
    rec = h.run_arm("deterministic_compacted", Question(text=QUESTION))
    assert "fallback" in rec.note.lower()  # headroom absent in test env


def test_harness_run_skips_agentic_by_default(tmp_path):
    for fname, text in VAULT.items():
        (tmp_path / fname).write_text(text, encoding="utf-8")
    h = FourArmHarness(tmp_path)
    recs = h.run([Question(text=QUESTION)],
                 arms=("dump", "agentic", "deterministic"))
    arms_run = {r.arm for r in recs}
    assert "dump" in arms_run and "deterministic" in arms_run
    agentic_recs = [r for r in recs if r.arm == "agentic"]
    assert len(agentic_recs) == 1
    assert "stub" in agentic_recs[0].note.lower()


def test_oracle_retrieval_loss_hook(tmp_path):
    """Oracle reads gold vs retrieved notes with the same (Echo) reader, free."""
    for fname, text in VAULT.items():
        (tmp_path / fname).write_text(text, encoding="utf-8")
    gold_paths = [str(tmp_path / GOLD_NOTE)]
    out = oracle_retrieval_loss(
        tmp_path,
        Question(text=QUESTION, gold_doc_ids=gold_paths),
        retriever=_graph_retriever(),
        top_k=8,
    )
    assert out["answer_gold"]
    assert out["answer_retrieved"]
    assert out["tokens_gold"] > 0
    assert out["recall_at_k"] == 1.0  # retriever found the gold note


# ---------------------------------------------------------------------------
# Chunking — the core token-minimal fix (FIX 1)
# ---------------------------------------------------------------------------
#
# A note longer than the chunk size must be split into several ~400-token
# chunks, each carrying its source-note identity, so a fat note never enters the
# reader whole and the wikilink graph still works at chunk granularity.

# >> CHUNK_SIZE_TOKENS whitespace words of prose, in real sentences so the
# sentence-aware chunker actually splits it. No query/wikilink tokens here.
_LONG_BODY = " ".join(
    f"Sentence number {i} describes some background lore about northern trade."
    for i in range(220)
)  # ~1.5k words >> 400-token chunk size → several chunks


def test_long_note_produces_multiple_chunks():
    """A note longer than the chunk size splits into >1 chunk (FIX 1)."""
    parts = _chunk_text(_LONG_BODY)
    assert len(parts) > 1, "a long note must be split into multiple chunks"
    # Each chunk is bounded near the configured size (overlap allows a margin).
    for p in parts:
        assert len(p.split()) <= CHUNK_SIZE_TOKENS + 5


def test_chunking_preserves_note_identity_on_disk(tmp_path):
    """Every chunk of a note shares one note slug; chunk_id = slug#chunkN."""
    (tmp_path / "Lore.md").write_text(_LONG_BODY, encoding="utf-8")
    chunks = _load_vault_chunks(tmp_path)
    lore = [c for c in chunks if c.metadata["note_slug"] == "lore"]
    assert len(lore) > 1, "long note should yield several chunks"
    for i, c in enumerate(lore):
        assert c.doc_id == "lore"  # doc_id is the note slug, not the path
        assert c.chunk_id == f"lore#chunk{i}"  # slug#chunkN identity


def test_retrieval_returns_chunks_not_whole_note(tmp_path):
    """Retrieval returns the matching CHUNK, not the entire (fat) note."""
    # One sentence in the middle is the only lexical match for the query.
    needle = "The vault keystone artifact is named Aurelith."
    body = (
        _LONG_BODY
        + " "
        + needle
        + " "
        + " ".join(f"More filler sentence {i} about weather." for i in range(220))
    )
    (tmp_path / "Vault.md").write_text(body, encoding="utf-8")
    (tmp_path / "Decoy.md").write_text(
        "Completely unrelated recipe text about soup.", encoding="utf-8"
    )

    rag = MemoryRAG(
        retriever=_graph_retriever(max_hops=0),  # lexical only: isolate chunking
        reader=EchoReader(),
    ).index(tmp_path)

    chunks, tokens = rag.retrieve("What is the keystone artifact named?", top_k=3)
    # The whole note is many chunks; we must NOT get all of them.
    vault_chunks = [c for c in chunks if c.metadata["note_slug"] == "vault"]
    assert vault_chunks, "the answer-bearing note must be retrieved"
    whole_note_tokens = count_tokens(body)
    assert tokens < whole_note_tokens, (
        "must feed the matching chunk(s), not the whole note: "
        f"{tokens} vs whole={whole_note_tokens}"
    )
    # The retrieved Vault chunk actually contains the needle sentence.
    assert any("Aurelith" in c.text for c in vault_chunks)


def test_multi_note_query_token_budget_is_bounded(tmp_path):
    """Total tokens fed for a query is well under a whole-vault dump."""
    # Several fat notes; only a small chunk-set should reach the reader.
    for n in range(6):
        (tmp_path / f"Topic{n}.md").write_text(
            _LONG_BODY + f" Topic {n} keyword alpha shared.", encoding="utf-8"
        )
    rag = MemoryRAG(retriever=_graph_retriever(), reader=EchoReader()).index(tmp_path)

    _, tokens = rag.retrieve("alpha shared keyword", top_k=4)
    whole_vault = count_tokens(_LONG_BODY) * 6
    assert tokens > 0
    assert tokens < whole_vault // 3, (
        f"retrieval must feed far fewer tokens than the whole vault: "
        f"{tokens} vs vault≈{whole_vault}"
    )


# ---------------------------------------------------------------------------
# Multi-hop still works across CHUNKED notes (FIX 1 graph integrity)
# ---------------------------------------------------------------------------
#
# The target note is now LONG (multi-chunk). The wikilink from the seed must
# still connect to the right chunk of the target, proving chunk-level identity
# keeps the graph intact.


def test_multihop_across_chunked_long_target(tmp_path):
    """A [[wikilink]] to a long (multi-chunk) note still connects the dots."""
    # Seed note: lexically matches the query, links to the long target.
    (tmp_path / "Eldoria.md").write_text(
        "Eldoria kingdom capital. The crown seat lives in [[Brightport]].",
        encoding="utf-8",
    )
    # Target note is LONG (many chunks); the answer phrase sits deep inside,
    # and the note shares ZERO query tokens, so it is reachable only by the hop.
    answer = "The royal mint of the harbour city forged the eternal seal."
    long_target = (
        " ".join(f"Harbour lore line {i} about ships and tides." for i in range(220))
        + " "
        + answer
        + " "
        + " ".join(f"Trailing lore line {i} about cargo." for i in range(220))
    )
    (tmp_path / "Brightport.md").write_text(long_target, encoding="utf-8")

    rag = MemoryRAG(retriever=_graph_retriever(max_hops=2), reader=EchoReader())
    rag.index(tmp_path)

    pairs = rag.retrieve_with_hops("Where is Eldoria's crown seat?", top_k=8)
    by_slug = {}
    for c, hop in pairs:
        by_slug.setdefault(c.metadata["note_slug"], hop)
    assert by_slug.get("eldoria") == 0, "lexical seed is hop 0"
    assert by_slug.get("brightport", -1) >= 1, (
        "the long target note must be reached via the wikilink graph "
        f"(chunk-level), got {by_slug}"
    )


# ---------------------------------------------------------------------------
# Index/TOC hub exclusion (FIX 2)
# ---------------------------------------------------------------------------
#
# MEMORY.md is a pure-navigation hub: it should contribute its [[links]] as
# graph EDGES but its own content must be excluded from the candidate set.


def _hub_vault(tmp_path):
    # A TOC hub that lexically matches the query and links to a topic note.
    (tmp_path / "MEMORY.md").write_text(
        "INDEX of everything: benchmark numbers, configs, repos. "
        "See [[Benchmark]] for the verified benchmark numbers.",
        encoding="utf-8",
    )
    # The real answer note: reachable from the hub's [[Benchmark]] link, and it
    # ALSO lexically matches, so it should be retrieved on its own merits.
    (tmp_path / "Benchmark.md").write_text(
        "Benchmark verified numbers: the system reaches parity on the suite.",
        encoding="utf-8",
    )
    (tmp_path / "Other.md").write_text(
        "Unrelated note about gardening.", encoding="utf-8"
    )
    return tmp_path


def test_exclude_content_drops_memory_md_chunks(tmp_path):
    """Default exclude drops MEMORY.md content from the candidate set."""
    _hub_vault(tmp_path)
    rag = MemoryRAG(retriever=_graph_retriever(max_hops=2)).index(tmp_path)  # default exclude
    chunks, _ = rag.retrieve("verified benchmark numbers", top_k=8)
    slugs = {c.metadata["note_slug"] for c in chunks}
    assert "memory" not in slugs, "MEMORY.md content must be excluded"
    assert "benchmark" in slugs, "the real answer note must still be retrieved"


def test_exclude_content_keeps_memory_md_edges(tmp_path):
    """Excluded MEMORY.md still contributes its [[links]] as graph edges."""
    _hub_vault(tmp_path)
    # Make Benchmark reachable ONLY via the hub edge: strip its lexical overlap
    # by querying tokens that match only the hub, then confirm the hop lands.
    rag = MemoryRAG(retriever=_graph_retriever(max_hops=2)).index(tmp_path)
    # The hub lexically matches "INDEX of everything"; from it, the [[Benchmark]]
    # edge must reach the Benchmark note even though the hub itself is excluded
    # from the returned candidates.
    pairs = rag.retrieve_with_hops("index of everything repos configs", top_k=8)
    slugs = {c.metadata["note_slug"] for c, _ in pairs}
    assert "memory" not in slugs, "hub content excluded from results"
    assert "benchmark" in slugs, (
        "the hub's [[Benchmark]] edge must still connect to the topic note "
        "(edges-only behaviour), got " + str(slugs)
    )


def test_exclude_content_off_keeps_memory_md(tmp_path):
    """exclude_content=() keeps MEMORY.md content retrievable."""
    _hub_vault(tmp_path)
    rag = MemoryRAG(
        retriever=_graph_retriever(max_hops=2), exclude_content=()
    ).index(tmp_path)
    chunks, _ = rag.retrieve("INDEX of everything benchmark", top_k=8)
    slugs = {c.metadata["note_slug"] for c in chunks}
    assert "memory" in slugs, "with exclusion off, MEMORY.md content is a candidate"


def test_slugify_note_matches_graph_identity():
    """WikiMoth's slugifier agrees with GraphRetriever's note identity."""
    from wikimoth.retrieval import _slugify as graph_slugify

    for name in ("MEMORY.md", "Foo/Bar Baz.md", "project_alpha-beta.markdown"):
        assert _slugify_note(name) == graph_slugify(name)
