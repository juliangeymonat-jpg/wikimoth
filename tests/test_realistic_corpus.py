# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the realistic (decoy-rich) corpus + the flat-vs-graph differentiator.

The credible claim: with active lexical decoys, a FLAT retriever retrieves the
answer note 0% of the time (misled by decoys), while the GRAPH (seed narrowly +
traverse links) retrieves it 100%. Locked here so it can't silently regress.
"""

from __future__ import annotations

import pytest

from wikimoth.benchmark.corpus import generate_realistic_corpus


def _answer_found_rate(harness, questions, gold, top_k):
    found = 0
    for q, g in zip(questions, gold):
        chunks, _ = harness._rag.retrieve(q.text, top_k=top_k)
        slugs = {c.metadata.get("note_slug", "") for c in chunks}
        if g.gold_doc_ids[-1].replace("-", " ") in slugs:
            found += 1
    return found / len(questions)


def test_realistic_corpus_is_deterministic(tmp_path):
    a = generate_realistic_corpus(tmp_path / "a", n_questions=8, n_decoys=20, seed=11)
    b = generate_realistic_corpus(tmp_path / "b", n_questions=8, n_decoys=20, seed=11)
    assert [(q.text, q.gold_doc_ids, q.answer, q.hop) for q in a] == \
           [(q.text, q.gold_doc_ids, q.answer, q.hop) for q in b]
    # same note files, byte for byte
    for p in (tmp_path / "a").glob("*.md"):
        assert p.read_text(encoding="utf-8") == \
               (tmp_path / "b" / p.name).read_text(encoding="utf-8")


def test_realistic_structure_and_decoys(tmp_path):
    vault = tmp_path / "v"
    gold = generate_realistic_corpus(vault, n_questions=9, hops=(1, 2, 3),
                                     n_decoys=30, seed=11)
    assert len(list(vault.glob("memo-*.md"))) == 30
    for g in gold:
        assert len(g.gold_doc_ids) == g.hop + 1
        assert g.hop_only_doc_ids == g.gold_doc_ids[1:]
        assert g.gold_doc_ids[0].endswith("-anchor")
        assert g.gold_doc_ids[-1].endswith("-answer")
        assert g.answer.startswith("CITY")


@pytest.mark.parametrize("_", [0])
def test_flat_misses_graph_finds(tmp_path, _):
    from wikimoth.retrieval import GraphRetriever
    from wikimoth.benchmark import FourArmHarness, summarize
    from wikimoth.benchmark.harness import Question

    vault = tmp_path / "v"
    gold = generate_realistic_corpus(vault, n_questions=18, hops=(1, 2, 3),
                                     n_decoys=120, seed=11)
    questions = [Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                          hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
                 for g in gold]
    top_k = 8

    flat = FourArmHarness(
        vault, retriever=GraphRetriever(source="wikilinks", max_hops=0), top_k=top_k)
    graph = FourArmHarness(
        vault, retriever=GraphRetriever(source="wikilinks", max_hops=3, seed_top_k=2),
        top_k=top_k)

    # Flat: misled by decoys/anchors, never reaches the answer note.
    assert _answer_found_rate(flat, questions, gold, top_k) == 0.0
    # Graph: seeds narrowly + traverses → answer note every time, hop-only intact.
    assert _answer_found_rate(graph, questions, gold, top_k) == 1.0
    gs = summarize(graph.run(questions, arms=("deterministic",)))["deterministic"]
    assert gs["mean_recall_at_k"] == 1.0
    assert gs["mean_hop_only_recall"] == 1.0


def test_real_bm25_baseline_also_misses_answer(tmp_path):
    """A REAL sparse retriever (Okapi BM25), not just lexical-graph, also 0%."""
    pytest.importorskip("rank_bm25")
    from wikimoth.benchmark import FourArmHarness
    from wikimoth.benchmark.baselines import BM25Retriever
    from wikimoth.benchmark.harness import Question

    vault = tmp_path / "v"
    gold = generate_realistic_corpus(vault, n_questions=12, hops=(1, 2, 3),
                                     n_decoys=80, seed=11)
    questions = [Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                          hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
                 for g in gold]
    h = FourArmHarness(vault, retriever=BM25Retriever(), top_k=8)
    assert _answer_found_rate(h, questions, gold, 8) == 0.0


def test_hybrid_wins_multihop(tmp_path):
    """Hybrid (BM25 seed + traverse) reaches the link-only answer like graph."""
    pytest.importorskip("rank_bm25")
    from wikimoth.benchmark import FourArmHarness
    from wikimoth.benchmark.harness import Question
    from wikimoth.hybrid import HybridRetriever

    vault = tmp_path / "v"
    gold = generate_realistic_corpus(vault, n_questions=18, hops=(1, 2, 3),
                                     n_decoys=120, seed=11)
    questions = [Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                          hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
                 for g in gold]
    # Multi-hop-tuned: narrow seed (avoid sibling-anchor flooding) + enough
    # reserve slots to reach the deepest (hop-3) answer.
    h = FourArmHarness(
        vault,
        retriever=HybridRetriever(source="wikilinks", max_hops=3, seed_top_k=1, graph_reserve=4),
        top_k=8)
    assert _answer_found_rate(h, questions, gold, 8) == 1.0


def test_hybrid_handles_direct_lookup(tmp_path):
    """Hybrid also nails a direct lookup (BM25 seed), where graph-pure can lag."""
    pytest.importorskip("rank_bm25")
    from wikimoth import MemoryRAG
    from wikimoth.hybrid import HybridRetriever

    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "alpha.md").write_text(
        "---\nname: alpha\n---\nThe capital of Wonderland is Florin City.\n", encoding="utf-8")
    (vault / "beta.md").write_text(
        "---\nname: beta\n---\nAn unrelated pasta recipe with tomato.\n", encoding="utf-8")
    rag = MemoryRAG(
        retriever=HybridRetriever(source="wikilinks", max_hops=2, seed_top_k=3),
        exclude_content=())
    rag.index(vault)
    chunks, _ = rag.retrieve("What is the capital of Wonderland?", top_k=2)
    slugs = {c.metadata.get("note_slug", "") for c in chunks}
    assert "alpha" in slugs
