# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Apples-to-apples baselines — real BM25 + dense (MiniLM) vs graph ($0, no key).

Upgrades the differentiator from "GraphRetriever with traversal off" to *real*
flat retrievers: Okapi BM25 (sparse) and a sentence-transformers MiniLM
bi-encoder (semantic). The honest question: does a SEMANTIC retriever reach the
link-only answer note where sparse cannot? Reports recall@k / hop-only recall /
answer-note-retrieved / mean tokens, on the realistic decoy-rich corpus.

Run:
    PYTHONPATH=. python examples/baselines_benchmark.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from wikimoth.benchmark import FourArmHarness, generate_realistic_corpus, summarize
from wikimoth.benchmark.baselines import BM25Retriever, STDenseRetriever
from wikimoth.benchmark.harness import Question
from wikimoth.tokens import count_passage_tokens


def _answer_found(harness, questions, gold, top_k):
    found = 0
    for q, g in zip(questions, gold):
        chunks, _ = harness._rag.retrieve(q.text, top_k=top_k)
        slugs = {c.metadata.get("note_slug", "") for c in chunks}
        if g.gold_doc_ids[-1].replace("-", " ") in slugs:
            found += 1
    return found / len(questions)


def main() -> None:
    hops = (1, 2, 3)
    n_questions = 30
    n_decoys = 100
    top_k = 8

    vault = Path(tempfile.mkdtemp(prefix="wikimoth_baselines_"))
    gold = generate_realistic_corpus(vault, n_questions=n_questions, hops=hops,
                                     n_decoys=n_decoys, seed=11)
    questions = [Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                          hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
                 for g in gold]

    from wikimoth.retrieval import GraphRetriever

    n_notes = len(list(vault.rglob("*.md")))
    dump_tokens = count_passage_tokens(
        [getattr(c, "text", "") or "" for c in FourArmHarness(vault, top_k=top_k)._chunks])

    print(f"\napples-to-apples baselines — realistic decoy-rich corpus ($0)")
    print(f"vault: {n_notes} notes ({n_decoys} decoys)  |  {n_questions} questions "
          f"(hops {hops})  |  top_k={top_k}  |  dump={dump_tokens} tok/query\n")

    from wikimoth.hybrid import HybridRetriever
    builders = [
        ("BM25 (sparse)", lambda: BM25Retriever()),
        ("dense MiniLM (semantic)", lambda: STDenseRetriever()),
        ("graph (seed=2 + traverse)",
         lambda: GraphRetriever(source="wikilinks", max_hops=3, seed_top_k=2)),
        ("hybrid (BM25 seed + reserve)",
         lambda: HybridRetriever(source="wikilinks", max_hops=3, seed_top_k=1, graph_reserve=4)),
    ]

    head = f"{'retriever':<28}{'recall@k':>10}{'hop-only':>10}{'answer found':>14}{'mean tok':>10}"
    print(head)
    print("-" * len(head))
    for label, build in builders:
        try:
            retriever = build()
        except Exception as e:  # dense may fail (download/env)
            print(f"{label:<28}  SKIPPED: {type(e).__name__}: {str(e)[:50]}")
            continue
        h = FourArmHarness(vault, retriever=retriever, top_k=top_k)
        s = summarize(h.run(questions, arms=("deterministic",)))["deterministic"]
        found = _answer_found(h, questions, gold, top_k)
        print(f"{label:<28}{s['mean_recall_at_k']:>10.3f}{s['mean_hop_only_recall']:>10.3f}"
              f"{found:>13.1%}{s['mean_tokens']:>10.0f}")

    print("\nReading: sparse (BM25) and even semantic (dense) retrieval rank the "
          "lexically/topically similar project anchors + decoys at the top; the "
          "answer note (generic 'coordinator stationed at CITY{i}', no shared "
          "words, only a [[wikilink]] away) stays out of top-k. Only graph "
          "traversal reaches it.")


if __name__ == "__main__":
    main()
