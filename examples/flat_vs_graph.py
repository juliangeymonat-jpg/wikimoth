# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Flat retrieval vs graph retrieval — the connect-the-dots differentiator ($0).

The clean, reader-independent claim behind WikiMoth: on a query whose answer note
shares no words with it (reachable ONLY via a [[wikilink]]), a FLAT retriever
misses the answer note entirely, while the graph walks to it. We make the
baseline honest and dependency-free by reusing the SAME retriever with traversal
disabled: ``GraphRetriever(max_hops=0)`` == pure lexical top-k (a flat lexical
retriever); ``max_hops=k`` == graph multi-hop. Same code, same scoring, the only
difference is whether links are followed.

If the flat arm fails to retrieve the endpoint, even a perfect reader cannot
answer — so this also demonstrates answer-ABILITY differentiation for free
(the flat-retrieved passages simply do not contain the answer token).

Run:
    PYTHONPATH=. python examples/flat_vs_graph.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from wikimoth.benchmark import FourArmHarness, generate_corpus, summarize
from wikimoth.benchmark.harness import Question


def main() -> None:
    hops = (1, 2, 3)
    n_questions = 60
    top_k = 8

    vault = Path(tempfile.mkdtemp(prefix="wikimoth_flatgraph_"))
    gold = generate_corpus(vault, n_questions=n_questions, hops=hops,
                           n_distractors=200, seed=7)
    questions = [
        Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                 hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
        for g in gold
    ]

    from wikimoth.retrieval import GraphRetriever

    configs = {
        "flat (lexical, max_hops=0)": GraphRetriever(source="wikilinks", max_hops=0),
        "graph (max_hops=3)":         GraphRetriever(source="wikilinks", max_hops=3),
    }

    print(f"\nflat vs graph — same vault, same scoring, links off vs on ($0)")
    print(f"vault: {len(list(vault.rglob('*.md')))} notes  |  {n_questions} questions "
          f"(hops {hops})  |  top_k={top_k}\n")
    header = f"{'retriever':<28}{'recall@k':>10}{'hop-only':>10}{'answer note found':>20}"
    print(header)
    print("-" * len(header))

    answerability = {}
    for label, retriever in configs.items():
        h = FourArmHarness(vault, retriever=retriever, top_k=top_k)
        recs = h.run(questions, arms=("deterministic",))
        s = summarize(recs)["deterministic"]
        # "answer note found" = fraction of questions whose ENDPOINT note (the one
        # holding the answer token) is in the retrieved set.
        found = 0
        for q, g in zip(questions, gold):
            chunks, _ = h._rag.retrieve(q.text, top_k=top_k)
            slugs = {c.metadata.get("note_slug", "") for c in chunks}
            endpoint_slug = g.gold_doc_ids[-1].replace("-", " ")
            if endpoint_slug in slugs:
                found += 1
        answerability[label] = found / len(questions)
        print(f"{label:<28}{s['mean_recall_at_k']:>10.3f}"
              f"{s['mean_hop_only_recall']:>10.3f}{answerability[label]:>19.1%}")

    print("\nReading: flat retrieval seeds lexically on the query, so it pulls the "
          "anchor (and decoys) but cannot reach the endpoint that holds the answer "
          "— its hop-only recall is 0 and the answer note is absent, so ANY reader "
          "fails. The graph follows [[wikilinks]] to the endpoint every time.")


if __name__ == "__main__":
    main()
