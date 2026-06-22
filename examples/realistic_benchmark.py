# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Realistic (decoy-rich) benchmark — flat vs graph + reader pairs ($0, no key).

The credible version of the differentiator: distractor "memo" notes share query
words (they actively mislead a flat/lexical retriever and bloat a dump), while
the answer lives only at the end of a [[wikilink]] chain. Reports, for the same
vault and scoring with traversal off vs on:
  * Recall@k, hop-only recall, "answer note retrieved" rate
  * mean tokens fed (deterministic vs dump)
and emits the GRAPH-retrieved vs FLAT-retrieved passages for a sample so the
in-session Claude can answer (graph → answerable; flat → answer absent).

Run:
    PYTHONPATH=. python examples/realistic_benchmark.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wikimoth.benchmark import FourArmHarness, generate_realistic_corpus, summarize
from wikimoth.benchmark.harness import Question
from wikimoth.tokens import count_passage_tokens


def _answer_note_found(harness, questions, gold, top_k):
    found = 0
    for q, g in zip(questions, gold):
        chunks, _ = harness._rag.retrieve(q.text, top_k=top_k)
        slugs = {c.metadata.get("note_slug", "") for c in chunks}
        if g.gold_doc_ids[-1].replace("-", " ") in slugs:
            found += 1
    return found / len(questions)


def main() -> None:
    hops = (1, 2, 3)
    n_questions = 60
    top_k = 8
    n_sample = 8

    vault = Path(tempfile.mkdtemp(prefix="wikimoth_realistic_"))
    gold = generate_realistic_corpus(vault, n_questions=n_questions, hops=hops,
                                     n_decoys=200, seed=11)
    questions = [
        Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                 hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
        for g in gold
    ]

    from wikimoth.retrieval import GraphRetriever
    # Flat = lexical top-k (traversal off, broad lexical seeding). Graph = seed
    # NARROWLY on the best match and trust the links (seed_top_k=2): under heavy
    # lexical decoys, broad seeding lets decoys (proximity 1.0) crowd the
    # BFS-reached chain out of top-k — a graph retriever's whole point is to not
    # rely on lexical breadth.
    configs = {
        "flat (lexical, max_hops=0)": GraphRetriever(source="wikilinks", max_hops=0),
        "graph (seed_top_k=2, max_hops=3)":
            GraphRetriever(source="wikilinks", max_hops=3, seed_top_k=2),
    }

    dump_tokens = count_passage_tokens(
        [getattr(c, "text", "") or "" for c in FourArmHarness(vault, top_k=top_k)._chunks]
    )

    print(f"\nrealistic decoy-rich benchmark — flat vs graph ($0)")
    print(f"vault: {len(list(vault.rglob('*.md')))} notes (incl. 200 lexical decoys)"
          f"  |  {n_questions} questions (hops {hops})  |  top_k={top_k}")
    print(f"dump (whole vault) = {dump_tokens} tokens/query\n")
    head = f"{'retriever':<34}{'recall@k':>10}{'hop-only':>10}{'answer found':>14}{'mean tok':>10}"
    print(head)
    print("-" * len(head))

    graph_harness = None
    for label, retriever in configs.items():
        h = FourArmHarness(vault, retriever=retriever, top_k=top_k)
        s = summarize(h.run(questions, arms=("deterministic",)))["deterministic"]
        found = _answer_note_found(h, questions, gold, top_k)
        print(f"{label:<34}{s['mean_recall_at_k']:>10.3f}{s['mean_hop_only_recall']:>10.3f}"
              f"{found:>13.1%}{s['mean_tokens']:>10.0f}")
        if "graph" in label:
            graph_harness = h

    # Emit reader pairs (graph vs flat) for the in-session agent to answer.
    flat = FourArmHarness(vault, retriever=GraphRetriever(source="wikilinks", max_hops=0), top_k=top_k)
    items = []
    for i in range(n_sample):
        q, g = questions[i], gold[i]
        gchunks, _ = graph_harness._rag.retrieve(q.text, top_k=top_k)
        fchunks, _ = flat._rag.retrieve(q.text, top_k=top_k)
        items.append({
            "idx": i, "hop": g.hop, "question": q.text, "gold_answer": g.answer,
            "graph_passages": [getattr(c, "text", "") or "" for c in gchunks],
            "flat_passages": [getattr(c, "text", "") or "" for c in fchunks],
        })
    out = vault / "reader_pairs.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== reader pairs (sample) — GRAPH vs FLAT retrieved context ===")
    for it in items:
        print(f"\n--- Q{it['idx']} (hop={it['hop']}) : {it['question']}  [gold={it['gold_answer']}]")
        print("  GRAPH passages:")
        for p in it["graph_passages"]:
            print(f"    - {' '.join(p.split())}")
        print("  FLAT passages:")
        for p in it["flat_passages"]:
            print(f"    - {' '.join(p.split())}")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
