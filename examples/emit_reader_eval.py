# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Emit reader-eval pairs for the in-session Claude to answer (no API, no key).

WikiMoth's product reader IS the agent already in the session. So instead of
paying a second LLM via the API, we use the Claude that's running Claude Code as
the reader: this script retrieves the deterministic note-chain per question and
writes (question, passages, gold answer, token counts) to JSON. The agent reads
that JSON, answers each question from the passages ONLY, and grades by
exact-match on the planted answer token (synthetic corpus → no LLM-judge needed).

Run:
    PYTHONPATH=. python examples/emit_reader_eval.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wikimoth.benchmark import FourArmHarness, generate_corpus
from wikimoth.benchmark.harness import Question
from wikimoth.tokens import count_passage_tokens


def main() -> None:
    hops = (1, 2, 3)
    n_sample = 12          # questions the agent will read+answer
    n_dump_sample = 2      # questions we also emit the FULL dump for (token contrast)
    top_k = 8

    vault = Path(tempfile.mkdtemp(prefix="wikimoth_readereval_"))
    gold = generate_corpus(vault, n_questions=n_sample, hops=hops,
                           n_distractors=200, seed=7)
    questions = [
        Question(text=g.text, gold_doc_ids=g.gold_doc_ids,
                 hop_only_doc_ids=g.hop_only_doc_ids, hop=g.hop)
        for g in gold
    ]

    from wikimoth.retrieval import GraphRetriever
    retriever = GraphRetriever(source="wikilinks", max_hops=max(hops))
    harness = FourArmHarness(vault, retriever=retriever, top_k=top_k)

    # Whole-vault passages (dump arm) — shared across questions.
    dump_passages = [getattr(c, "text", "") or "" for c in harness._chunks]
    dump_tokens = count_passage_tokens(dump_passages)

    items = []
    for i, (q, g) in enumerate(zip(questions, gold)):
        chunks, det_tokens = harness._rag.retrieve(q.text, top_k=top_k)
        det_passages = [getattr(c, "text", "") or "" for c in chunks]
        item = {
            "idx": i,
            "hop": g.hop,
            "question": q.text,
            "gold_answer": g.answer,
            "deterministic_passages": det_passages,
            "deterministic_tokens": det_tokens,
            "dump_tokens": dump_tokens,
        }
        if i < n_dump_sample:
            item["dump_passages"] = dump_passages
        items.append(item)

    out = vault / "reader_eval.json"
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console: the deterministic-arm pairs (compact) for the agent to answer.
    print(f"vault: {vault}")
    print(f"dump tokens (whole vault): {dump_tokens}")
    print(f"questions: {len(items)} (deterministic chains); "
          f"full dump emitted for first {n_dump_sample}\n")
    for it in items:
        print(f"--- Q{it['idx']} (hop={it['hop']}, det_tokens={it['deterministic_tokens']}) ---")
        print(f"QUESTION: {it['question']}")
        print("PASSAGES (deterministic retrieval):")
        for p in it["deterministic_passages"]:
            print(f"  - {' '.join(p.split())}")
        print()
    print(f"JSON (incl. full dump for first {n_dump_sample}): {out}")


if __name__ == "__main__":
    main()
