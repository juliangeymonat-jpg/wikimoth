# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Run the WikiMoth 4-arm benchmark — the FREE portion (zero API cost).

Generates the frozen synthetic gold vault, then measures, with the API-free
``EchoReader``, the signals that do NOT need a paid reader:

  * tokens fed to the reader per arm  (the thing you pay for)
  * Recall@k vs the gold note-chain
  * hop-only recall   (gold notes reachable ONLY via [[wikilinks]])
  * reproducibility   (run-to-run drift of the retrieved note-set)
  * oracle retrieval-loss token gap   (reader-on-gold vs reader-on-retrieved)

What this canNOT show for free (needs an approved API budget) and is therefore
reported as TODO, not faked:

  * answer correctness (LLM-as-judge)   — needs a real reader
  * the `agentic` arm                   — needs real LLM tool-calls
  * the real Claude reader / real Headroom answer divergence

Run:
    PYTHONPATH=. python examples/run_benchmark.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wikimoth.benchmark import FourArmHarness, generate_corpus, summarize
from wikimoth.benchmark.harness import Question


def main() -> None:
    hops = (1, 2, 3)
    n_questions = 60
    top_k = 8

    vault = Path(tempfile.mkdtemp(prefix="wikimoth_bench_"))
    gold = generate_corpus(
        vault, n_questions=n_questions, hops=hops, n_distractors=200, seed=7
    )
    questions = [
        Question(
            text=g.text,
            gold_doc_ids=g.gold_doc_ids,
            hop_only_doc_ids=g.hop_only_doc_ids,
            hop=g.hop,
        )
        for g in gold
    ]

    # max_hops must cover the deepest chain so endpoints are reachable.
    from wikimoth.retrieval import GraphRetriever

    retriever = GraphRetriever(source="wikilinks", max_hops=max(hops))
    harness = FourArmHarness(vault, retriever=retriever, top_k=top_k)

    arms = ("dump", "deterministic", "deterministic_compacted")
    records = harness.run(questions, arms=arms)
    summ = summarize(records)

    # Reproducibility on a sample of questions (deterministic retrieval → 1).
    repro = [harness.retrieval_reproducibility(q, repeats=5) for q in questions[:10]]
    all_det = all(r["deterministic"] for r in repro)

    # ---- report ----
    n_notes = len(list(vault.rglob("*.md")))
    print(f"\nWikiMoth 4-arm benchmark (FREE portion, EchoReader, $0)")
    print(f"vault: {n_notes} notes  |  questions: {len(questions)} "
          f"(hops {hops})  |  top_k={top_k}\n")

    dump_tok = summ.get("dump", {}).get("mean_tokens") or 0
    header = f"{'arm':<26}{'mean tokens':>12}{'recall@k':>10}{'hop-only':>10}{'vs dump':>10}"
    print(header)
    print("-" * len(header))
    for arm in arms:
        s = summ.get(arm, {})
        mt = s.get("mean_tokens") or 0
        rk = s.get("mean_recall_at_k")
        ho = s.get("mean_hop_only_recall")
        ratio = (mt / dump_tok) if dump_tok else 1.0
        rk_s = f"{rk:.3f}" if rk is not None else "  -  "
        ho_s = f"{ho:.3f}" if ho is not None else "  -  "
        print(f"{arm:<26}{mt:>12.0f}{rk_s:>10}{ho_s:>10}{ratio:>9.1%}")

    det_tok = summ.get("deterministic", {}).get("mean_tokens") or 0
    if dump_tok:
        print(f"\ntoken reduction (deterministic vs dump): "
              f"{(1 - det_tok / dump_tok):.1%}")
    print(f"reproducibility (10 q × 5 repeats): "
          f"{'deterministic — 1 distinct result each' if all_det else 'DRIFT DETECTED'}")

    # Oracle retrieval-loss token gap on a sample (free; Echo answers).
    from wikimoth.benchmark.harness import oracle_retrieval_loss

    o = oracle_retrieval_loss(vault, questions[1], retriever=retriever, top_k=top_k)
    print(f"oracle sample (q#1, {questions[1].hop}-hop): "
          f"tokens_gold={o['tokens_gold']} tokens_retrieved={o['tokens_retrieved']} "
          f"recall@k={o['recall_at_k']}")

    print("\nTODO (needs approved API budget, NOT faked):")
    print("  - answer correctness via LLM-judge (real reader)")
    print("  - agentic arm (real LLM tool-calls): tokens + reproducibility-drift contrast")
    print("  - real Claude reader / real Headroom answer divergence")

    out = vault / "benchmark_results.json"
    out.write_text(json.dumps({"summary": summ, "reproducibility": repro}, indent=2),
                   encoding="utf-8")
    print(f"\nresults JSON: {out}")


if __name__ == "__main__":
    main()
