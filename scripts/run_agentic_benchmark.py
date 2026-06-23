# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Run the agentic arm against real Claude and print the head-to-head table.

*Let the model prune its own context* vs deterministic retrieval vs dumping the
vault, on a frozen, reproducible ``[[wikilink]]`` corpus with known gold chains.

This is the **paid** script (real Claude tool-calls). It needs:

    pip install "wikimoth[claude,tokens]"
    export ANTHROPIC_API_KEY=...        # in your shell, not committed

Then, for example:

    python scripts/run_agentic_benchmark.py --n 12 --model claude-sonnet-4-6 \
        --repeats 3 --out agentic_benchmark.json

It reports, per arm, mean tokens fed to the reader, recall of the gold note-chain,
hop-only recall (the link-only notes a flat retriever can't reach), and for the
agentic arm: the real billed API tokens, exact-match answer rate, and how much its
self-curated context drifts run to run (the determinism contrast). The deterministic
and dump arms use the free EchoReader (token volume + recall are what we compare);
the agentic arm necessarily uses the real model to browse and answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import mkdtemp

# Make the package importable when run as `python scripts/run_agentic_benchmark.py`
# (the script's own dir, not the repo root, lands on sys.path[0] otherwise).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _hops(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="number of questions")
    ap.add_argument("--hops", type=_hops, default=(1, 2, 3), help="hop lengths, e.g. 1,2,3")
    ap.add_argument("--decoys", type=int, default=120, help="lexical decoy notes")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default="claude-sonnet-4-6", help="Claude model for the agent")
    ap.add_argument("--steps", type=int, default=14, help="max browse steps per question")
    ap.add_argument("--search-limit", type=int, default=10, help="max search hits returned")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=3, help="determinism repeats (first M questions)")
    ap.add_argument("--repeat-questions", type=int, default=3, help="how many questions to repeat")
    ap.add_argument("--vault", default=None, help="use an existing vault instead of the synthetic corpus")
    ap.add_argument("--out", default=None, help="write the full record JSON here")
    args = ap.parse_args()

    # Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Export it in your shell first:\n"
            "    export ANTHROPIC_API_KEY=...\n"
            "(this script makes real, paid Claude tool-calls).",
            file=sys.stderr,
        )
        return 2

    from wikimoth.benchmark import (
        FourArmHarness,
        generate_realistic_corpus,
        summarize,
    )
    from wikimoth.benchmark.agentic import AnthropicAgenticModel
    from wikimoth.benchmark.harness import Question
    from wikimoth.reader import EchoReader
    from wikimoth.retrieval import GraphRetriever
    from wikimoth.tokens import token_backend

    # ---- corpus ----------------------------------------------------------
    if args.vault:
        vault = Path(args.vault)
        gold = []  # no gold for a BYO vault → recall/EM are reported as n/a
        questions = [Question(text=q) for q in []]  # caller must supply questions
        print("BYO vault provided but this script needs gold chains; use the "
              "synthetic corpus (omit --vault) for the graded comparison.", file=sys.stderr)
        return 2
    else:
        vault = Path(mkdtemp(prefix="wikimoth_agentic_"))
        gold = generate_realistic_corpus(
            vault, n_questions=args.n, hops=args.hops, n_decoys=args.decoys, seed=args.seed
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

    # ---- harness ---------------------------------------------------------
    max_hops = max(args.hops)
    model = AnthropicAgenticModel(
        model=args.model, max_tokens=1024, temperature=args.temperature
    )
    h = FourArmHarness(
        vault,
        reader=EchoReader(),  # free reader for dump/deterministic (token+recall story)
        retriever=GraphRetriever(source="wikilinks", max_hops=max_hops, seed_top_k=2),
        top_k=8,
        agentic_model=model,
        agentic_max_steps=args.steps,
        agentic_search_limit=args.search_limit,
    )

    print(f"corpus: {len(questions)} questions, hops={args.hops}, decoys={args.decoys}, "
          f"vault={vault}\nmodel: {args.model}  token_backend={token_backend()}\n")

    # ---- run arms --------------------------------------------------------
    records = []
    em_hits = 0
    em_total = 0
    for i, (q, g) in enumerate(zip(questions, gold), 1):
        for arm in ("dump", "deterministic", "agentic"):
            rec = h.run_arm(arm, q)
            records.append(rec)
            if arm == "agentic":
                em_total += 1
                hit = g.answer and (g.answer.lower() in (rec.answer or "").lower())
                em_hits += 1 if hit else 0
                print(f"[{i:>2}/{len(questions)}] hop={g.hop} agentic: "
                      f"content_tokens={rec.tokens_fed_to_reader:>5} "
                      f"recall={rec.retrieval_recall_at_k} hop_only={rec.hop_only_recall} "
                      f"EM={'Y' if hit else 'n'}  ({rec.note})")

    # ---- determinism (paid; first few questions) -------------------------
    drift = []
    for q in questions[: max(0, args.repeat_questions)]:
        rep = h.agentic_reproducibility(q, repeats=args.repeats)
        drift.append(rep)

    # ---- summary ---------------------------------------------------------
    s = summarize(records)
    print("\n=== mean tokens fed to the reader (+ retrieval reach) ===")
    print(f"{'arm':<16}{'n':>4}{'mean_tokens':>14}{'recall@k':>12}{'hop_only':>12}")
    for arm in ("dump", "deterministic", "agentic"):
        if arm not in s:
            continue
        a = s[arm]
        def _f(x):
            return f"{x:.3f}" if isinstance(x, float) else ("n/a" if x is None else str(x))
        print(f"{arm:<16}{a['n']:>4}{a['mean_tokens']:>14.0f}"
              f"{_f(a['mean_recall_at_k']):>12}{_f(a['mean_hop_only_recall']):>12}")

    agentic_recs = [r for r in records if r.arm == "agentic"]
    api_in = sum(r.api_input_tokens for r in agentic_recs)
    api_out = sum(r.api_output_tokens for r in agentic_recs)
    det_tokens = s.get("deterministic", {}).get("mean_tokens") or 0
    mult = (api_in / len(agentic_recs) / det_tokens) if (det_tokens and agentic_recs) else 0
    print(f"\nagentic real billed tokens (all questions): input={api_in}  output={api_out}")
    if mult:
        print(f"  -> mean billed input per question is ~{mult:.1f}x the deterministic "
              f"arm's tokens fed to the reader")
    print(f"agentic exact-match answers: {em_hits}/{em_total}")
    if drift:
        nondet = sum(1 for d in drift if not d["deterministic"])
        print(f"agentic determinism: {nondet}/{len(drift)} questions gave a DIFFERENT "
              f"note-set across {args.repeats} runs "
              f"(distinct sets: {[d['distinct_results'] for d in drift]})")
        print("deterministic arm: 1 distinct note-set by construction (bit-stable).")

    if args.out:
        payload = {
            "config": vars(args) | {"hops": list(args.hops)},
            "token_backend": token_backend(),
            "summary": s,
            "agentic_billed": {"input": api_in, "output": api_out},
            "agentic_exact_match": {"hits": em_hits, "total": em_total},
            "agentic_determinism": drift,
            "records": [vars(r) for r in records],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
