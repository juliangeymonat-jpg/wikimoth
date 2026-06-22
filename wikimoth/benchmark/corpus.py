# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Synthetic gold corpus — a frozen ``[[wikilink]]`` benchmark vault.

No public benchmark does QA over a ``[[wikilink]]`` vault (research verdict, see
``MEMORY_BENCHMARK_DESIGN.md``), so we build one. This generator plants
multi-hop chains with **known gold note-chains and hop labels** — exactly the
ground truth Recall@k / hop-only-recall / oracle metrics need — and is fully
**deterministic** (seeded), so the benchmark is a re-runnable, shareable artifact
(no private data, no model in the loop).

Each question plants one chain of ``k`` hops::

    anchor ──[[link]]──▶ relay₁ ──▶ … ──▶ relay_{k-1} ──▶ endpoint

- **anchor** is the only note that lexically matches the question (it carries the
  unique ``topicNNNN`` token + the query's content words), so a lexical retriever
  seeds *here*.
- **relays + endpoint** share **zero** tokens with the question, so they are
  reachable **only by following ``[[wikilinks]]``** — the connect-the-dots
  property a flat retriever misses. The endpoint holds the answer token.

So ``gold`` = the whole chain; ``hop_only`` = everything past the anchor (the
notes a flat retriever cannot reach). Distractor "filing" notes add corpus bulk
and noise (they never carry a ``topicNNNN`` token, so they never seed a query).

The vocabularies of the question, the relays, and the endpoint are kept disjoint
*by construction* (see ``_QUERY_WORDS`` / ``_RELAY_WORDS`` / ``_ENDPOINT_WORDS``)
so the hop-only property is guaranteed, not hoped for.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

# Disjoint vocabularies — the guarantee behind "hop-only". The question's ONLY
# token that appears in ANY note is the unique ``topicNNNN`` (in anchor_i), so
# anchor_i is the SOLE lexical seed. Every other query word ("starting from
# what is the final destination") is absent from all note bodies; the anchor,
# relay, and endpoint vocabularies are mutually disjoint and disjoint from the
# query. This matters: GraphRetriever ranks seeds (proximity 1.0) above
# BFS-reached notes, so any spurious extra seed would crowd the real chain out
# of top-k. One seed → clean BFS down the chain → recall≈1, hop-only≈1.
_RELAY_WORDS = ("Relay junction.", "Forward hop.", "Passthrough segment.",
                "Intermediate waypoint.", "Continue along.")
_ENDPOINT_WORDS = "Endpoint marker recorded. Archive sealed."        # no query word


@dataclass
class GoldQuestion:
    """A benchmark question with its gold note-chain + hop label.

    Mirrors :class:`wikimoth.benchmark.harness.Question` plus the hop-only subset,
    so it can be passed straight to the harness (which reads ``text`` /
    ``gold_doc_ids`` and, if present, ``hop`` / ``hop_only_doc_ids``).
    """

    text: str
    gold_doc_ids: list[str] = field(default_factory=list)
    hop_only_doc_ids: list[str] = field(default_factory=list)
    hop: int = 0
    answer: str = ""


def _anchor_stem(i: int) -> str:
    return f"dossier-{i:04d}-anchor"


def _relay_stem(i: int, m: int) -> str:
    return f"dossier-{i:04d}-relay-{m}"


def _endpoint_stem(i: int) -> str:
    return f"dossier-{i:04d}-endpoint"


def _filing_stem(j: int) -> str:
    return f"filing-{j:05d}"


def _write_note(out: Path, stem: str, body: str) -> None:
    fm = f"---\nname: {stem}\ndescription: \"synthetic benchmark note\"\n---\n"
    out.joinpath(f"{stem}.md").write_text(fm + body + "\n", encoding="utf-8")


def generate_corpus(
    out_dir: str | Path,
    *,
    n_questions: int = 60,
    hops: tuple[int, ...] = (1, 2, 3),
    n_distractors: int = 200,
    seed: int = 7,
) -> list[GoldQuestion]:
    """Write a frozen synthetic vault to ``out_dir`` and return its questions.

    Parameters
    ----------
    n_questions
        Number of planted chains (= questions).
    hops
        Hop lengths to cycle through (chain length per question). The harness
        retriever's ``max_hops`` must be ``>= max(hops)`` to reach every
        endpoint (use ``GraphRetriever(source="wikilinks", max_hops=max(hops))``).
    n_distractors
        Number of noise "filing" notes (corpus bulk; never seed a query).
    seed
        PRNG seed for distractor link wiring (chain structure is fully
        determined by the question index, so the gold set is seed-independent).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    questions: list[GoldQuestion] = []

    for i in range(n_questions):
        k = hops[i % len(hops)]
        topic = f"topic{i:04d}"
        answer = f"ANSWERTOKEN{i:04d}"

        anchor = _anchor_stem(i)
        relays = [_relay_stem(i, m) for m in range(1, k)]   # k-1 relays
        endpoint = _endpoint_stem(i)
        chain = [anchor] + relays + [endpoint]

        # anchor: the ONLY note carrying the unique topic (so the ONLY lexical
        # seed for this query) + the first chain link. Its other words
        # ("dossier reference path begins here") are absent from the query, so
        # they create no spurious seeds across the 60 anchors.
        nxt = chain[1]
        _write_note(
            out, anchor,
            f"Dossier {topic}. Reference path begins here. [[{nxt}]]",
        )
        # relays: generic vocab only (no query words), link onward.
        for idx, r in enumerate(relays):
            nxt = chain[chain.index(r) + 1]
            phrase = _RELAY_WORDS[idx % len(_RELAY_WORDS)]
            _write_note(out, r, f"{phrase} Segment {idx + 1}. [[{nxt}]]")
        # endpoint: holds the answer, disjoint vocab, no outgoing chain link.
        _write_note(
            out, endpoint,
            f"{_ENDPOINT_WORDS} Marker value: {answer}.",
        )

        # Query: its ONLY token present in any note is `topic` (→ anchor_i).
        # "starting from what is the final destination" appear in no note body.
        text = f"Starting from {topic}, what is the final destination?"
        questions.append(
            GoldQuestion(
                text=text,
                gold_doc_ids=list(chain),
                hop_only_doc_ids=relays + [endpoint],
                hop=k,
                answer=answer,
            )
        )

    # Distractor "filing" notes: bulk + noise, optionally cross-linked among
    # themselves so the graph is not trivially star-shaped. They never carry a
    # topicNNNN token, so they never lexically seed a question.
    filings = [_filing_stem(j) for j in range(n_distractors)]
    for j, f in enumerate(filings):
        link = ""
        if filings and rng.random() < 0.5:
            tgt = rng.choice(filings)
            if tgt != f:
                link = f" [[{tgt}]]"
        _write_note(
            out, f,
            f"Filing {j:05d}. Routine archived record, no cross-reference of "
            f"note.{link}",
        )

    return questions


# ---------------------------------------------------------------------------
# Realistic corpus — decoy-rich (the credible differentiator)
# ---------------------------------------------------------------------------
# The clean corpus above proves the mechanism but its distractors share NO words
# with the query (flat retrieval fails by having nothing to match). The realistic
# corpus is the convincing version: distractor "memo" notes DO share query words
# (they actively mislead a flat/lexical retriever and bloat a dump), while the
# answer still lives only at the end of a [[wikilink]] chain. So flat retrieval
# is tempted toward decoys and misses the answer; the graph walks to it.
#
# Ranking guarantee (so the result is robust, not luck): the anchor contains the
# unique topic token PLUS every phrase word → it is always the single top seed
# (graph BFS starts there). Decoys contain a SUBSET of the phrase words → they
# rank below the anchor but above the chain. The chain (links + answer) contains
# NO phrase word → flat never seeds it; only graph traversal reaches it.
_PHRASE = ("city", "team", "lead", "responsible")  # query words; anchor has all


def generate_realistic_corpus(
    out_dir: str | Path,
    *,
    n_questions: int = 60,
    hops: tuple[int, ...] = (1, 2, 3),
    n_decoys: int = 200,
    seed: int = 11,
) -> list[GoldQuestion]:
    """Write a frozen *decoy-rich* synthetic vault and return its questions.

    Like :func:`generate_corpus` but with knowledge-base-style notes and
    **lexical decoys** that share query words, so a flat/lexical retriever is
    actively misled (not merely starved). Deterministic; answers are unique
    tokens (EM-gradeable, no LLM-judge needed).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    questions: list[GoldQuestion] = []

    for i in range(n_questions):
        k = hops[i % len(hops)]
        topic = f"proj{i:04d}"
        answer = f"CITY{i:04d}"

        anchor = f"record-{i:04d}-anchor"
        relays = [f"record-{i:04d}-link-{m}" for m in range(1, k)]
        ans_note = f"record-{i:04d}-answer"
        chain = [anchor] + relays + [ans_note]

        # anchor: unique topic + ALL phrase words → always the top seed.
        _write_note(
            out, anchor,
            f"Project {topic}. The responsible team lead and city for this "
            f"project are on record. See the owning unit. [[{chain[1]}]]",
        )
        # relays: no phrase word → flat never seeds them.
        for idx, r in enumerate(relays):
            nxt = chain[chain.index(r) + 1]
            _write_note(out, r, f"Owning unit node {idx + 1}. Continue to record. [[{nxt}]]")
        # answer note: no phrase word; holds the unique answer city token.
        _write_note(
            out, ans_note,
            f"Coordinator profile. Stationed permanently at {answer}.",
        )

        # The question shares the unique topic with the anchor and the phrase
        # words with the anchor + decoys (NOT the chain).
        text = (
            f"Which city hosts the responsible team lead for {topic}?"
        )
        questions.append(
            GoldQuestion(
                text=text,
                gold_doc_ids=list(chain),
                hop_only_doc_ids=relays + [ans_note],
                hop=k,
                answer=answer,
            )
        )

    # Decoys: each carries a 2-subset of the phrase words (so it lexically
    # matches the query and ranks above the chain, but below the all-phrase
    # anchor) and never links into a chain.
    for j in range(n_decoys):
        words = rng.sample(_PHRASE, 2)
        _write_note(
            out, f"memo-{j:05d}",
            f"Memo {j:05d}: notes on {words[0]} and {words[1]} matters, "
            f"unrelated to any specific project record.",
        )

    return questions


__all__ = ["GoldQuestion", "generate_corpus", "generate_realistic_corpus"]
