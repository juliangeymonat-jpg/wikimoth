# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the synthetic gold-corpus generator + the new harness metrics.

Covers ``wikimoth.benchmark.corpus.generate_corpus`` (determinism, gold/hop
structure, the vocabulary-disjointness property the benchmark relies on) and the
harness additions (``hop_only_recall``, ``retrieval_reproducibility``,
``summarize``).

Everything is API-free: the generator is pure file I/O + a seeded PRNG, and the
end-to-end retrieval checks use the default :class:`EchoReader` (zero cost).
WikiMoth is self-contained, so just put it on the path.

Run:
    $env:PYTHONPATH = "."
    python -m pytest ./tests/test_benchmark.py -q
"""

from __future__ import annotations

import re

import pytest

from wikimoth.benchmark.corpus import GoldQuestion, generate_corpus
from wikimoth.benchmark.harness import ArmRecord, Question, summarize

# Lowercase word tokens, matching the benchmark's lexical-overlap reasoning.
_WORD = re.compile(r"\w+")
# The unique per-question topic token (the ONLY query word planted in a note).
_TOPIC = re.compile(r"topic\d{4}")


def _read_note(vault, stem: str) -> str:
    """Full text of a note file (frontmatter + body) by stem."""
    return (vault / f"{stem}.md").read_text(encoding="utf-8")


def _note_body(vault, stem: str) -> str:
    """Body of a note (everything after the closing frontmatter ``---``)."""
    text = _read_note(vault, stem)
    parts = text.split("---", 2)
    return parts[2] if len(parts) == 3 else text


def _query_content_tokens(text: str) -> set[str]:
    """Lowercase content tokens of a query, minus the unique ``topicNNNN``."""
    return {t for t in _WORD.findall(text.lower()) if not _TOPIC.fullmatch(t)}


# ---------------------------------------------------------------------------
# 1. Determinism of generation
# ---------------------------------------------------------------------------

def test_generation_is_deterministic_questions_and_files(tmp_path):
    """Same seed/params → identical question objects AND identical note files."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    qs_a = generate_corpus(a, n_questions=8, hops=(1, 2, 3), n_distractors=20, seed=7)
    qs_b = generate_corpus(b, n_questions=8, hops=(1, 2, 3), n_distractors=20, seed=7)

    assert len(qs_a) == len(qs_b) == 8
    for qa, qb in zip(qs_a, qs_b):
        assert qa.text == qb.text
        assert qa.gold_doc_ids == qb.gold_doc_ids
        assert qa.hop_only_doc_ids == qb.hop_only_doc_ids
        assert qa.hop == qb.hop
        assert qa.answer == qb.answer

    # Identical set of note files, byte-for-byte identical contents.
    files_a = sorted(p.name for p in a.glob("*.md"))
    files_b = sorted(p.name for p in b.glob("*.md"))
    assert files_a == files_b
    assert files_a, "generator must write at least one note"
    for name in files_a:
        assert (a / name).read_text(encoding="utf-8") == (
            b / name
        ).read_text(encoding="utf-8"), f"note {name} differs between identical runs"


def test_question_count_param_changes_count(tmp_path):
    qs_small = generate_corpus(tmp_path / "s", n_questions=5, n_distractors=10, seed=7)
    qs_big = generate_corpus(tmp_path / "g", n_questions=13, n_distractors=10, seed=7)
    assert len(qs_small) == 5
    assert len(qs_big) == 13
    assert len(qs_small) != len(qs_big)


# ---------------------------------------------------------------------------
# 2. Gold / hop structure
# ---------------------------------------------------------------------------

def test_gold_and_hop_structure(tmp_path):
    hops = (1, 2, 3)
    n_distractors = 15
    qs = generate_corpus(
        tmp_path, n_questions=12, hops=hops, n_distractors=n_distractors, seed=7
    )

    for i, q in enumerate(qs):
        assert isinstance(q, GoldQuestion)
        # hop label is one of the requested hop lengths
        assert q.hop in hops
        # gold = anchor + (hop-1 relays) + endpoint  ==  hop + 1 notes
        assert len(q.gold_doc_ids) == q.hop + 1
        # hop-only = gold minus the anchor (the first element)
        assert q.gold_doc_ids[0] == f"dossier-{i:04d}-anchor"
        assert q.hop_only_doc_ids == q.gold_doc_ids[1:]
        # anchor stem + endpoint stem naming
        anchor = q.gold_doc_ids[0]
        endpoint = q.gold_doc_ids[-1]
        assert anchor == f"dossier-{i:04d}-anchor"
        assert endpoint == f"dossier-{i:04d}-endpoint"
        # the anchor + endpoint notes exist on disk
        assert (tmp_path / f"{anchor}.md").exists()
        assert (tmp_path / f"{endpoint}.md").exists()

    # exactly n_distractors filing-* notes exist on disk
    filings = sorted(tmp_path.glob("filing-*.md"))
    assert len(filings) == n_distractors


# ---------------------------------------------------------------------------
# 3. Vocabulary disjointness (the property the benchmark relies on)
# ---------------------------------------------------------------------------

def test_relays_endpoints_filings_share_no_query_content_tokens(tmp_path):
    """No query content token (minus topicNNNN) appears in any relay/endpoint/
    filing note body — the guarantee that the hop-only notes are reachable
    ONLY by following [[wikilinks]], never lexically."""
    qs = generate_corpus(
        tmp_path, n_questions=12, hops=(1, 2, 3), n_distractors=30, seed=7
    )

    # Union of every query's content tokens.
    all_query_tokens: set[str] = set()
    for q in qs:
        all_query_tokens |= _query_content_tokens(q.text)
    assert all_query_tokens, "queries must carry content tokens"

    # Every non-anchor note (relay + endpoint + filing).
    non_anchor = [
        p for p in tmp_path.glob("*.md") if not p.stem.endswith("-anchor")
    ]
    assert non_anchor, "expected relay/endpoint/filing notes on disk"
    for p in non_anchor:
        body = _note_body(tmp_path, p.stem)
        body_tokens = set(_WORD.findall(body.lower()))
        overlap = all_query_tokens & body_tokens
        assert not overlap, (
            f"note {p.name} leaks query content tokens {overlap}; the hop-only "
            "notes must be lexically invisible to the query"
        )


def test_topic_token_is_unique_to_its_anchor(tmp_path):
    """Each unique topicNNNN appears in EXACTLY one note across the vault: its
    own anchor (single-seed property)."""
    qs = generate_corpus(
        tmp_path, n_questions=12, hops=(1, 2, 3), n_distractors=30, seed=7
    )

    # Map every topic token to the set of note stems whose FULL text contains it.
    all_notes = list(tmp_path.glob("*.md"))
    for i, q in enumerate(qs):
        topic = f"topic{i:04d}"
        carriers = [
            p.stem
            for p in all_notes
            if topic in set(_WORD.findall(p.read_text(encoding="utf-8").lower()))
        ]
        assert carriers == [f"dossier-{i:04d}-anchor"], (
            f"{topic} must appear ONLY in its anchor, found in {carriers}"
        )


# ---------------------------------------------------------------------------
# Helpers for the graph-dependent (e2e) tests
# ---------------------------------------------------------------------------

def _harness_for(vault, hops):
    """Build a FourArmHarness over ``vault`` with a wikilink GraphRetriever."""
    from wikimoth.retrieval import GraphRetriever

    from wikimoth.benchmark.harness import FourArmHarness

    retriever = GraphRetriever(source="wikilinks", max_hops=max(hops))
    return FourArmHarness(vault, retriever=retriever, top_k=8)


def _questions_from(gold):
    return [
        Question(
            text=g.text,
            gold_doc_ids=g.gold_doc_ids,
            hop_only_doc_ids=g.hop_only_doc_ids,
            hop=g.hop,
        )
        for g in gold
    ]


# ---------------------------------------------------------------------------
# 4. Retrieval property end-to-end (graph reaches the hop-only chain)
# ---------------------------------------------------------------------------

def test_retrieval_property_end_to_end(tmp_path):
    hops = (1, 2, 3)
    gold = generate_corpus(
        tmp_path, n_questions=12, hops=hops, n_distractors=40, seed=7
    )
    questions = _questions_from(gold)
    harness = _harness_for(tmp_path, hops)

    records = harness.run(questions, arms=("dump", "deterministic"))
    summ = summarize(records)

    det = summ["deterministic"]
    dump = summ["dump"]

    # The graph reaches the full gold chain AND the hop-only subset for free.
    assert det["mean_recall_at_k"] == 1.0
    assert det["mean_hop_only_recall"] == 1.0
    # Dump trivially contains everything → recall 1.0.
    assert dump["mean_recall_at_k"] == 1.0
    # Massive token reduction: deterministic feeds < 5% of the dump tokens.
    assert dump["mean_tokens"] > 0
    assert det["mean_tokens"] < 0.05 * dump["mean_tokens"], (
        f"deterministic mean_tokens={det['mean_tokens']} not < 5% of "
        f"dump mean_tokens={dump['mean_tokens']}"
    )


# ---------------------------------------------------------------------------
# 5. Reproducibility metric
# ---------------------------------------------------------------------------

def test_retrieval_reproducibility_is_deterministic(tmp_path):
    hops = (1, 2, 3)
    gold = generate_corpus(
        tmp_path, n_questions=6, hops=hops, n_distractors=20, seed=7
    )
    questions = _questions_from(gold)
    harness = _harness_for(tmp_path, hops)

    repro = harness.retrieval_reproducibility(questions[0], repeats=5)
    assert repro["deterministic"] is True
    assert repro["distinct_results"] == 1


# ---------------------------------------------------------------------------
# 6. summarize: correct means + skipped-arm marker excluded
# ---------------------------------------------------------------------------

def _rec(arm, q, tokens, recall, hop_only, latency, n_passages=2):
    return ArmRecord(
        arm=arm,
        question=q,
        tokens_fed_to_reader=tokens,
        retrieval_recall_at_k=recall,
        answer="x",
        latency_s=latency,
        n_passages=n_passages,
        hop_only_recall=hop_only,
    )


def test_summarize_computes_means_and_excludes_skipped_marker():
    records = [
        _rec("deterministic", "q1", 100, 1.0, 1.0, 0.10),
        _rec("deterministic", "q2", 200, 0.0, 0.0, 0.30),
        _rec("dump", "q1", 1000, 1.0, 1.0, 1.00),
        _rec("dump", "q2", 3000, 1.0, 1.0, 3.00),
        # skipped-arm marker row — must be excluded from every aggregate.
        ArmRecord(
            arm="agentic",
            question="(all)",
            tokens_fed_to_reader=999999,
            retrieval_recall_at_k=None,
            answer="",
            latency_s=0.0,
            n_passages=0,
            note="STUB — skipped",
        ),
    ]
    summ = summarize(records)

    # The skipped marker row is dropped entirely → no "agentic" key.
    assert "agentic" not in summ

    det = summ["deterministic"]
    assert det["n"] == 2
    assert det["mean_tokens"] == 150.0
    assert det["mean_recall_at_k"] == 0.5
    assert det["mean_hop_only_recall"] == 0.5
    assert det["mean_latency_s"] == pytest.approx(0.20)

    dump = summ["dump"]
    assert dump["n"] == 2
    assert dump["mean_tokens"] == 2000.0
    assert dump["mean_recall_at_k"] == 1.0
    assert dump["mean_latency_s"] == pytest.approx(2.0)


def test_summarize_ignores_none_metrics_in_means():
    """A None recall contributes nothing to the mean (not counted as 0)."""
    records = [
        _rec("deterministic", "q1", 100, 1.0, None, 0.10),
        _rec("deterministic", "q2", 200, None, None, 0.30),
    ]
    summ = summarize(records)["deterministic"]
    assert summ["n"] == 2
    assert summ["mean_recall_at_k"] == 1.0  # only the one non-None value
    assert summ["mean_hop_only_recall"] is None  # all None → None


# ---------------------------------------------------------------------------
# 7. ArmRecord.hop_only_recall on the dump arm
# ---------------------------------------------------------------------------

def test_dump_hop_only_recall_present_and_absent(tmp_path):
    hops = (1, 2, 3)
    gold = generate_corpus(
        tmp_path, n_questions=6, hops=hops, n_distractors=20, seed=7
    )
    questions = _questions_from(gold)
    harness = _harness_for(tmp_path, hops)

    # Pick a question that actually has hop-only gold notes (hop >= 1).
    q_with = next(q for q in questions if q.hop_only_doc_ids)
    rec = harness.run_dump(q_with)
    assert rec.hop_only_recall == 1.0  # dump trivially contains the hop-only set

    # Same question, but hop-only stripped → metric is undefined → None.
    q_without = Question(
        text=q_with.text, gold_doc_ids=q_with.gold_doc_ids, hop_only_doc_ids=[]
    )
    rec_none = harness.run_dump(q_without)
    assert rec_none.hop_only_recall is None
