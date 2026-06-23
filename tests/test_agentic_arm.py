# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the agentic arm — *let the model prune its own context*.

The real Claude-backed model is exercised by ``scripts/run_agentic_benchmark.py``
(paid); here we drive the arm with deterministic *scripted* policies so the suite
stays fully offline (the repo invariant: tests never touch the network). The
scripted models are faithful to the contract the loop drives, so they cover:

* a link-follower that walks the [[wikilink]] chain → recall 1.0, hop-only 1.0;
* a naive model that reads only the top lexical hit → multi-hop recall 0.0 (proof
  the metric is real, not rigged to pass);
* token accounting (distinct notes counted once) and determinism detection.
"""

from __future__ import annotations

import re

import pytest

from wikimoth.benchmark import FourArmHarness, summarize
from wikimoth.benchmark.agentic import (
    Answer,
    NoteView,
    Read,
    Search,
    agentic_browse,
    load_notes_from_vault,
)
from wikimoth.benchmark.agentic import _NoteIndex  # type: ignore[attr-defined]
from wikimoth.benchmark.corpus import generate_realistic_corpus
from wikimoth.benchmark.harness import Question
from wikimoth.retrieval import GraphRetriever

_TOPIC_RE = re.compile(r"(topic\d+|proj\d+)")
_ANSWER_RE = re.compile(r"(ANSWERTOKEN\d+|CITY\d+)")
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _first_listed(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            return line[2:].strip()
    return ""


class LinkFollower:
    """Seed on the unique topic token, then follow the first [[link]] to the end.

    This is the *competent* agent: it reaches the answer note on any hop depth, so
    it should score recall 1.0 / hop-only 1.0. Deterministic (no randomness).
    """

    browse_input_tokens = 0
    browse_output_tokens = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.browse_input_tokens = 0
        self.browse_output_tokens = 0

    def act(self, question, observation):
        if observation is None:
            m = _TOPIC_RE.search(question)
            return Search(m.group(1) if m else question)
        if observation.tool == "search_notes":
            name = _first_listed(observation.text)
            return Read(name) if name else Answer("no hits")
        if observation.tool == "read_note":
            ans = _ANSWER_RE.search(observation.text)
            if ans:
                return Answer(f"The answer is {ans.group(1)}.")
            link = _LINK_RE.search(observation.text)
            if link:
                return Read(link.group(1))
            return Answer("dead end")
        return Answer("error")


class TopHitOnly:
    """Reads only the single top lexical hit, never follows links, then gives up.

    The *incompetent* agent: it cannot reach link-only notes, so its hop-only recall
    must be 0.0. Proves the arm can fail (the metric is not rigged).
    """

    browse_input_tokens = 0
    browse_output_tokens = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.browse_input_tokens = 0
        self.browse_output_tokens = 0

    def act(self, question, observation):
        if observation is None:
            # search the phrase words (the anchor + decoys), not the unique topic
            return Search("city team lead responsible")
        if observation.tool == "search_notes":
            return Read(_first_listed(observation.text))
        return Answer("I don't know")


class DriftModel:
    """Like LinkFollower but reads a different decoy first on each browse.

    The instance counter advances every ``reset()``, so successive browses curate
    different note-sets → the determinism check must report ``distinct_results > 1``.
    """

    browse_input_tokens = 0
    browse_output_tokens = 0

    def __init__(self) -> None:
        self._browse = -1
        self.reset()

    def reset(self) -> None:
        self._browse += 1
        self._read_decoy = False
        self.browse_input_tokens = 0
        self.browse_output_tokens = 0

    def act(self, question, observation):
        if observation is None:
            return Search("memo")  # list decoys
        if observation.tool == "search_notes" and not self._read_decoy:
            self._read_decoy = True
            names = [
                ln.strip()[2:].strip()
                for ln in observation.text.splitlines()
                if ln.strip().startswith("- ")
            ]
            if names:
                return Read(names[self._browse % len(names)])
            return Answer("no decoys")
        # then behave like the link follower from a fresh topic search
        return Answer("done (read one decoy)")


# ---------------------------------------------------------------------------
# Note loading + index
# ---------------------------------------------------------------------------
def test_load_notes_and_index(tmp_path):
    vault = tmp_path / "v"
    generate_realistic_corpus(vault, n_questions=6, hops=(1, 2, 3), n_decoys=20, seed=11)
    notes = load_notes_from_vault(vault)
    assert len(notes) == len(list(vault.glob("*.md")))
    idx = _NoteIndex(notes)
    # resolve by filename, by stem, and by slug
    assert idx.get("record-0000-anchor.md") is not None
    assert idx.get("record-0000-anchor") is not None
    assert idx.get("record-0000-anchor").slug == idx.get("record-0000-anchor.md").slug
    # the all-phrase anchor outranks the 2-phrase decoys for a phrase query
    hits = idx.search("city team lead responsible proj0000", limit=5)
    assert hits, "expected at least one match"
    assert hits[0] == "record-0000-anchor.md"


def test_exclude_filters_notes(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "MEMORY.md").write_text("# hub\n[[a]]\n", encoding="utf-8")
    (vault / "a.md").write_text("content\n", encoding="utf-8")
    notes = load_notes_from_vault(vault, exclude=("MEMORY.md",))
    assert [n.filename for n in notes] == ["a.md"]


# ---------------------------------------------------------------------------
# agentic_browse — token accounting + behaviour
# ---------------------------------------------------------------------------
def test_browse_counts_distinct_note_once():
    """A note read twice is counted once in content_tokens; read-set is distinct."""

    class ReadTwice:
        browse_input_tokens = 0
        browse_output_tokens = 0

        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def act(self, q, obs):
            self.calls += 1
            if self.calls == 1:
                return Read("only.md")
            if self.calls == 2:
                return Read("only.md")  # same note again
            return Answer("done")

    notes = [NoteView(slug="only", filename="only.md", text="hello world " * 50)]
    res = agentic_browse("q", notes, ReadTwice(), max_steps=5)
    assert res.notes_read == ["only"]  # distinct
    assert res.content_tokens > 0
    # counted once: re-reading does not double the token tally
    from wikimoth.tokens import count_tokens

    assert res.content_tokens == count_tokens(notes[0].text)


def test_browse_step_budget_exhaustion():
    class NeverAnswers:
        browse_input_tokens = 0
        browse_output_tokens = 0

        def reset(self):
            pass

        def act(self, q, obs):
            return Search("x")

    res = agentic_browse("q", [], NeverAnswers(), max_steps=3)
    assert res.steps == 3
    assert "step budget" in res.answer


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------
def _questions(vault, **kw):
    gold = generate_realistic_corpus(vault, seed=11, **kw)
    return gold, [
        Question(
            text=g.text,
            gold_doc_ids=g.gold_doc_ids,
            hop_only_doc_ids=g.hop_only_doc_ids,
            hop=g.hop,
        )
        for g in gold
    ]


def test_agentic_arm_competent_reaches_chain(tmp_path):
    vault = tmp_path / "v"
    _, questions = _questions(vault, n_questions=9, hops=(1, 2, 3), n_decoys=40)
    h = FourArmHarness(
        vault,
        retriever=GraphRetriever(source="wikilinks", max_hops=3),
        agentic_model=LinkFollower(),
        agentic_max_steps=12,
    )
    recs = h.run(questions, arms=("agentic",), skip_agentic=False)
    assert len(recs) == len(questions)
    for r, q in zip(recs, questions):
        assert r.arm == "agentic"
        assert r.tokens_fed_to_reader > 0
        assert r.note.startswith("agentic browse")
        # competent agent reaches the answer token
        assert re.search(r"CITY\d+", r.answer)
    s = summarize(recs)["agentic"]
    assert s["mean_recall_at_k"] == 1.0
    assert s["mean_hop_only_recall"] == 1.0


def test_agentic_arm_naive_fails_multihop(tmp_path):
    vault = tmp_path / "v"
    _, questions = _questions(vault, n_questions=9, hops=(2, 3), n_decoys=40)
    h = FourArmHarness(
        vault,
        retriever=GraphRetriever(source="wikilinks", max_hops=3),
        agentic_model=TopHitOnly(),
    )
    recs = h.run(questions, arms=("agentic",), skip_agentic=False)
    s = summarize(recs)["agentic"]
    # never follows a link → cannot reach the link-only notes
    assert s["mean_hop_only_recall"] == 0.0
    assert s["mean_recall_at_k"] < 1.0


def test_agentic_arm_requires_model(tmp_path):
    vault = tmp_path / "v"
    _, questions = _questions(vault, n_questions=3)
    h = FourArmHarness(vault)  # no agentic_model
    with pytest.raises(NotImplementedError, match="browsing model"):
        h.run_arm("agentic", questions[0])


def test_run_marks_agentic_skipped(tmp_path):
    vault = tmp_path / "v"
    _, questions = _questions(vault, n_questions=3)
    # no model → marker explains the absence
    h0 = FourArmHarness(vault)
    rec0 = h0.run(questions, arms=("agentic",), skip_agentic=False)
    assert len(rec0) == 1 and "no agentic_model" in rec0[0].note
    # model present but skip flag on → marker explains the skip
    h1 = FourArmHarness(vault, agentic_model=LinkFollower())
    rec1 = h1.run(questions, arms=("agentic",), skip_agentic=True)
    assert len(rec1) == 1 and "skip_agentic=True" in rec1[0].note


def test_agentic_reproducibility_detects_drift(tmp_path):
    vault = tmp_path / "v"
    _, questions = _questions(vault, n_questions=3, hops=(2,), n_decoys=30)
    # deterministic policy → one distinct read-set
    h_det = FourArmHarness(
        vault,
        retriever=GraphRetriever(source="wikilinks", max_hops=3),
        agentic_model=LinkFollower(),
    )
    rep = h_det.agentic_reproducibility(questions[0], repeats=3)
    assert rep["deterministic"] is True and rep["distinct_results"] == 1
    # drifting policy → multiple distinct read-sets (the real-LLM contrast)
    h_drift = FourArmHarness(vault, agentic_model=DriftModel())
    rep2 = h_drift.agentic_reproducibility(questions[0], repeats=3)
    assert rep2["deterministic"] is False and rep2["distinct_results"] > 1
