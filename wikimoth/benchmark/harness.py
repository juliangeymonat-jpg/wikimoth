# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""4-arm efficiency harness — *tokens fed to the reader*, the WikiMoth metric.

The thesis: a deterministic wikilink retriever feeds the (paid) reader far
fewer tokens than dumping the whole vault, *without* losing the note-chain the
answer needs. This harness measures exactly that, across four arms:

================  =================================================  ===========
arm               what it feeds the reader                           implemented
================  =================================================  ===========
``dump``          the WHOLE vault (every note) → reader              YES (real)
``agentic``       an LLM iteratively picks notes to read             STUB (later)
``deterministic`` MemoryRAG retrieve, NoOp compaction                YES (real)
``deterministic_  MemoryRAG retrieve + Headroom compaction           YES (real;
 compacted``       (reversible CCR; falls back to NoOp if absent)      degrades)
================  =================================================  ===========

For each (arm × question) it records an :class:`ArmRecord`:
``tokens_fed_to_reader``, ``retrieval_recall_at_k`` (vs a gold note-chain,
doc-level — MOTHRAG's convention), ``answer``, and ``latency_s``.

**No paid API calls by default.** Every arm's reader defaults to the API-free
:class:`EchoReader`, so the harness measures *token volume and retrieval recall*
— the free, deterministic signals — without spending anything. Pass a real
reader (e.g. :class:`ClaudeReader`) only when you explicitly want answers.

Oracle hook
-----------
:func:`oracle_retrieval_loss` runs the reader twice for a question — once on the
**gold** note-chain, once on the **retrieved** note-chain — so you can isolate
how much answer quality is lost to *retrieval* (vs the reader). Like the arms
it defaults to Echo (free); supply a real reader to measure real loss.

The ``agentic`` arm is intentionally a stub: a faithful implementation makes
real LLM tool-calls to let the model browse notes, which costs money and is out
of scope for this code-only skeleton. It raises :class:`NotImplementedError`
with a clear pointer rather than fabricating numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from wikimoth.compaction import HeadroomCompactor, NoOpCompactor
from wikimoth.pipeline import MemoryRAG, _load_vault_chunks, _slugify_note
from wikimoth.reader import EchoReader, Reader
from wikimoth.tokens import count_passage_tokens, token_backend

ARMS = ("dump", "agentic", "deterministic", "deterministic_compacted")


@dataclass
class Question:
    """A benchmark question with its gold note-chain (for recall@k).

    ``hop_only_doc_ids`` is the subset of gold notes that are NOT lexically
    reachable (they share no tokens with the question) and so are reachable only
    by following ``[[wikilinks]]`` — the connect-the-dots notes a flat retriever
    misses. ``hop`` is the chain length. Both are optional; when empty the
    hop-only metric is reported as ``None``.
    """

    text: str
    # doc_ids (== note slug / filename) of the notes that together answer the
    # question. Used to score retrieval recall@k, note-level.
    gold_doc_ids: list[str] = field(default_factory=list)
    hop_only_doc_ids: list[str] = field(default_factory=list)
    hop: int = 0


@dataclass
class ArmRecord:
    """One (arm, question) measurement."""

    arm: str
    question: str
    tokens_fed_to_reader: int
    retrieval_recall_at_k: float | None  # None when no gold supplied
    answer: str
    latency_s: float
    n_passages: int
    hop_only_recall: float | None = None  # recall over the hop-only gold subset
    token_backend: str = ""
    note: str = ""


def _recall_at_k(
    retrieved_chunks: Sequence[Any],
    gold_doc_ids: Sequence[str],
    top_k: int,
) -> float | None:
    """Note-level Recall@K (matches ``mothrag.core.api._attach_recall_at_k``).

    ``R@K = |distinct retrieved notes[:k] ∩ gold notes| / |gold notes|``.
    Identity is compared at *note* granularity (slug), so it is robust to
    chunking (``doc_id`` is now the note slug, and several chunks may share one
    note) and to gold supplied as either a filename, a path, or a slug. Returns
    ``None`` when no gold is supplied (recall undefined).
    """
    if not gold_doc_ids:
        return None
    gold = {_slugify_note(g) for g in gold_doc_ids}
    gold.discard("")
    if not gold:
        return None
    retrieved_slugs: list[str] = []
    seen: set[str] = set()
    for c in retrieved_chunks:
        slug = _slugify_note(getattr(c, "doc_id", "") or "")
        if slug and slug not in seen:
            seen.add(slug)
            retrieved_slugs.append(slug)
    hits = sum(1 for s in retrieved_slugs[:top_k] if s in gold)
    return hits / len(gold)


class FourArmHarness:
    """Run the 4-arm efficiency comparison over a ``[[wikilink]]`` vault.

    Parameters
    ----------
    vault_dir
        Folder of ``.md`` notes (the corpus shared by every arm).
    reader
        Reader used by every arm (kept identical so the comparison is fair).
        Default :class:`EchoReader` → **zero API cost**.
    retriever
        Optional retriever for the deterministic arms (anything satisfying
        MOTHRAG's ``Retriever`` Protocol). Default: MemoryRAG's own default
        ``GraphRetriever(source="wikilinks")``.
    top_k
        Retrieval depth for the deterministic arms and recall@k.
    """

    def __init__(
        self,
        vault_dir: str | Path,
        *,
        reader: Reader | None = None,
        retriever: Any | None = None,
        top_k: int = 8,
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.reader: Reader = reader if reader is not None else EchoReader()
        self.top_k = int(top_k)

        # Load the corpus once; every arm shares it.
        self._chunks: list[Any] = _load_vault_chunks(self.vault_dir)

        # Deterministic arms share one indexed MemoryRAG (NoOp); the compacted
        # arm reuses the same retriever but swaps in the Headroom compactor.
        self._rag = MemoryRAG(
            retriever=retriever, compactor=NoOpCompactor(), reader=self.reader
        )
        self._rag.index_chunks(self._chunks)

    # ------------------------------------------------------------------
    # Individual arms
    # ------------------------------------------------------------------

    def run_dump(self, q: Question) -> ArmRecord:
        """Whole vault → reader. The naive baseline; maximal tokens, recall=1."""
        t0 = time.perf_counter()
        passages = [getattr(c, "text", "") or "" for c in self._chunks]
        tokens = count_passage_tokens(passages)
        answer = self.reader.read(q.text, passages)
        dt = time.perf_counter() - t0
        # Dumping everything trivially contains the gold notes → recall 1.0.
        recall = 1.0 if q.gold_doc_ids else None
        hop_only = 1.0 if q.hop_only_doc_ids else None
        return ArmRecord(
            arm="dump",
            question=q.text,
            tokens_fed_to_reader=tokens,
            retrieval_recall_at_k=recall,
            answer=answer,
            latency_s=dt,
            n_passages=len(passages),
            hop_only_recall=hop_only,
            token_backend=token_backend(),
            note="whole vault fed to reader (baseline)",
        )

    def run_agentic(self, q: Question) -> ArmRecord:
        """STUB — agentic note-selection (LLM browses notes). NOT implemented.

        A faithful arm gives the model a tool to list/open notes and lets it
        iteratively decide which to read, then answers from its selection. That
        is real, paid, multi-call LLM agency — deliberately out of scope for
        this code-only skeleton. Implement against MOTHRAG's reader tool-calling
        surface when an API budget is approved.
        """
        raise NotImplementedError(
            "agentic arm is a stub: it requires real LLM tool-calls to let the "
            "model browse/select notes (paid, multi-call). Out of scope for the "
            "code-only harness. Implement later with an explicit API budget."
        )

    def run_deterministic(self, q: Question) -> ArmRecord:
        """MemoryRAG retrieve (NoOp compaction). The core WikiMoth arm."""
        return self._run_memoryrag(q, compactor=NoOpCompactor(), arm="deterministic")

    def run_deterministic_compacted(self, q: Question) -> ArmRecord:
        """MemoryRAG retrieve + Headroom compaction (falls back to NoOp).

        When ``headroom`` is not installed the compactor degrades to identity,
        so this arm equals ``deterministic`` until headroom is available — the
        record's ``note`` says which path ran.
        """
        compactor = HeadroomCompactor()
        rec = self._run_memoryrag(q, compactor=compactor, arm="deterministic_compacted")
        if not compactor.available:
            rec.note = "headroom absent → NoOp fallback (== deterministic)"
        else:
            rec.note = "headroom CCR compaction active"
        return rec

    def _run_memoryrag(self, q: Question, *, compactor, arm: str) -> ArmRecord:
        """Shared deterministic-arm body: retrieve → compact → read.

        Retrieval goes through :meth:`MemoryRAG.retrieve` (not the raw
        retriever) so edges-only (excluded-content) chunks are dropped here too,
        keeping the deterministic arm consistent with the pipeline.
        """
        t0 = time.perf_counter()
        chunks, _ = self._rag.retrieve(q.text, top_k=self.top_k)
        raw = [getattr(c, "text", "") or "" for c in chunks]
        passages = compactor.compact(raw)
        tokens = count_passage_tokens(passages)
        answer = self.reader.read(q.text, passages)
        dt = time.perf_counter() - t0
        recall = _recall_at_k(chunks, q.gold_doc_ids, self.top_k)
        hop_only = _recall_at_k(chunks, q.hop_only_doc_ids, self.top_k)
        return ArmRecord(
            arm=arm,
            question=q.text,
            tokens_fed_to_reader=tokens,
            retrieval_recall_at_k=recall,
            answer=answer,
            latency_s=dt,
            n_passages=len(passages),
            hop_only_recall=hop_only,
            token_backend=token_backend(),
        )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def retrieval_reproducibility(self, q: Question, *, repeats: int = 5) -> dict:
        """Run deterministic retrieval ``repeats`` times; measure run-to-run drift.

        Returns ``{"distinct_results": n, "deterministic": bool}`` where a value
        of 1 distinct result means the retrieved note-set is bit-stable across
        runs (WikiMoth's reproducibility property; the contrast is the agentic arm,
        which re-decides and drifts). Free — retrieval only, no reader call.
        """
        signatures: set[tuple[str, ...]] = set()
        for _ in range(max(1, repeats)):
            chunks, _ = self._rag.retrieve(q.text, top_k=self.top_k)
            sig = tuple(
                _slugify_note(getattr(c, "doc_id", "") or "") for c in chunks
            )
            signatures.add(sig)
        return {"distinct_results": len(signatures), "deterministic": len(signatures) == 1}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_arm(self, arm: str, q: Question) -> ArmRecord:
        dispatch: dict[str, Callable[[Question], ArmRecord]] = {
            "dump": self.run_dump,
            "agentic": self.run_agentic,
            "deterministic": self.run_deterministic,
            "deterministic_compacted": self.run_deterministic_compacted,
        }
        if arm not in dispatch:
            raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
        return dispatch[arm](q)

    def run(
        self,
        questions: Sequence[Question],
        *,
        arms: Sequence[str] = ("dump", "deterministic", "deterministic_compacted"),
        skip_agentic: bool = True,
    ) -> list[ArmRecord]:
        """Run ``arms`` over ``questions``. ``agentic`` is skipped by default.

        Returns a flat list of :class:`ArmRecord`. The ``agentic`` arm, if
        requested, raises unless ``skip_agentic`` is left ``True`` (in which
        case it is silently dropped from the arm set with a note record).
        """
        records: list[ArmRecord] = []
        for arm in arms:
            if arm == "agentic" and skip_agentic:
                records.append(
                    ArmRecord(
                        arm="agentic",
                        question="(all)",
                        tokens_fed_to_reader=0,
                        retrieval_recall_at_k=None,
                        answer="",
                        latency_s=0.0,
                        n_passages=0,
                        token_backend=token_backend(),
                        note="STUB — skipped (NotImplementedError); needs LLM tool-calls",
                    )
                )
                continue
            for q in questions:
                records.append(self.run_arm(arm, q))
        return records


def oracle_retrieval_loss(
    vault_dir: str | Path,
    question: Question,
    *,
    reader: Reader | None = None,
    retriever: Any | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Retrieval-loss oracle: reader on GOLD notes vs reader on RETRIEVED notes.

    Reads the same question twice with the same reader — once over the gold
    note-chain (the ceiling: perfect retrieval) and once over what the
    deterministic retriever actually pulled — so the difference attributable to
    *retrieval* (not the reader) is isolated.

    Returns ``answer_gold``, ``answer_retrieved``, ``tokens_gold``,
    ``tokens_retrieved``, and ``recall_at_k``. Defaults to :class:`EchoReader`
    → **zero API cost**; pass a real reader to measure real answer divergence.
    """
    reader = reader if reader is not None else EchoReader()
    chunks = _load_vault_chunks(vault_dir)

    # Collect *all* chunks of each gold note (notes are chunked now, so a note's
    # text is spread over several chunks sharing one slug doc_id).
    gold_slugs = {_slugify_note(g) for g in question.gold_doc_ids}
    gold_slugs.discard("")
    gold_chunks = [
        c for c in chunks
        if _slugify_note(getattr(c, "doc_id", "") or "") in gold_slugs
    ]
    gold_passages = [getattr(c, "text", "") or "" for c in gold_chunks]

    rag = MemoryRAG(retriever=retriever, compactor=NoOpCompactor(), reader=reader)
    rag.index_chunks(chunks)
    retrieved_chunks, _ = rag.retrieve(question.text, top_k=top_k)
    retrieved_passages = [getattr(c, "text", "") or "" for c in retrieved_chunks]

    return {
        "question": question.text,
        "answer_gold": reader.read(question.text, gold_passages),
        "answer_retrieved": reader.read(question.text, retrieved_passages),
        "tokens_gold": count_passage_tokens(gold_passages),
        "tokens_retrieved": count_passage_tokens(retrieved_passages),
        "recall_at_k": _recall_at_k(retrieved_chunks, question.gold_doc_ids, top_k),
        "token_backend": token_backend(),
    }


def _mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(records: Sequence[ArmRecord]) -> dict[str, dict[str, Any]]:
    """Aggregate per-arm means from a flat list of :class:`ArmRecord`.

    Returns ``{arm: {n, mean_tokens, mean_recall_at_k, mean_hop_only_recall,
    mean_latency_s}}``. Stub/skipped arm rows (``n_passages == 0`` with a note)
    are excluded from the per-question means.
    """
    by_arm: dict[str, list[ArmRecord]] = {}
    for r in records:
        if r.question == "(all)":  # skipped-arm marker row
            continue
        by_arm.setdefault(r.arm, []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for arm, rs in by_arm.items():
        out[arm] = {
            "n": len(rs),
            "mean_tokens": _mean([r.tokens_fed_to_reader for r in rs]),
            "mean_recall_at_k": _mean([r.retrieval_recall_at_k for r in rs]),
            "mean_hop_only_recall": _mean([r.hop_only_recall for r in rs]),
            "mean_latency_s": _mean([r.latency_s for r in rs]),
        }
    return out


__all__ = [
    "ARMS",
    "Question",
    "ArmRecord",
    "FourArmHarness",
    "oracle_retrieval_loss",
    "summarize",
]
