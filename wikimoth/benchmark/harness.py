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
``agentic``       an LLM browses/prunes its own context              YES (needs model)
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

The ``agentic`` arm (*let the model prune its own context*) is real but needs a
browsing model: pass ``agentic_model=AnthropicAgenticModel(...)`` to spend on real
Claude tool-calls, or a scripted :class:`~wikimoth.benchmark.agentic.AgenticModel`
to drive it offline. With no model it raises a clear :class:`NotImplementedError`
rather than fabricating numbers. See :mod:`wikimoth.benchmark.agentic`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from wikimoth.benchmark.agentic import (
    AgenticModel,
    agentic_browse,
    load_notes_from_vault,
)
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
    # agentic arm only: real billed API tokens (0 for the free arms)
    api_input_tokens: int = 0
    api_output_tokens: int = 0


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


def _recall_from_slugs(
    read_slugs: Sequence[str], gold_doc_ids: Sequence[str]
) -> float | None:
    """Recall of gold notes over a *set* of read notes (no top-k cap).

    The agentic arm has no ranked top-k — it reads whatever it chooses — so its
    recall is the fraction of gold notes that appear anywhere in the read set.
    Returns ``None`` when no gold is supplied.
    """
    if not gold_doc_ids:
        return None
    gold = {_slugify_note(g) for g in gold_doc_ids}
    gold.discard("")
    if not gold:
        return None
    readset = {_slugify_note(s) for s in read_slugs}
    return sum(1 for g in gold if g in readset) / len(gold)


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
        agentic_model: AgenticModel | None = None,
        agentic_max_steps: int = 12,
        agentic_search_limit: int = 10,
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.reader: Reader = reader if reader is not None else EchoReader()
        self.top_k = int(top_k)

        # Agentic arm config (the model is optional; only the agentic arm uses it).
        self.agentic_model = agentic_model
        self.agentic_max_steps = int(agentic_max_steps)
        self.agentic_search_limit = int(agentic_search_limit)
        self._agentic_notes_cache: list[Any] | None = None

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

    def _agentic_notes(self) -> list[Any]:
        """Whole-note view of the vault for the agentic arm (loaded once)."""
        if self._agentic_notes_cache is None:
            self._agentic_notes_cache = load_notes_from_vault(self.vault_dir)
        return self._agentic_notes_cache

    def run_agentic(self, q: Question) -> ArmRecord:
        """Let the model browse the vault and prune its own context, then answer.

        Drives :func:`~wikimoth.benchmark.agentic.agentic_browse` with this
        harness's ``agentic_model``. ``tokens_fed_to_reader`` is the tokens of note
        bodies the agent pulled in (its self-curated context); the billed
        ``api_input``/``api_output`` and step count land in ``note``. Recall is over
        the *set* of notes the agent opened (it has no ranked top-k).

        Raises :class:`NotImplementedError` if no ``agentic_model`` was supplied —
        the arm is real, but it needs a browsing model (paid Claude tool-calls, or
        an offline scripted policy).
        """
        if self.agentic_model is None:
            raise NotImplementedError(
                "agentic arm needs a browsing model. Pass "
                "agentic_model=AnthropicAgenticModel(...) (pip install "
                "'wikimoth[claude]' + ANTHROPIC_API_KEY) to run it against real "
                "Claude tool-calls, or a scripted AgenticModel to drive it offline."
            )
        t0 = time.perf_counter()
        res = agentic_browse(
            q.text,
            self._agentic_notes(),
            self.agentic_model,
            max_steps=self.agentic_max_steps,
            search_limit=self.agentic_search_limit,
        )
        dt = time.perf_counter() - t0
        recall = _recall_from_slugs(res.notes_read, q.gold_doc_ids)
        hop_only = _recall_from_slugs(res.notes_read, q.hop_only_doc_ids)
        return ArmRecord(
            arm="agentic",
            question=q.text,
            tokens_fed_to_reader=res.content_tokens,
            retrieval_recall_at_k=recall,
            answer=res.answer,
            latency_s=dt,
            n_passages=len(res.notes_read),
            hop_only_recall=hop_only,
            token_backend=token_backend(),
            note=(
                f"agentic browse: {res.steps} steps, "
                f"api_input={res.api_input_tokens}, api_output={res.api_output_tokens}"
            ),
            api_input_tokens=res.api_input_tokens,
            api_output_tokens=res.api_output_tokens,
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

    def agentic_reproducibility(self, q: Question, *, repeats: int = 3) -> dict:
        """Run the agentic browse ``repeats`` times; measure how the read-set drifts.

        The contrast to :meth:`retrieval_reproducibility`: a real LLM re-decides
        each run, so its self-curated note-set typically varies (``distinct_results
        > 1``), while deterministic retrieval is bit-stable (``== 1``). Compares the
        *set* of notes read (order-independent). **Paid** — each repeat is a full
        browse. Requires an ``agentic_model``.
        """
        if self.agentic_model is None:
            raise NotImplementedError("agentic_reproducibility needs an agentic_model.")
        notes = self._agentic_notes()
        signatures: set[frozenset[str]] = set()
        for _ in range(max(1, repeats)):
            res = agentic_browse(
                q.text,
                notes,
                self.agentic_model,
                max_steps=self.agentic_max_steps,
                search_limit=self.agentic_search_limit,
            )
            signatures.add(frozenset(res.notes_read))
        return {
            "distinct_results": len(signatures),
            "deterministic": len(signatures) == 1,
            "runs": max(1, repeats),
        }

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
        """Run ``arms`` over ``questions``.

        Returns a flat list of :class:`ArmRecord`. The ``agentic`` arm runs per
        question only when an ``agentic_model`` is configured *and* ``skip_agentic``
        is ``False`` (it makes paid LLM calls); otherwise it is recorded as a single
        skipped marker row explaining why.
        """
        records: list[ArmRecord] = []
        for arm in arms:
            if arm == "agentic" and (skip_agentic or self.agentic_model is None):
                why = (
                    "skipped (skip_agentic=True)"
                    if self.agentic_model is not None
                    else "skipped (no agentic_model supplied)"
                )
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
                        note=why,
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
