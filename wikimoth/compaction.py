# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Compaction stage — shrink retrieved passages *before* the reader.

A :class:`Compactor` sits between MOTHRAG's retriever and the Claude reader in
the WikiMoth pipeline. Its job is to reduce the token volume fed to the (paid)
reader without dropping the information the answer depends on. It is a pure
pass-through by default (:class:`NoOpCompactor`) so the pipeline stays
deterministic and API-free unless a real compressor is explicitly wired in.

Compactors are *deterministic* and *side-effect free* with respect to the
retrieval result: same passages in → same passages out. The optional
:class:`HeadroomCompactor` integrates `chopratejas/headroom` (Apache-2.0): a
reversible Context Compression Ratio (CCR) compressor exposed as both an MCP
server and a library, designed to compress RAG chunks before the LLM with the
*originals recoverable*. Headroom is **not** a hard dependency — the import is
guarded so WikiMoth installs and tests with zero extra packages; if headroom is
absent the compactor logs once and falls back to the no-op identity.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

logger = logging.getLogger("wikimoth.compaction")


@runtime_checkable
class Compactor(Protocol):
    """Passages in → (fewer / shorter) passages out.

    The contract is intentionally minimal and list-preserving: a compactor
    takes the list of retrieved passage strings and returns a list of passage
    strings (possibly the same objects, possibly compressed, possibly fewer).
    Returning a ``list[str]`` (rather than one blob) keeps the per-passage
    structure the reader expects and lets the harness count tokens the same
    way across arms.
    """

    def compact(self, passages: Sequence[str]) -> list[str]:
        """Return a token-reduced view of ``passages`` (order preserved)."""
        ...


class NoOpCompactor:
    """Identity compactor — the default. Deterministic, no deps, no network.

    Returns the passages unchanged (as a fresh ``list``). This is the baseline
    "deterministic, retriever-only" arm: all the token savings come from
    retrieval picking the right note-chain rather than from compression.
    """

    name = "noop"

    def compact(self, passages: Sequence[str]) -> list[str]:
        return list(passages)


class HeadroomCompactor:
    """Reversible CCR compaction via `chopratejas/headroom` (import-guarded).

    Headroom (https://github.com/chopratejas/headroom, Apache-2.0) is a
    reversible context-compression layer — exposed as an MCP server and a
    Python library — that compresses RAG chunks before they reach the LLM
    while keeping the originals recoverable (its "Context Compression Ratio").
    WikiMoth wraps it as an optional compaction stage.

    Headroom is **optional**: it is imported lazily inside :meth:`__init__`.
    When it is not installed (the default for WikiMoth's test/dogfood path) the
    instance degrades gracefully to the :class:`NoOpCompactor` identity and
    logs the fallback exactly once. This keeps ``import wikimoth`` and the whole
    test suite free of any third-party requirement.

    Parameters
    ----------
    ratio
        Target compression ratio hint passed through to headroom when present
        (semantics owned by headroom; WikiMoth does not interpret it). Ignored
        in the fallback path.
    strict
        If ``True``, raise :class:`ImportError` when headroom is missing
        instead of falling back. Useful in a benchmark arm that must *fail
        loudly* rather than silently measure the no-op baseline. Default
        ``False`` (graceful fallback).
    """

    name = "headroom"

    def __init__(self, *, ratio: float = 0.5, strict: bool = False) -> None:
        self.ratio = float(ratio)
        self._fallback = NoOpCompactor()
        self._backend = None  # set if headroom imports successfully
        self.available = False
        try:
            import headroom  # type: ignore  # noqa: F401

            self._backend = headroom
            self.available = True
            logger.info("HeadroomCompactor: headroom backend active.")
        except ImportError as e:
            if strict:
                raise ImportError(
                    "HeadroomCompactor(strict=True) requires the optional "
                    "`headroom` package (chopratejas/headroom). Install it or "
                    "use NoOpCompactor."
                ) from e
            logger.warning(
                "HeadroomCompactor: `headroom` not installed — falling back to "
                "NoOpCompactor (identity). Install `chopratejas/headroom` to "
                "enable reversible CCR compression. This message logs once."
            )

    def compact(self, passages: Sequence[str]) -> list[str]:
        """Compress each passage via headroom, or identity if unavailable."""
        if not self.available or self._backend is None:
            return self._fallback.compact(passages)
        return self._compact_with_headroom(passages)

    def _compact_with_headroom(self, passages: Sequence[str]) -> list[str]:
        """Best-effort adapter over the headroom public surface.

        headroom's library API has evolved; rather than pin one entry point we
        probe a few documented shapes (a ``compress``/``compact`` function or a
        ``Compressor``-style object) and fall back to identity for any passage
        the backend cannot handle, so a backend version mismatch degrades to
        "no compression" instead of crashing the pipeline.
        """
        backend = self._backend
        fn = (
            getattr(backend, "compress", None)
            or getattr(backend, "compact", None)
        )
        out: list[str] = []
        for p in passages:
            try:
                if callable(fn):
                    res = fn(p)
                    out.append(res if isinstance(res, str) else str(res))
                else:  # unknown surface — preserve original
                    out.append(p)
            except Exception:  # noqa: BLE001 — never let compaction break a run
                logger.debug("HeadroomCompactor: passage skipped (backend error).")
                out.append(p)
        return out


__all__ = ["Compactor", "NoOpCompactor", "HeadroomCompactor"]
