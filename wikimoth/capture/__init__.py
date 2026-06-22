# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""WikiMoth capture — turn Claude Code sessions into deterministic ``[[wikilink]]`` notes.

This subpackage is WikiMoth's *write* half (the pipeline in :mod:`wikimoth.pipeline`
is the *read* half). It hooks into Claude Code's lifecycle, buffers what happens
in a session, and on flush writes **one markdown note per session** into a
``[[wikilink]]`` vault that :class:`wikimoth.MemoryRAG` can later retrieve.

The one invariant that makes this WikiMoth and not "yet another auto-memory":

    **The edges are computed by code, never by a model.**

A note's ``[[wikilinks]]`` come only from deterministic string/path matching
(:mod:`wikimoth.capture.links`): the previous session in the chain, files whose
basename matches an existing note, and existing note titles that appear verbatim
in the session text. An LLM may *optionally* draft the prose summary, but its
output is stripped of any ``[[...]]`` it tries to emit — so the graph the
retriever walks is reproducible and auditable, not sampled.

Nothing here makes a network call by default: with ``enable_llm_prose=False``
(the default) capture is pure-Python and API-free, mirroring the rest of WikiMoth
(the ``EchoReader`` philosophy). Imports are kept light; MOTHRAG's slugify is the
only cross-module reuse and it is imported lazily.
"""

from wikimoth.capture.buffer import SessionBuffer, make_record
from wikimoth.capture.config import CaptureConfig
from wikimoth.capture.note import build_note, write_session_note
from wikimoth.capture.recall import build_recall_block

__all__ = [
    "CaptureConfig",
    "SessionBuffer",
    "make_record",
    "build_note",
    "write_session_note",
    "build_recall_block",
]
