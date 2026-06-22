# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Deterministic link computation — the heart of WikiMoth capture.

Everything here is a **pure function of its inputs** and uses only string/path
matching. No model, no embeddings, no network decides an edge. This is the
property that separates WikiMoth from LLM-summarised auto-memory: given the same
session buffer and the same vault, the emitted ``[[wikilinks]]`` are identical
every run, and you can read the code that produced each one.

Three edge sources, each conservative on purpose (a wrong edge is worse than a
missing one — the previous-session chain and file-to-note matches are the
load-bearing edges; title-phrase matches are a bonus):

1. **Previous session** — the chronologically prior session note (the chain that
   lets the graph walk back through your history).
2. **File → note** — a file whose basename matches an existing note's slug
   (editing ``foo.md`` in the vault links the session to ``[[foo]]``).
3. **Title phrase** — an existing note whose full title appears as a verbatim,
   word-bounded phrase in the session text (prompts + commands + file paths).

Targets are emitted as ``[[<filename-stem>]]``; MOTHRAG's ``GraphRetriever``
slugifies the target, so the stem resolves to that note's identity. Links whose
target isn't an indexed note are silently dropped by the retriever (dangling).
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

# Tiny stop-list so a single-token note title that happens to be a common word
# doesn't link everything. Kept intentionally small + lowercase.
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "test", "tests",
    "note", "notes", "todo", "main", "code", "file", "files", "data", "memory",
}

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _slugify(name: str) -> str:
    """Note-identity slug, identical to MOTHRAG's ``GraphRetriever._slugify``.

    Reuses the lab symbol so WikiMoth and the retriever agree on identity; falls
    back to an equivalent local implementation only if that symbol moves.
    """
    try:
        from wikimoth.pipeline import _slugify_note

        return _slugify_note(name)
    except Exception:  # pragma: no cover - pipeline present in practice
        s = str(name or "").strip().replace("\\", "/")
        if "/" in s:
            s = s.rsplit("/", 1)[-1]
        low = s.lower()
        for ext in (".markdown", ".md"):
            if low.endswith(ext):
                s = s[: -len(ext)]
                break
        s = s.lower().replace("_", " ").replace("-", " ")
        return _WS.sub(" ", s).strip()


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, unify ``_``/``-`` to spaces, collapse runs."""
    s = (text or "").lower().replace("_", " ").replace("-", " ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def is_session_stem(stem: str) -> bool:
    """True for a WikiMoth session note (``session-<date>-<id>``)."""
    return str(stem or "").startswith("session-")


def collect_corpus(events: Sequence[Mapping], *, max_chars: int) -> str:
    """Build the (bounded) text scanned for note-title matches.

    Sources: user prompts + bash commands + file paths. Tool *outputs* are never
    included (leanness + we don't want output noise driving edges).
    """
    parts: list[str] = []
    for rec in events:
        if not isinstance(rec, Mapping):
            continue
        if rec.get("prompt"):
            parts.append(str(rec["prompt"]))
        if rec.get("command"):
            parts.append(str(rec["command"]))
        if rec.get("file_path"):
            parts.append(str(rec["file_path"]))
    corpus = "\n".join(parts)
    return corpus[:max_chars]


def files_touched(events: Sequence[Mapping]) -> list[str]:
    """Unique file paths touched by mutating tools, in first-seen order."""
    mutating = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
    seen: dict[str, None] = {}
    for rec in events:
        if not isinstance(rec, Mapping):
            continue
        if rec.get("tool") in mutating and rec.get("file_path"):
            seen.setdefault(str(rec["file_path"]), None)
    return list(seen)


def match_notes(
    corpus_text: str,
    note_index: Mapping[str, str],
    *,
    min_token_len: int = 4,
    exclude_slugs: Iterable[str] = (),
) -> list[str]:
    """Existing notes whose full title appears verbatim in ``corpus_text``.

    ``note_index`` maps ``slug -> filename_stem``. Returns the matched **stems**,
    sorted. A single-token title must be at least ``min_token_len`` chars and not
    a stop-word; a multi-token title must appear as a contiguous word-bounded
    phrase. Conservative by design.
    """
    norm = _normalize(corpus_text)
    if not norm:
        return []
    padded = f" {norm} "
    excl = set(exclude_slugs)
    out: dict[str, None] = {}
    for slug, stem in note_index.items():
        if slug in excl:
            continue
        phrase = _normalize(slug)
        if not phrase:
            continue
        toks = phrase.split()
        if len(toks) == 1:
            t = toks[0]
            if len(t) < min_token_len or t in _STOPWORDS:
                continue
        if f" {phrase} " in padded:
            out[stem] = None
    return sorted(out)


def file_note_links(
    files: Sequence[str],
    note_index: Mapping[str, str],
    *,
    exclude_slugs: Iterable[str] = (),
) -> list[str]:
    """Notes whose slug equals a touched file's basename slug. Returns stems."""
    excl = set(exclude_slugs)
    out: dict[str, None] = {}
    for fp in files:
        slug = _slugify(fp)
        if slug and slug not in excl and slug in note_index:
            out[note_index[slug]] = None
    return sorted(out)


def previous_session(self_stem: str, session_stems: Iterable[str]) -> str | None:
    """The chronologically prior session note stem (or ``None``).

    Session stems sort chronologically because the date is an ISO prefix, so the
    largest stem strictly less than ``self_stem`` is the previous session.
    """
    prior = sorted(s for s in session_stems if s and s < self_stem)
    return prior[-1] if prior else None


def compute_links(
    *,
    self_stem: str,
    events: Sequence[Mapping],
    note_index: Mapping[str, str],
    session_stems: Sequence[str],
    max_match_chars: int = 8000,
    min_token_len: int = 4,
) -> list[str]:
    """All ``[[wikilink]]`` target *stems* for a session note, deduped + ordered.

    Order is stable and meaningful: previous-session first (the chain), then
    file→note links, then title-phrase links — each group sorted, the whole list
    deduped preserving that priority. Pure: same inputs → same list.
    """
    ordered: list[str] = []

    prev = previous_session(self_stem, session_stems)
    if prev:
        ordered.append(prev)

    files = files_touched(events)
    ordered.extend(file_note_links(files, note_index))

    corpus = collect_corpus(events, max_chars=max_match_chars)
    ordered.extend(match_notes(corpus, note_index, min_token_len=min_token_len))

    # Dedup preserving first occurrence (priority order); never link to self.
    out: list[str] = []
    seen = {self_stem}
    for stem in ordered:
        if stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


__all__ = [
    "collect_corpus",
    "files_touched",
    "match_notes",
    "file_note_links",
    "previous_session",
    "compute_links",
    "is_session_stem",
]
