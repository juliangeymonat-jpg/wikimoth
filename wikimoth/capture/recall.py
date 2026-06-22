# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""SessionStart recall — the deterministic memory block injected at boot.

When a new session starts, the hook injects a compact summary of the most recent
sessions (newest first) into Claude's context via ``hookSpecificOutput
.additionalContext``. This is the "boot with memory" behaviour, kept simple and
deterministic: recency by filename (session stems carry an ISO date prefix, so
lexicographic sort *is* chronological). It does not call the retriever — the
graph walk happens on demand once there's an actual question.

The block is bounded by ``config.recall_max_notes`` and ``recall_max_chars`` so
it never floods the context (the whole point of WikiMoth).
"""

from __future__ import annotations

import re
from typing import Mapping

from wikimoth.capture.config import CaptureConfig
from wikimoth.capture.links import is_session_stem

_DESC_RE = re.compile(r"^description:\s*(.*)$", re.MULTILINE)
_WIKILINK = re.compile(r"\[\[([^\]\|#]+)(?:[#\|][^\]]*)?\]\]")


def _frontmatter_description(text: str) -> str:
    """Pull the ``description:`` scalar out of a note's YAML frontmatter."""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    head = text[: end if end != -1 else len(text)]
    m = _DESC_RE.search(head)
    if not m:
        return ""
    val = m.group(1).strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        val = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return val


def _links(text: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    for raw in _WIKILINK.findall(text):
        t = raw.strip()
        if t and t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def build_recall_block(config: CaptureConfig) -> str:
    """Return the SessionStart context block (``""`` if there's nothing yet)."""
    vault = config.vault_dir
    if not vault.exists():
        return ""

    sessions_dir = config.sessions_dir.resolve()
    stems: list[tuple[str, "Mapping"]] = []
    paths = []
    for p in sorted(vault.rglob("*.md")):
        try:
            if sessions_dir in p.resolve().parents:
                continue
        except OSError:  # pragma: no cover
            pass
        if is_session_stem(p.stem):
            paths.append(p)

    if not paths:
        return ""

    # Newest first; cap to recall_max_notes.
    paths = sorted(paths, key=lambda p: p.stem, reverse=True)[: config.recall_max_notes]

    header = [
        "## Recalled memory (WikiMoth — deterministic, token-minimal)",
        "Recent sessions in this vault (newest first):",
    ]
    body: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover
            continue
        desc = _frontmatter_description(text) or "(no description)"
        body.append(f"- {p.stem} — {desc}")
        links = _links(text)
        if links:
            body.append("  ↳ links: " + " ".join(f"[[{l}]]" for l in links))

    footer = [
        "",
        "These notes are retrievable on demand: ask about any topic and WikiMoth "
        "walks the [[wikilink]] graph to pull its connected note-chain "
        "(no model picks the edges).",
    ]

    block = "\n".join(header + body + footer)
    if len(block) > config.recall_max_chars:
        block = block[: config.recall_max_chars].rstrip() + "\n…(truncated)"
    return block


__all__ = ["build_recall_block"]
