# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Shared, dependency-free helpers for reading note YAML frontmatter.

One home for the two idioms that were drifting between :mod:`wikimoth.capture.recall`
and :mod:`wikimoth.conflicts`: locating the ``---`` fenced block, and unquoting a
YAML scalar written by :func:`wikimoth.capture.note._yaml_str`. Keeping them here
means a change to the writer's escaping is decoded the same way by every reader.

Pure stdlib (``re`` only); no model, no network.
"""

from __future__ import annotations

import re

# Frontmatter ``key: value`` (NOT the Dataview ``key:: value`` inline form — the
# negative lookahead keeps ``::`` out). A nested block (``metadata:`` then indented
# children) flattens to ``metadata.<child>``.
_FM_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_.\-]+):(?!:)\s?(.*)$")


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter_block, body)``; the block excludes the ``---`` fences.

    No frontmatter -> ``("", text)``. An unterminated opening fence -> everything
    after it is the block and the body is empty. Tolerant of CRLF: the closing
    fence is found via ``"\\n---"`` so ``\\r\\n---`` still matches.
    """
    if not text.startswith("---"):
        return "", text
    end = text.find("\n---", 3)
    if end == -1:
        return text[3:], ""
    block = text[3:end]
    after = text[end + 4:]  # skip the "\n---"
    nl = after.find("\n")
    body = after[nl + 1:] if nl != -1 else ""
    return block, body


def unquote_scalar(raw: str) -> str:
    """Decode a YAML scalar: strip double-quotes (note.py escaping) or a comment.

    A quoted scalar may be followed by a trailing ``# comment`` (``"x" # note``):
    take only the content up to the first unescaped closing quote. An unquoted
    scalar drops a trailing `` # comment`` (space-hash, so ``C#`` survives).
    """
    s = raw.strip()
    if s.startswith('"'):
        out: list[str] = []
        i, n = 1, len(s)
        while i < n:
            ch = s[i]
            if ch == "\\" and i + 1 < n and s[i + 1] in ('"', "\\"):
                out.append(s[i + 1])  # escaped quote or backslash -> literal
                i += 2
                continue
            if ch == '"':
                return "".join(out)  # the real closing quote
            out.append(ch)
            i += 1
        return "".join(out)  # unterminated quote: best-effort
    hash_at = s.find(" #")
    if hash_at != -1:
        s = s[:hash_at]
    return s.strip()


def parse_frontmatter(text: str) -> dict[str, str]:
    """Flat ``key -> value`` map of a note's YAML frontmatter.

    A one-level nested block (``metadata:`` then indented children) flattens to
    ``metadata.child`` keys. Container headers (a key with no scalar value) are
    not stored as values. An indented line with no container parent (malformed)
    is skipped.
    """
    block, _ = split_frontmatter(text)
    if not block:
        return {}

    out: dict[str, str] = {}
    parent: str | None = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _FM_KEY_RE.match(line)
        if not m:
            continue
        indent, key, rawval = len(m.group(1)), m.group(2), m.group(3)
        val = unquote_scalar(rawval)
        if indent == 0:
            if val == "":
                parent = key  # container header, e.g. ``metadata:``
            else:
                out[key] = val
                parent = None
        else:
            if parent is None:
                continue  # indented line with no container parent: malformed, skip
            if val != "":
                out[f"{parent}.{key}"] = val
    return out


__all__ = ["split_frontmatter", "unquote_scalar", "parse_frontmatter"]
