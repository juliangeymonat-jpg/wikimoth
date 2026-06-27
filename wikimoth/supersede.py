# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth supersede OLD NEW`` — mark OLD as replaced by NEW, without deleting it.

Invalidate-don't-delete + bitemporal (Zep/Graphiti, Snodgrass): the OLD note is
never removed (git stays the audit trail); only its frontmatter is mutated to
record the supersession as machine-readable, top-level keys:

* ``superseded_by: "[[new-stem]]"`` -- the ``dcterms:isReplacedBy`` edge (subject =
  OLD, points at NEW). Written as a real ``[[wikilink]]`` so the graph traverses
  it; the stem is resolved to NEW's identity BEFORE writing so it can never dangle.
* ``valid_to`` / ``invalidated_at`` -- when the fact stopped being true / when we
  recorded it (the bitemporal close).
* ``status: superseded`` -- a Dataview-queryable flag.
* ``supersession_reason`` -- optional audit note.

This module is the WRITE op only (the as-of / supersession-aware retrieval is a
separate read-side change). It is the single non-deterministic boundary of the
hygiene suite: a human or agent ASSERTS that OLD is superseded by NEW; the tool
only executes that assertion, it never infers a contradiction. Pure stdlib,
atomic write (temp file + ``os.replace``).
"""

from __future__ import annotations

import difflib
import os
from datetime import date
from pathlib import Path
from typing import Any

from wikimoth.capture.links import is_session_stem
from wikimoth.capture.note import _yaml_str
from wikimoth.frontmatter import parse_frontmatter, split_frontmatter
from wikimoth.retrieval.graph import _WIKILINK_RE, _slugify

# Top-level keys this op owns (written on supersede, removed on --undo).
_MANAGED = ("superseded_by", "status", "valid_to", "invalidated_at", "supersession_reason")


class SupersedeError(ValueError):
    """A user/usage error (unresolved note, self-supersede, session note, ...)."""


def _resolve(vault: Path, ref: str) -> Path:
    """Resolve OLD/NEW (a stem, slug, or path) to one ``.md`` file INSIDE ``vault``.

    Refuses any ref that resolves outside the vault (an absolute path or ``../``
    escape) so the write op can never touch a file beyond the memory folder.
    """
    p = Path(ref)
    resolved: Path | None = None
    if p.is_file():
        resolved = p
    else:
        for cand in (vault / ref, vault / f"{ref}.md"):
            if cand.is_file():
                resolved = cand
                break
        if resolved is None:
            target = _slugify(ref)
            matches = [q for q in sorted(vault.rglob("*.md")) if _slugify(q.name) == target]
            if len(matches) > 1:
                raise SupersedeError(f"{ref!r} is ambiguous ({len(matches)} notes share that slug); pass a path")
            if matches:
                resolved = matches[0]
    if resolved is None:
        raise SupersedeError(f"note not found: {ref!r} (tried stem, slug, and path under {vault})")
    try:
        resolved.resolve().relative_to(vault.resolve())
    except ValueError:
        raise SupersedeError(f"{ref!r} resolves outside the vault; refusing to write there")
    return resolved


def _rewrite_frontmatter(text: str, *, set_lines: dict[str, str], drop_keys: set[str]) -> str:
    """Return ``text`` with top-level ``set_lines`` inserted and ``drop_keys`` removed.

    Only TOP-LEVEL (column-0) keys are touched; nested children and the body are
    preserved. Managed keys are inserted at the TOP of the frontmatter (right after
    the opening fence) so placement never depends on guessing where a nested block
    starts and can never split a ``tags:`` list or orphan children. Dropping a
    top-level key also drops any indented lines that belong to it. ``text`` is the
    in-memory note (read_text already normalised CRLF to ``\\n``). A note with no
    frontmatter fence gets a minimal one.
    """
    drop = set(drop_keys) | set(set_lines)
    new_kv = [f"{k}: {v}" for k, v in set_lines.items()]

    if not text.startswith("---"):
        if not new_kv:
            return text
        return "---\n" + "\n".join(new_kv) + "\n---\n" + text

    block, body = split_frontmatter(text)
    lines = block.split("\n")
    kept: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_top = line[:1] not in (" ", "\t") and ":" in line
        if is_top and line.split(":", 1)[0].strip() in drop:
            i += 1  # skip the managed key and any indented children it owns
            while i < len(lines) and lines[i][:1] in (" ", "\t"):
                i += 1
            continue
        kept.append(line)
        i += 1

    # block begins with the "" left of the opening fence's newline; insert after it.
    insert_at = 1 if kept and kept[0] == "" else 0
    merged = kept[:insert_at] + new_kv + kept[insert_at:]
    return "---" + "\n".join(merged) + "\n---\n" + body


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".wm-tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def supersede(
    vault_dir: str | Path,
    old: str,
    new: str,
    *,
    as_of: date | None = None,
    reason: str = "",
    undo: bool = False,
    reverse: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Mark OLD superseded by NEW (or undo it). Returns a result dict.

    Raises :class:`SupersedeError` on a usage error. ``as_of`` defaults to today.
    ``reverse`` also stamps ``replaces: "[[old]]"`` on NEW. ``dry_run`` writes
    nothing and returns a unified diff. Session notes are refused (their note is
    rebuilt from the buffer on every flush, which would erase the supersession).
    """
    vault = Path(vault_dir)
    if not vault.is_dir():
        raise SupersedeError(f"vault not found (or not a directory): {vault}")
    old_path = _resolve(vault, old)
    new_path = _resolve(vault, new)
    if old_path.resolve() == new_path.resolve():
        raise SupersedeError("OLD and NEW are the same note")
    if is_session_stem(old_path.stem):
        raise SupersedeError(
            f"{old_path.name} is a session note; supersede targets curated content "
            "notes (a session note is rebuilt from its buffer and would lose the edit)"
        )

    old_text = old_path.read_text(encoding="utf-8", errors="replace")
    old_fields = parse_frontmatter(old_text)
    as_of_iso = (as_of or date.today()).isoformat()

    edits: list[tuple[Path, str, str]] = []  # (path, before, after)

    if undo:
        if "superseded_by" not in old_fields:
            raise SupersedeError(f"{old_path.name} is not superseded; nothing to undo")
        after = _rewrite_frontmatter(old_text, set_lines={}, drop_keys=set(_MANAGED))
        edits.append((old_path, old_text, after))
        if reverse:
            new_text = new_path.read_text(encoding="utf-8", errors="replace")
            edits.append((new_path, new_text, _rewrite_frontmatter(new_text, set_lines={}, drop_keys={"replaces"})))
    else:
        # slugify-resolves-before-write: the stamped [[stem]] must resolve to a
        # single NEW note (a slug collision would make the edge ambiguous).
        new_stem = new_path.stem
        new_slug = _slugify(new_stem)
        if sum(1 for q in vault.rglob("*.md") if _slugify(q.name) == new_slug) != 1:
            raise SupersedeError(
                f"[[{new_stem}]] is ambiguous (multiple notes share that slug); "
                "cannot stamp a resolvable supersession edge"
            )
        set_lines = {
            "superseded_by": f'"[[{new_stem}]]"',
            "status": "superseded",
            "valid_to": as_of_iso,
            "invalidated_at": as_of_iso,
        }
        if reason:
            set_lines["supersession_reason"] = _yaml_str(reason)
        after = _rewrite_frontmatter(old_text, set_lines=set_lines, drop_keys=set(_MANAGED))
        edits.append((old_path, old_text, after))
        if reverse:
            new_text = new_path.read_text(encoding="utf-8", errors="replace")
            after_new = _rewrite_frontmatter(
                new_text, set_lines={"replaces": f'"[[{old_path.stem}]]"'}, drop_keys={"replaces"}
            )
            edits.append((new_path, new_text, after_new))

    diff_lines: list[str] = []
    changed = False
    for path, before, after in edits:
        if before != after:
            changed = True
        if dry_run:
            rel = path.name
            diff_lines += list(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"{rel} (before)",
                    tofile=f"{rel} (after)",
                )
            )

    if not dry_run and changed:
        for path, before, after in edits:
            if before != after:
                _atomic_write(path, after)

    return {
        "action": "undo" if undo else "supersede",
        "old": old_path.name,
        "new": new_path.name,
        "as_of": as_of_iso,
        "changed": changed,
        "dry_run": dry_run,
        "diff": "".join(diff_lines),
    }


def format_result(result: dict[str, Any]) -> str:
    """Human/agent-readable one-liner (or the dry-run diff)."""
    if result.get("dry_run"):
        head = f"[dry-run] {result['action']}: {result['old']} -> {result['new']}\n"
        return head + (result["diff"] or "(no change)")
    if not result["changed"]:
        return f"{result['action']}: no change ({result['old']})"
    if result["action"] == "undo":
        return f"undo: {result['old']} is no longer marked superseded"
    return (
        f"supersede: {result['old']} -> [[{result['new']}]] "
        f"(valid_to {result['as_of']}, status superseded). The file is kept; git is the audit trail."
    )


__all__ = ["supersede", "format_result", "SupersedeError"]
