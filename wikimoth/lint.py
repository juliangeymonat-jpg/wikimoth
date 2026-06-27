# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth lint`` — deterministic vault-hygiene checks (pure stdlib, read-only).

Surfaces the classic graph/text defects every PKM tool reports, computed over the
same ``[[wikilink]]`` graph the retriever walks, with ZERO false positives and no
model:

* **broken links** -- a ``[[target]]`` that resolves to no note (a typo, a
  renamed/deleted note, or a mis-stamped ``superseded_by`` edge). WikiMoth's
  retriever drops these silently at index time, so today nothing tells you.
* **orphans** -- a content note with no inlinks AND no outlinks (invisible to
  every graph walk).
* **duplicate slugs** -- two physically distinct files that collapse to one note
  identity (their chunks merge in the graph, corrupting retrieval identity).
* **stubs** -- a note with an empty body.
* **stale / expired** -- a note whose ``valid_to`` or ``expires`` date is in the
  past, or (opt-in) whose mtime is older than a threshold.
* **supersession chains / cycles** -- ``A superseded_by B superseded_by C`` (you
  should re-point A at C) or a cycle from two mis-directed ``supersede`` calls.

One shared inlink/title index backs every check (O(V+E)). It never imports the
retrieval pipeline, so it cannot regress retrieval. Read-only: it reports, never
writes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from wikimoth.capture.links import is_session_stem
from wikimoth.frontmatter import parse_frontmatter, split_frontmatter
from wikimoth.retrieval.graph import _WIKILINK_RE, _slugify


@dataclass
class _LNote:
    relpath: str
    slug: str
    is_session: bool
    body: str
    raw_targets: list[str]
    fields: dict[str, str]
    mtime: float


def _as_date(v: str) -> date | None:
    s = (v or "").strip()[:10]
    if len(s) != 10:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _link_slug(raw: str) -> str:
    """Slug of a frontmatter value that holds a ``[[wikilink]]`` (or a bare name)."""
    m = _WIKILINK_RE.search(raw or "")
    return _slugify(m.group(1) if m else (raw or ""))


def _supersede_target(fields: dict[str, str]) -> str:
    """The slug a note's ``superseded_by`` points at ('' if none)."""
    return _link_slug(fields.get("superseded_by", ""))


def _load(vault: Path) -> list[_LNote]:
    notes: list[_LNote] = []
    for p in sorted(vault.rglob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        _, body = split_frontmatter(text)
        try:
            rel = p.relative_to(vault).as_posix()
        except ValueError:  # pragma: no cover
            rel = p.name
        try:
            mtime = p.stat().st_mtime
        except OSError:  # pragma: no cover
            mtime = 0.0
        notes.append(
            _LNote(
                relpath=rel,
                slug=_slugify(p.name),
                is_session=is_session_stem(p.stem),
                body=body,
                raw_targets=list(_WIKILINK_RE.findall(text)),
                fields=parse_frontmatter(text),
                mtime=mtime,
            )
        )
    return notes


def scan_lint(
    vault_dir: str | Path,
    *,
    stale_days: int = 0,
    min_stub_chars: int = 0,
    include_sessions: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """Scan ``vault_dir`` and return a deterministic hygiene report (JSON-able).

    ``orphans``/``stubs``/``stale`` skip ``session-*`` notes (operational logs)
    unless ``include_sessions`` is set; graph-integrity checks (broken links,
    duplicate slugs, supersession chains) always cover every note. ``stale_days``
    enables the opt-in mtime check (0 = off). Output lists are sorted, so the
    report is byte-identical for the same vault EXCEPT the opt-in mtime check,
    which is inherently clock-relative. ``today`` overrides the date used for
    valid_to/expires (tests).
    """
    vault = Path(vault_dir)
    today = today or date.today()
    notes = _load(vault) if vault.is_dir() else []

    slug_to_paths: dict[str, list[str]] = {}
    for n in notes:
        slug_to_paths.setdefault(n.slug, []).append(n.relpath)

    # One shared pass: resolved inlinks + per-note resolved-outlink flag.
    inlinked: set[str] = set()
    for n in notes:
        for raw in n.raw_targets:
            t = _slugify(raw)
            if t and t != n.slug and t in slug_to_paths:
                inlinked.add(t)

    def has_authored_outlink(n: _LNote) -> bool:
        # An authored link to ANOTHER note (resolved or dangling): a note that
        # links somewhere is not an orphan, it has a broken link (reported there).
        return any((t := _slugify(raw)) and t != n.slug for raw in n.raw_targets)

    broken: list[dict[str, str]] = []
    orphans: list[str] = []
    stubs: list[str] = []
    stale: list[dict[str, str]] = []
    for n in notes:
        # broken links (all notes); dedup by (note, target slug). A ``replaces:``
        # value points BACKWARD to a possibly-deleted old note (audit, not a live
        # edge), so it is not a broken link.
        replaces_slug = _link_slug(n.fields.get("replaces", ""))
        seen_t: set[str] = set()
        for raw in n.raw_targets:
            t = _slugify(raw)
            if not t or t == n.slug or t in slug_to_paths or t in seen_t or t == replaces_slug:
                continue
            seen_t.add(t)
            broken.append({"note": n.relpath, "target": raw.strip()})

        if n.is_session and not include_sessions:
            continue  # orphans / stubs / stale skip operational logs
        if not has_authored_outlink(n) and n.slug not in inlinked:
            orphans.append(n.relpath)
        if len(n.body.strip()) <= min_stub_chars:
            stubs.append(n.relpath)
        vt = _as_date(n.fields.get("valid_to", ""))
        if vt is not None and vt < today:
            stale.append({"note": n.relpath, "reason": "valid_to", "date": vt.isoformat()})
        ex = _as_date(n.fields.get("expires", ""))
        if ex is not None and ex < today:
            stale.append({"note": n.relpath, "reason": "expires", "date": ex.isoformat()})
        if stale_days > 0 and (time.time() - n.mtime) > stale_days * 86400:
            stale.append({"note": n.relpath, "reason": "mtime", "date": ""})

    duplicate_slugs = [
        {"slug": s, "files": sorted(paths)}
        for s, paths in sorted(slug_to_paths.items())
        if len(paths) > 1
    ]

    chains, cycles = _supersession_chains(notes, slug_to_paths)

    broken.sort(key=lambda b: (b["note"], b["target"]))
    orphans.sort()
    stubs.sort()
    stale.sort(key=lambda s: (s["note"], s["reason"]))

    issues = (
        len(broken) + len(orphans) + len(duplicate_slugs) + len(stubs)
        + len(stale) + len(chains) + len(cycles)
    )
    return {
        "vault": str(vault),
        "scanned_notes": len(notes),
        "issues": issues,
        "broken_links": broken,
        "orphans": orphans,
        "duplicate_slugs": duplicate_slugs,
        "stubs": stubs,
        "stale": stale,
        "supersession_chains": chains,
        "supersession_cycles": cycles,
    }


def _supersession_chains(
    notes: list[_LNote], slug_to_paths: dict[str, list[str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Report transitive ``superseded_by`` chains (>=2 edges) + cycles.

    ``superseded_by`` is a functional edge (each note supersedes at most one), so
    the graph is a functional graph: every component has at most one cycle. Chains
    are reported from heads that terminate at a current note; cycles are found once
    each in canonical rotation (smallest slug first) regardless of how many heads
    feed them.
    """
    edge: dict[str, str] = {}
    for n in notes:
        tgt = _supersede_target(n.fields)
        if tgt and tgt != n.slug:
            edge[n.slug] = tgt
    targets = set(edge.values())

    # Chains: walk each head to a terminal; a walk that re-enters a node has hit a
    # downstream cycle (reported below), so it is not a chain.
    chains: list[dict[str, Any]] = []
    for start in sorted(s for s in edge if s not in targets):
        path = [start]
        seen_in_path = {start}
        cur = start
        while cur in edge:
            nxt = edge[cur]
            if nxt in seen_in_path:
                break  # enters a cycle; the cycle is reported separately
            path.append(nxt)
            seen_in_path.add(nxt)
            cur = nxt
        if cur not in edge and len(path) >= 3:  # ended at a current note, 2+ edges
            chains.append({"chain": path})

    # Cycles: functional-graph detection, each cycle once, canonical rotation.
    cycles: list[dict[str, Any]] = []
    global_seen: set[str] = set()
    for s in sorted(edge):
        if s in global_seen:
            continue
        path: list[str] = []
        idx: dict[str, int] = {}
        cur = s
        while cur in edge and cur not in global_seen and cur not in idx:
            idx[cur] = len(path)
            path.append(cur)
            cur = edge[cur]
        if cur in idx:  # re-entered the current walk: cur..cur is the loop
            loop = path[idx[cur]:]
            k = loop.index(min(loop))  # canonical: start at the smallest slug
            canon = loop[k:] + loop[:k] + [loop[k]]
            if all(c["cycle"] != canon for c in cycles):
                cycles.append({"cycle": canon})
        global_seen.update(path)
    return chains, cycles


def format_lint(report: dict[str, Any], *, fmt: str = "text") -> str:
    """Render a lint report as ``text`` (human/agent) or ``json``."""
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)

    head = (
        f"WikiMoth lint\nvault: {report['vault']}\n"
        f"scanned {report['scanned_notes']} note(s), {report['issues']} issue(s).\n"
    )
    if not report["issues"]:
        return head + "No hygiene issues found."

    out = [head]
    if report["broken_links"]:
        out.append("Broken links (target note does not exist):")
        out += [f"    {b['note']}  ->  [[{b['target']}]]" for b in report["broken_links"]]
    if report["orphans"]:
        out.append("Orphans (no inlinks and no outlinks):")
        out += [f"    {o}" for o in report["orphans"]]
    if report["duplicate_slugs"]:
        out.append("Duplicate identities (distinct files, same slug):")
        out += [f"    {d['slug']}  <-  {', '.join(d['files'])}" for d in report["duplicate_slugs"]]
    if report["stubs"]:
        out.append("Stubs (empty body):")
        out += [f"    {s}" for s in report["stubs"]]
    if report["stale"]:
        out.append("Stale / expired:")
        out += [f"    {s['note']}  ({s['reason']}{(' ' + s['date']) if s['date'] else ''})" for s in report["stale"]]
    if report["supersession_chains"]:
        out.append("Supersession chains (re-point the head at the current note):")
        out += [f"    {' -> '.join(c['chain'])}" for c in report["supersession_chains"]]
    if report["supersession_cycles"]:
        out.append("Supersession cycles (mis-directed supersede):")
        out += [f"    {' -> '.join(c['cycle'])}" for c in report["supersession_cycles"]]
    return "\n".join(out)


__all__ = ["scan_lint", "format_lint"]
