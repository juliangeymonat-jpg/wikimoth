# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth conflicts`` — deterministic contradiction *candidate* surfacing.

Finds notes that assert **different values for the same (subject, predicate)** —
the functional-property conflict (PaTeCon-style) — over a ``[[wikilink]]`` vault,
using only the notes' YAML frontmatter and pure stdlib. No model, no network, no
dependency: same vault in, byte-identical report out.

What it is NOT
-------------
This module never decides that a contradiction is *real*. Deciding "B genuinely
contradicts A / B supersedes A / they coexist" is a semantic (NLI) judgement that
needs a model — and WikiMoth keeps the model OUT of its loop. So ``conflicts``
emits a **candidate contract** (structured JSON): the calling agent (Claude via
the MCP server, which already has a model in the loop) renders the verdict and,
if it decides one note wins, calls a curation op. This is the retrieve-then-
adjudicate split: deterministic candidate-gen here, semantic verdict upstream.

Subject resolution
------------------
A note's *subject* is the entity its facts are about. By default it is the value
of the first present subject key (``subject`` / ``entity`` / ``about`` / ``topic``)
in the frontmatter, normalised with the same ``_slugify`` the retriever uses, so
``about: "[[Brain Forge]]"`` and ``subject: brain-forge`` group together. With no
subject key the subject falls back to the note's own slug, which surfaces the
*slug-collision* case (two physically distinct files that collapse to one graph
identity) disagreeing on a field.

Temporal precision
-----------------
Two notes asserting the same predicate with **disjoint** ``valid_from``/``valid_to``
intervals are a legitimate temporal succession, not a contradiction; only
**overlapping** valid-time with different values is the real conflict. Each
candidate is tagged ``valid_time: overlapping | disjoint | unknown`` and a
matching confidence, so the agent can prioritise (and disjoint pairs hint a
``supersede`` rather than a conflict).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

# Reuse shared readers (don't fork them): the subject identity must match exactly
# how the graph walks edges, and the frontmatter idioms live in one place.
from wikimoth.capture.links import is_session_stem
from wikimoth.frontmatter import parse_frontmatter, split_frontmatter, unquote_scalar
from wikimoth.retrieval.graph import _WIKILINK_RE, _slugify

# Dataview inline field in the body (only scanned with include_inline=True).
_INLINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+)::\s*(.*)$", re.MULTILINE)
# A full ISO calendar date and nothing else (avoids fromisoformat accepting
# week-dates / basic-format on 3.11+, which would make typing interpreter-
# version-dependent; and avoids a value's trailing text being truncated to a date).
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Keys that name the subject, in resolution order.
DEFAULT_SUBJECT_KEYS: tuple[str, ...] = ("subject", "entity", "about", "topic")

# Structural / bookkeeping keys that are never domain *facts*: comparing them
# across notes is pure noise (every session note has a unique session_id, date,
# cwd...). Excluded from predicates unless ``all_keys=True``.
DEFAULT_IGNORE_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "aliases",
        # bitemporal / supersession bookkeeping (the hygiene schema itself)
        "valid_from",
        "valid_to",
        "superseded_by",
        "replaces",
        "status",
        "invalidated_at",
        "supersession_reason",
        # WikiMoth session-note metadata block
        "metadata.type",
        "metadata.session_id",
        "metadata.date",
        "metadata.cwd",
        "metadata.source",
        "metadata.tools",
        "metadata.files",
    }
)


# ---------------------------------------------------------------------------
# Frontmatter parsing (deterministic, stdlib-only)
# ---------------------------------------------------------------------------
def _inline_fields(text: str) -> dict[str, str]:
    """Dataview ``key:: value`` inline fields from the note body (best-effort)."""
    _, body = split_frontmatter(text)
    out: dict[str, str] = {}
    for key, raw in _INLINE_RE.findall(body):
        out.setdefault(key, unquote_scalar(raw))
    return out


# ---------------------------------------------------------------------------
# Value typing (bool -> date -> numeric -> string), all deterministic
# ---------------------------------------------------------------------------
_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}


def _as_date(v: str) -> date | None:
    """Parse a value that is EXACTLY a full ISO date ``YYYY-MM-DD``, else None.

    Requires the whole stripped value to match (no truncation): ``2026-01-01 x``
    stays a string, and ``2026-6-1`` / ``2026-W01-1`` / ``20260101`` are rejected
    so typing does not depend on the Python version's fromisoformat leniency.
    """
    s = v.strip()
    if not _DATE_RE.match(s):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _as_num(v: str) -> float | None:
    s = v.strip()
    if "_" in s:  # YAML treats 1_000 as a string, not a number; don't collapse it
        return None
    try:
        n = float(s)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(n):  # inf / -inf / nan are not domain numbers (and int() would raise)
        return None
    return n


def _classify(v: str, *, case_insensitive: bool = False) -> tuple[str, Any, str]:
    """Return ``(kind, comparable, canonical)`` for a scalar value string."""
    low = v.strip().lower()
    if low in _BOOL_TRUE:
        return "boolean", True, "true"
    if low in _BOOL_FALSE:
        return "boolean", False, "false"
    d = _as_date(v)
    if d is not None:
        return "date", d, d.isoformat()
    n = _as_num(v)
    if n is not None:
        # Normalise 5 and 5.0 to one canonical form.
        canon = str(int(n)) if n == int(n) else repr(n)
        return "numeric", n, canon
    s = v.strip()
    canon = s.lower() if case_insensitive else s
    return "string", s, canon


# ---------------------------------------------------------------------------
# Note model + scan
# ---------------------------------------------------------------------------
@dataclass
class _Note:
    slug: str
    filename: str
    fields: dict[str, str]
    valid_from: date | None
    valid_to: date | None
    subject: str
    subject_source: str


def _resolve_subject(fields: dict[str, str], slug: str, subject_keys: Iterable[str]) -> tuple[str, str]:
    """``(subject, source)``: first present subject key (slugified), else the note slug."""
    for key in subject_keys:
        if key in fields and fields[key].strip():
            raw = fields[key]
            m = _WIKILINK_RE.search(raw)
            target = m.group(1) if m else raw
            subj = _slugify(target)
            if subj:
                return subj, f"frontmatter:{key}"
    return slug, "note-slug"


def _load_note(path: Path, *, display: str, subject_keys: Iterable[str], include_inline: bool) -> _Note:
    text = path.read_text(encoding="utf-8", errors="replace")
    fields = parse_frontmatter(text)
    if include_inline:
        for k, v in _inline_fields(text).items():
            fields.setdefault(k, v)
    slug = _slugify(path.name)
    subject, source = _resolve_subject(fields, slug, subject_keys)
    return _Note(
        slug=slug,
        filename=display,
        fields=fields,
        valid_from=_as_date(fields.get("valid_from", "")),
        valid_to=_as_date(fields.get("valid_to", "")),
        subject=subject,
        subject_source=source,
    )


def _intervals_overlap(a: _Note, b: _Note) -> bool:
    """Half-open ``[valid_from, valid_to)`` overlap; missing bound = open (-inf/+inf)."""
    a_lo = a.valid_from or date.min
    a_hi = a.valid_to or date.max
    b_lo = b.valid_from or date.min
    b_hi = b.valid_to or date.max
    return a_lo < b_hi and b_lo < a_hi


def _has_bound(n: _Note) -> bool:
    return bool(n.valid_from or n.valid_to)


def _temporal_status(items: list[tuple[_Note, str]]) -> tuple[str, str]:
    """``(valid_time, confidence)`` judged over DIFFERING-value pairs only.

    Items are ``(note, canonical_value)``. Only pairs whose canonical values
    actually differ are a contradiction; same-value pairs (which can overlap as a
    legitimate restatement) must not force ``overlapping``. A differing pair is
    judged temporally only when BOTH notes carry valid-time:

    * any differing pair (both bounded) overlaps -> ``overlapping`` / high (real);
    * every differing pair is bounded and disjoint -> ``disjoint`` / low (a
      legitimate succession; likely a ``supersede``, not a conflict);
    * otherwise (no differing pair is fully bounded) -> ``unknown`` / medium.
    """
    diff_pairs = [
        (items[i][0], items[j][0])
        for i in range(len(items))
        for j in range(i + 1, len(items))
        if items[i][1] != items[j][1]
    ]
    bounded = [(a, b) for a, b in diff_pairs if _has_bound(a) and _has_bound(b)]
    if not bounded:
        return "unknown", "medium"
    if any(_intervals_overlap(a, b) for a, b in bounded):
        return "overlapping", "high"
    if len(bounded) == len(diff_pairs):
        return "disjoint", "low"
    return "unknown", "medium"  # some differing pair is unjudgeable (missing bounds)


def _group_equivalent(values: list[tuple[str, Any, str]], *, num_tol: float, date_tol_days: int) -> bool:
    """True if all values are equivalent (no conflict), honouring type tolerances."""
    kinds = {k for k, _, _ in values}
    if kinds == {"numeric"} and num_tol > 0:
        nums = [c for _, c, _ in values]
        return max(nums) - min(nums) <= num_tol
    if kinds == {"date"} and date_tol_days > 0:
        ds = [c for _, c, _ in values]
        return (max(ds) - min(ds)).days <= date_tol_days
    return len({canon for _, _, canon in values}) == 1


def scan_conflicts(
    vault_dir: str | Path,
    *,
    subject_keys: Iterable[str] = DEFAULT_SUBJECT_KEYS,
    extra_ignore: Iterable[str] = (),
    all_keys: bool = False,
    num_tol: float = 0.0,
    date_tol_days: int = 0,
    include_inline: bool = False,
    include_sessions: bool = False,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Scan ``vault_dir`` and return a deterministic conflict-candidate report.

    The report is a JSON-serialisable dict (the candidate contract the agent
    adjudicates). Conflicts and their values are sorted so the output is
    byte-identical for the same vault. ``session-*`` notes (operational logs, not
    curated facts) are skipped unless ``include_sessions`` is set, so the scanned
    count reflects curated content rather than capture logs.
    """
    vault = Path(vault_dir)
    subject_keys = tuple(subject_keys)
    ignore = set(DEFAULT_IGNORE_KEYS) | set(extra_ignore) | set(subject_keys)

    notes: list[_Note] = []
    if vault.is_dir():
        for p in sorted(vault.rglob("*.md")):
            if not include_sessions and is_session_stem(p.stem):
                continue  # operational session log, not a curated fact
            try:
                display = p.relative_to(vault).as_posix()  # unique per file, stable across OSes
            except ValueError:  # pragma: no cover - p is always under vault
                display = p.name
            notes.append(
                _load_note(p, display=display, subject_keys=subject_keys, include_inline=include_inline)
            )

    # Bucket: (subject, predicate) -> list of (note, value-string)
    buckets: dict[tuple[str, str], list[tuple[_Note, str]]] = {}
    for n in notes:
        for key, val in n.fields.items():
            if not all_keys and key in ignore:
                continue
            if all_keys and key in set(subject_keys):
                continue  # the subject is never its own predicate
            buckets.setdefault((n.subject, key), []).append((n, val))

    conflicts: list[dict[str, Any]] = []
    for (subject, predicate), pairs in buckets.items():
        # Need at least two DISTINCT notes to disagree.
        by_file: dict[str, tuple[_Note, str]] = {}
        for n, val in pairs:
            by_file.setdefault(n.filename, (n, val))
        if len(by_file) < 2:
            continue
        # Sort FIRST so classification, output, and temporal status all share one
        # order; filename is the vault-relative path (unique), so the key is total
        # even when two files slugify to the same identity (slug collision).
        members = sorted(by_file.values(), key=lambda nv: (nv[0].slug, nv[0].filename))
        classified = [_classify(v, case_insensitive=case_insensitive) for _, v in members]
        if _group_equivalent(classified, num_tol=num_tol, date_tol_days=date_tol_days):
            continue

        kinds = {k for k, _, _ in classified}
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        valid_time, confidence = _temporal_status(
            [(members[k][0], classified[k][2]) for k in range(len(members))]
        )
        subject_source = members[0][0].subject_source
        conflicts.append(
            {
                "subject": subject,
                "subject_source": subject_source,
                "predicate": predicate,
                "kind": kind,
                "valid_time": valid_time,
                "confidence": confidence,
                "values": [
                    {
                        "value": val,
                        "note": n.slug,
                        "file": n.filename,
                        "valid_from": n.valid_from.isoformat() if n.valid_from else None,
                        "valid_to": n.valid_to.isoformat() if n.valid_to else None,
                    }
                    for n, val in members
                ],
            }
        )

    conflicts.sort(key=lambda c: (c["subject"], c["predicate"]))
    return {
        "vault": str(vault),
        "scanned_notes": len(notes),
        "subject_keys": list(subject_keys),
        "conflicts": conflicts,
        "note": (
            "Deterministic candidates only: each is two notes disagreeing on one "
            "(subject, predicate). The contradiction VERDICT (real conflict? which "
            "wins? supersede or coexist?) is for the calling agent to decide."
        ),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def format_conflicts(report: dict[str, Any], *, fmt: str = "text") -> str:
    """Render a scan report as ``text`` (human/agent) or ``json`` (the contract)."""
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)

    conflicts = report["conflicts"]
    head = (
        f"WikiMoth conflict candidates\nvault: {report['vault']}\n"
        f"scanned {report['scanned_notes']} note(s), "
        f"{len(conflicts)} candidate conflict(s).\n"
    )
    if not conflicts:
        return head + "No contradicting (subject, predicate) found. Vault is consistent on structured fields."
    # ASCII-only markers: text output must never raise UnicodeEncodeError on a
    # cp1252 console / when piped or redirected.
    marker = {"high": "[!]", "medium": "[*]", "low": "[.]"}
    blocks = []
    for c in conflicts:
        m = marker.get(c["confidence"], "[*]")
        lines = [
            f"{m} [{c['confidence']}] {c['subject']} | {c['predicate']} "
            f"({c['kind']}, valid-time {c['valid_time']}, subject from {c['subject_source']})"
        ]
        for v in c["values"]:
            span = ""
            if v["valid_from"] or v["valid_to"]:
                span = f"  (valid {v['valid_from'] or '-inf'} .. {v['valid_to'] or 'open'})"
            lines.append(f"    {v['value']!r}  <- {v['file']}{span}")
        blocks.append("\n".join(lines))
    foot = (
        "\nThese are deterministic candidates. The VERDICT (is it a real "
        "contradiction? which note wins?) is yours to decide."
    )
    return head + "\n" + "\n\n".join(blocks) + "\n" + foot


__all__ = ["scan_conflicts", "format_conflicts", "parse_frontmatter"]
