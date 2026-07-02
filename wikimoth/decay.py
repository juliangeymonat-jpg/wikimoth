# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth decay`` — a retrieval-decay "fading" review queue (read-only).

Surfaces notes that are going cold so a human can review/archive them, WITHOUT
ever deleting anything. A note's *strength* combines three deterministic signals:

* **recency** -- ``exp(-age_days / tau)`` where ``age`` is days since the note's
  ``last_access`` (frontmatter) or, absent that, its file mtime. Old notes decay.
* **access** -- a ``hit_count`` frontmatter counter reinforces a note (the more it
  is recalled, the slower it fades). Forgetting-curve style.
* **connectivity** -- a note many others ``[[link]]`` to is load-bearing and
  should resist fading even when old.

``strength = recency + 0.15*ln(1+hit_count) + 0.15*inlink_degree``. Notes below a
threshold are the *fading queue*, reported with their signals so the decision is
the human's. Nothing is auto-deleted (WikiMoth is append-only + git-audited).

Note: automatic ``hit_count``/``last_access`` instrumentation on the recall path
(write-on-read) is deliberately deferred — it would mutate the vault on every
query. This command reads those fields when present and falls back to mtime +
connectivity, so it is useful today and richer once access is tracked.

Pure stdlib, read-only. The age signal is clock-relative by nature (decay is), so
the report moves as time passes; for a fixed ``today`` it is reproducible for
notes that carry a date (``last_access``/``metadata.date``/``date``/``created``).
Notes that fall back to file mtime can vary across a fresh git checkout (which
resets mtime), so a dated frontmatter field is preferred when present.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from wikimoth.capture.links import is_session_stem
from wikimoth.frontmatter import parse_frontmatter
from wikimoth.retrieval.graph import _WIKILINK_RE, _slugify

_W_HITS = 0.15   # reinforcement weight per ln(1+hit): ~50 hits protect, a few do not
_W_LINKS = 0.15  # connectivity weight per inlink: a small hub (>=2 inlinks) resists fading


def _parse_iso_date(s: Any) -> date | None:
    if not isinstance(s, str):
        return None
    s = s.strip()[:10]
    if len(s) != 10:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _as_int(v: Any) -> int:
    try:
        n = int(str(v).strip())
        return n if n >= 0 else 0
    except (ValueError, TypeError):
        return 0


def scan_decay(
    vault_dir: str | Path,
    *,
    tau_days: float = 90.0,
    threshold: float = 0.25,
    limit: int = 50,
    include_sessions: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """Scan ``vault_dir`` and return the fading-queue report (read-only).

    Notes whose ``strength`` falls below ``threshold`` are returned (most-faded
    first), capped at ``limit`` (``truncated`` flags when more exist). ``today``
    overrides the reference date (tests); ``tau_days`` sets the decay half-life.
    """
    vault = Path(vault_dir)
    today = today or date.today()
    tau = max(1.0, float(tau_days))

    raw: list[tuple[str, str, dict[str, str], float, list[str]]] = []
    if vault.is_dir():
        for p in sorted(vault.rglob("*.md")):
            if not include_sessions and is_session_stem(p.stem):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            try:
                mtime = p.stat().st_mtime
            except OSError:  # pragma: no cover
                mtime = 0.0
            try:
                rel = p.relative_to(vault).as_posix()
            except ValueError:  # pragma: no cover
                rel = p.name
            raw.append((rel, _slugify(p.name), parse_frontmatter(text), mtime, list(_WIKILINK_RE.findall(text))))

    slug_set = {slug for _rel, slug, _fm, _mt, _t in raw}
    inlink: Counter[str] = Counter()
    for _rel, slug, _fm, _mt, targets in raw:
        seen: set[str] = set()
        for t_raw in targets:
            t = _slugify(t_raw)
            if t and t != slug and t in slug_set and t not in seen:
                seen.add(t)
                inlink[t] += 1

    items: list[dict[str, Any]] = []
    for rel, slug, fm, mtime, _targets in raw:
        # Prefer a content-derived date (git-stable) over file mtime (which a clone
        # resets), so the report is reproducible for notes that carry a date.
        la = _parse_iso_date(fm.get("last_access"))
        if la is None:
            for key in ("metadata.date", "date", "created"):
                la = _parse_iso_date(fm.get(key))
                if la is not None:
                    break
        if la is None:
            try:
                la = date.fromtimestamp(mtime)
            except (OSError, OverflowError, ValueError):  # pragma: no cover
                la = today
        age_days = max(0, (today - la).days)
        hits = _as_int(fm.get("hit_count"))
        deg = inlink.get(slug, 0)
        recency = math.exp(-age_days / tau)
        strength = recency + _W_HITS * math.log1p(hits) + _W_LINKS * deg
        items.append(
            {
                "note": rel,
                "strength": round(strength, 3),
                "age_days": age_days,
                "hit_count": hits,
                "inlink_degree": deg,
            }
        )

    fading_all = sorted(
        (i for i in items if i["strength"] < threshold),
        key=lambda i: (i["strength"], i["note"]),
    )
    cap = limit if limit > 0 else len(fading_all)  # limit <= 0 means "no cap"
    fading = fading_all[:cap]
    return {
        "vault": str(vault),
        "scanned_notes": len(raw),
        "tau_days": tau,
        "threshold": threshold,
        "fading": fading,
        "truncated": len(fading_all) > len(fading),
        "note": (
            "Read-only review queue: these notes are going cold (old, rarely linked, "
            "rarely recalled). Nothing is deleted. Archive, refresh, or supersede as you see fit."
        ),
    }


def format_decay(report: dict[str, Any], *, fmt: str = "text") -> str:
    """Render a decay report as ``text`` (human/agent) or ``json``."""
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)

    fading = report["fading"]
    head = (
        f"WikiMoth decay (fading queue)\nvault: {report['vault']}\n"
        f"scanned {report['scanned_notes']} note(s), {len(fading)} fading "
        f"(strength < {report['threshold']}, tau {report['tau_days']:.0f}d).\n"
    )
    if not fading:
        return head + "Nothing is fading. Your memory is warm."
    lines = [head, "Fading (most faded first; strength | age | hits | inlinks):"]
    for i in fading:
        lines.append(
            f"    {i['strength']:.3f}  {i['note']}  "
            f"({i['age_days']}d, {i['hit_count']} hits, {i['inlink_degree']} inlinks)"
        )
    if report.get("truncated"):
        lines.append(f"  (showing {len(fading)}; more are fading, raise --limit)")
    lines.append("\nRead-only: nothing deleted. Archive, refresh, or supersede as you see fit.")
    return "\n".join(lines)


__all__ = ["scan_decay", "format_decay"]
