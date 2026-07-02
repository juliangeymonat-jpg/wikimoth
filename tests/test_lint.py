# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for ``wikimoth lint`` — deterministic vault hygiene. Fully offline."""

from __future__ import annotations

import json
from datetime import date

from wikimoth.lint import format_lint, scan_lint


def _w(vault, name, content):
    p = vault / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_broken_link_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\n---\nSee [[ghost]] and [[b]].\n")
    _w(v, "b.md", "---\nname: b\n---\nReal note.\n")
    rep = scan_lint(v)
    targets = {b["target"] for b in rep["broken_links"]}
    assert targets == {"ghost"}  # [[b]] resolves, [[ghost]] does not


def test_broken_link_from_superseded_by(tmp_path):
    # a mis-stamped superseded_by [[X]] where X is missing shows as a broken link.
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", '---\nname: old\nsuperseded_by: "[[missing-new]]"\n---\nbody\n')
    rep = scan_lint(v)
    assert any(b["target"] == "missing-new" for b in rep["broken_links"])


def test_orphan_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "hub.md", "---\nname: hub\n---\nLinks [[leaf]].\n")
    _w(v, "leaf.md", "---\nname: leaf\n---\nLinked-to.\n")
    _w(v, "lonely.md", "---\nname: lonely\n---\nNo links in or out.\n")
    rep = scan_lint(v)
    assert rep["orphans"] == ["lonely.md"]  # hub has outlink, leaf has inlink


def test_duplicate_slug_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "Brain-Forge.md", "---\nname: bf1\n---\nbody one\n")
    _w(v, "brain forge.md", "---\nname: bf2\n---\nbody two\n")
    rep = scan_lint(v)
    assert len(rep["duplicate_slugs"]) == 1
    d = rep["duplicate_slugs"][0]
    assert d["slug"] == "brain forge"
    assert d["files"] == ["Brain-Forge.md", "brain forge.md"]


def test_stub_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "empty.md", "---\nname: empty\n---\n")
    _w(v, "full.md", "---\nname: full\n---\nHas content.\n")
    rep = scan_lint(v)
    assert rep["stubs"] == ["empty.md"]


def test_stale_valid_to_and_expires(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", "---\nname: old\nvalid_to: 2000-01-01\n---\nbody\n")
    _w(v, "exp.md", "---\nname: exp\nexpires: 2000-06-01\n---\nbody\n")
    _w(v, "live.md", "---\nname: live\nvalid_to: 2099-01-01\n---\nbody\n")
    rep = scan_lint(v, today=date(2026, 1, 1))
    reasons = {(s["note"], s["reason"]) for s in rep["stale"]}
    assert ("old.md", "valid_to") in reasons
    assert ("exp.md", "expires") in reasons
    assert all(s["note"] != "live.md" for s in rep["stale"])  # future validity = not stale


def test_supersession_chain_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", '---\nname: a\nsuperseded_by: "[[b]]"\n---\nbody\n')
    _w(v, "b.md", '---\nname: b\nsuperseded_by: "[[c]]"\n---\nbody\n')
    _w(v, "c.md", "---\nname: c\n---\ncurrent\n")
    rep = scan_lint(v)
    assert rep["supersession_chains"] == [{"chain": ["a", "b", "c"]}]
    assert rep["supersession_cycles"] == []


def test_supersession_cycle_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", '---\nname: a\nsuperseded_by: "[[b]]"\n---\nbody\n')
    _w(v, "b.md", '---\nname: b\nsuperseded_by: "[[a]]"\n---\nbody\n')
    rep = scan_lint(v)
    assert rep["supersession_cycles"] == [{"cycle": ["a", "b", "a"]}]  # canonical
    assert rep["supersession_chains"] == []


def test_cycle_with_upstream_head_reported_once(tmp_path):
    # a -> b -> c -> b: the b<->c cycle must be reported EXACTLY once (regression:
    # the old two-pass logic double/triple-reported head-reachable cycles).
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", '---\nname: a\nsuperseded_by: "[[b]]"\n---\nx\n')
    _w(v, "b.md", '---\nname: b\nsuperseded_by: "[[c]]"\n---\nx\n')
    _w(v, "c.md", '---\nname: c\nsuperseded_by: "[[b]]"\n---\nx\n')
    rep = scan_lint(v)
    assert rep["supersession_cycles"] == [{"cycle": ["b", "c", "b"]}]


def test_multi_head_into_cycle_reported_once(tmp_path):
    # two heads (a, b) both feed a c<->d cycle: still one cycle row.
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", '---\nname: a\nsuperseded_by: "[[c]]"\n---\nx\n')
    _w(v, "b.md", '---\nname: b\nsuperseded_by: "[[c]]"\n---\nx\n')
    _w(v, "c.md", '---\nname: c\nsuperseded_by: "[[d]]"\n---\nx\n')
    _w(v, "d.md", '---\nname: d\nsuperseded_by: "[[c]]"\n---\nx\n')
    rep = scan_lint(v)
    assert rep["supersession_cycles"] == [{"cycle": ["c", "d", "c"]}]


def test_dangling_only_note_is_not_orphan(tmp_path):
    # A note linking only to a missing note authored an outlink: it is a broken
    # link, NOT an orphan (it must not appear in both buckets).
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "x.md", "---\nname: x\n---\n[[nowhere]]\n")
    rep = scan_lint(v)
    assert rep["orphans"] == []
    assert any(b["target"] == "nowhere" for b in rep["broken_links"])


def test_replaces_audit_link_not_flagged_broken(tmp_path):
    # `replaces:` points backward to a possibly-deleted note (audit), not a live edge.
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "new.md", '---\nname: new\nreplaces: "[[deleted-old]]"\n---\nReal body.\n')
    rep = scan_lint(v)
    assert all(b["target"] != "deleted-old" for b in rep["broken_links"])


def test_clean_vault_no_issues(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\n---\nLinks [[b]].\n")
    _w(v, "b.md", "---\nname: b\n---\nLinks [[a]].\n")
    rep = scan_lint(v)
    assert rep["issues"] == 0
    assert "No hygiene issues" in format_lint(rep)


def test_sessions_skipped_for_orphan_stub_by_default(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    # a lonely session note: would be an orphan+stub, but sessions are skipped
    _w(v, "session-2026-01-01-aa.md", "---\nname: s\n---\n")
    rep = scan_lint(v)
    assert rep["orphans"] == []
    assert rep["stubs"] == []
    rep2 = scan_lint(v, include_sessions=True)
    assert "session-2026-01-01-aa.md" in rep2["orphans"]


def test_report_is_deterministic_and_json(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\n---\n[[ghost]] [[b]]\n")
    _w(v, "b.md", "---\nname: b\n---\nx\n")
    assert scan_lint(v) == scan_lint(v)
    parsed = json.loads(format_lint(scan_lint(v), fmt="json"))
    assert parsed["scanned_notes"] == 2
