# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for ``wikimoth decay`` — the fading-queue review report. Offline + deterministic."""

from __future__ import annotations

import json
from datetime import date

from wikimoth.decay import format_decay, scan_decay

TODAY = date(2026, 6, 27)


def _w(v, name, body="body text here", **fm):
    lines = [f"name: {name[:-3]}"]
    for k, val in fm.items():
        lines.append(f"{k}: {val}")
    (v / name).write_text("---\n" + "\n".join(lines) + f"\n---\n{body}\n", encoding="utf-8")


def test_old_cold_note_fades(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", last_access="2020-01-01")   # ancient, no hits, no inlinks
    _w(v, "fresh.md", last_access="2026-06-26")  # touched yesterday
    rep = scan_decay(v, today=TODAY)
    faded = {f["note"] for f in rep["fading"]}
    assert "old.md" in faded
    assert "fresh.md" not in faded


def test_inlinks_protect_from_fading(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    # old hub linked by 3 notes should NOT fade despite age
    _w(v, "hub.md", last_access="2020-01-01")
    for i in range(3):
        _w(v, f"src{i}.md", body="see [[hub]] for details", last_access="2026-06-26")
    rep = scan_decay(v, today=TODAY)
    assert "hub.md" not in {f["note"] for f in rep["fading"]}  # connectivity protects


def test_hits_protect_from_fading(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", last_access="2020-01-01", hit_count=50)  # old but heavily recalled
    _w(v, "b.md", last_access="2020-01-01")                # old and cold
    rep = scan_decay(v, today=TODAY)
    faded = {f["note"] for f in rep["fading"]}
    assert "b.md" in faded
    assert "a.md" not in faded  # high hit_count keeps strength up


def test_signals_reported(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", last_access="2020-01-01", hit_count=2)
    f = scan_decay(v, today=TODAY)["fading"][0]
    assert f["note"] == "old.md"
    assert f["age_days"] > 2000
    assert f["hit_count"] == 2
    assert "strength" in f and "inlink_degree" in f


def test_tau_controls_decay(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", last_access="2026-03-01")  # ~118 days old at TODAY
    short = scan_decay(v, tau_days=30, today=TODAY)   # fast decay -> fades
    long = scan_decay(v, tau_days=3650, today=TODAY)  # slow decay -> warm
    assert "a.md" in {f["note"] for f in short["fading"]}
    assert "a.md" not in {f["note"] for f in long["fading"]}


def test_threshold_controls_queue(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", last_access="2026-04-01")  # ~87 days
    lo = scan_decay(v, threshold=0.1, today=TODAY)
    hi = scan_decay(v, threshold=0.9, today=TODAY)
    assert len(hi["fading"]) >= len(lo["fading"])


def test_limit_caps_and_flags_truncation(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    for i in range(5):
        _w(v, f"old{i}.md", last_access="2020-01-01")
    rep = scan_decay(v, limit=2, today=TODAY)
    assert len(rep["fading"]) == 2
    assert rep["truncated"] is True
    assert "more are fading" in format_decay(rep)


def test_sessions_skipped_by_default(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "session-2020-01-01-aa.md").write_text("---\nname: s\nlast_access: 2020-01-01\n---\nlog\n", encoding="utf-8")
    assert scan_decay(v, today=TODAY)["fading"] == []
    assert scan_decay(v, today=TODAY, include_sessions=True)["fading"]


def test_no_fading_message(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", last_access="2026-06-26")
    assert "warm" in format_decay(scan_decay(v, today=TODAY)).lower()


def test_limit_zero_means_unlimited(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    for i in range(3):
        _w(v, f"o{i}.md", last_access="2000-01-01")
    rep = scan_decay(v, limit=0, today=TODAY)
    assert len(rep["fading"]) == 3       # not capped
    assert rep["truncated"] is False     # honest: nothing hidden
    assert "warm" not in format_decay(rep).lower()  # does not falsely claim warm


def test_content_date_fallback_is_git_stable(tmp_path):
    # No last_access, but a content-derived metadata.date -> deterministic age
    # (not the file mtime, which a clone would reset).
    v = tmp_path / "v"
    v.mkdir()
    (v / "a.md").write_text(
        "---\nname: a\nmetadata:\n  date: 2000-01-01\n---\nold session content.\n", encoding="utf-8"
    )
    f = scan_decay(v, today=TODAY)["fading"]
    assert len(f) == 1 and f[0]["note"] == "a.md"
    assert f[0]["age_days"] > 9000  # derived from metadata.date, ~26 years


def test_deterministic_and_json(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", last_access="2020-01-01")
    _w(v, "b.md", last_access="2021-01-01")
    assert scan_decay(v, today=TODAY) == scan_decay(v, today=TODAY)
    parsed = json.loads(format_decay(scan_decay(v, today=TODAY), fmt="json"))
    assert parsed["scanned_notes"] == 2
