# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for ``wikimoth dedup`` — deterministic near/exact-duplicate detection. Offline."""

from __future__ import annotations

import json

from wikimoth.dedup import format_dedup, scan_dedup


def _w(v, name, body):
    (v / name).write_text(f"---\nname: {name[:-3]}\n---\n{body}\n", encoding="utf-8")


_LOREM = (
    "The quarterly planning meeting covered the roadmap for the analytics platform "
    "and the migration of the ingestion pipeline to the new cluster with autoscaling."
)


def test_exact_duplicate_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", _LOREM)  # identical body
    _w(v, "c.md", "Totally unrelated content about gardening tomatoes in spring.")
    rep = scan_dedup(v)
    assert len(rep["exact_duplicates"]) == 1
    assert rep["exact_duplicates"][0]["files"] == ["a.md", "b.md"]


def test_exact_duplicate_ignores_frontmatter_and_whitespace(tmp_path):
    # same body, different frontmatter + whitespace -> still exact (normalised body).
    v = tmp_path / "v"
    v.mkdir()
    (v / "a.md").write_text(f"---\nname: a\ndate: 2026-01-01\n---\n{_LOREM}\n", encoding="utf-8")
    (v / "b.md").write_text(f"---\nname: b\n---\n  {_LOREM}   \n\n", encoding="utf-8")
    rep = scan_dedup(v)
    assert len(rep["exact_duplicates"]) == 1


def test_near_duplicate_detected(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", _LOREM + " A few extra words were appended here at the very end.")
    rep = scan_dedup(v, threshold=0.6)
    pairs = {(p["a"], p["b"]) for p in rep["near_duplicates"]}
    assert ("a.md", "b.md") in pairs
    assert rep["near_duplicates"][0]["similarity"] >= 0.6


def test_distinct_notes_not_flagged(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", "An entirely different note about baking sourdough bread at home over weekends.")
    rep = scan_dedup(v)
    assert rep["exact_duplicates"] == []
    assert rep["near_duplicates"] == []


def test_threshold_controls_near(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", _LOREM + " plus a moderate amount of additional distinct trailing text here.")
    lo = scan_dedup(v, threshold=0.5)
    hi = scan_dedup(v, threshold=0.99)
    assert len(lo["near_duplicates"]) >= len(hi["near_duplicates"])


def test_exact_not_double_reported_as_near(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", _LOREM)
    rep = scan_dedup(v, threshold=0.5)
    assert len(rep["exact_duplicates"]) == 1
    assert rep["near_duplicates"] == []  # the exact pair is not also a near pair


def test_empty_body_ignored(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "a.md").write_text("---\nname: a\n---\n", encoding="utf-8")
    (v / "b.md").write_text("---\nname: b\n---\n   \n", encoding="utf-8")
    rep = scan_dedup(v)
    assert rep["exact_duplicates"] == []  # empty bodies are not "duplicates"
    assert rep["scanned_notes"] == 0


def test_sessions_skipped_by_default(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "session-2026-01-01-aa.md").write_text(f"---\nname: s1\n---\n{_LOREM}\n", encoding="utf-8")
    (v / "session-2026-01-02-bb.md").write_text(f"---\nname: s2\n---\n{_LOREM}\n", encoding="utf-8")
    assert scan_dedup(v)["exact_duplicates"] == []
    assert len(scan_dedup(v, include_sessions=True)["exact_duplicates"]) == 1


def test_deterministic_and_json(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", _LOREM)
    _w(v, "c.md", _LOREM + " trailing variation words appended to make a near duplicate pair.")
    assert scan_dedup(v) == scan_dedup(v)  # byte-identical (seeded hashing)
    parsed = json.loads(format_dedup(scan_dedup(v), fmt="json"))
    assert parsed["scanned_notes"] == 3


def test_threshold_clamped(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", _LOREM)
    assert scan_dedup(v, threshold=1.5)["threshold"] == 1.0   # not a misleading >1 header
    assert scan_dedup(v, threshold=-1)["threshold"] == 0.0


def test_near_pairs_capped_and_flagged(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    for i in range(10):  # 10 near-identical notes -> C(10,2)=45 near pairs
        _w(v, f"n{i}.md", _LOREM + f" trailing variation token number {i} appended here.")
    capped = scan_dedup(v, threshold=0.5, max_pairs=5)
    assert len(capped["near_duplicates"]) == 5
    assert capped["truncated_near"] is True
    assert "truncated" in format_dedup(capped)
    full = scan_dedup(v, threshold=0.5)  # default cap 1000 -> no truncation
    assert full["truncated_near"] is False


def test_clean_vault_text(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", _LOREM)
    _w(v, "b.md", "Distinct content about hiking trails in the northern mountains during autumn.")
    assert "No duplicate" in format_dedup(scan_dedup(v))
