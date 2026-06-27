# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for ``wikimoth conflicts`` — deterministic contradiction candidate-gen.

Fully offline: writes a small markdown vault to ``tmp_path`` and scans it. No
model, no network — same invariant as the rest of the suite.
"""

from __future__ import annotations

import json

from wikimoth.conflicts import (
    format_conflicts,
    parse_frontmatter,
    scan_conflicts,
)


def _note(vault, name, frontmatter, body=""):
    (vault / f"{name}.md").write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------
def test_parse_frontmatter_nested_and_quoted():
    text = (
        '---\nname: alpha\ndescription: "a \\"quoted\\" desc"\n'
        "metadata:\n  type: session\n  date: 2026-01-15\nvalid_from: 2026-01-01\n---\nbody\n"
    )
    fm = parse_frontmatter(text)
    assert fm["name"] == "alpha"
    assert fm["description"] == 'a "quoted" desc'
    assert fm["metadata.type"] == "session"
    assert fm["metadata.date"] == "2026-01-15"
    assert fm["valid_from"] == "2026-01-01"  # top-level key after a nested block


def test_parse_frontmatter_strips_trailing_comment_but_keeps_hash_tokens():
    fm = parse_frontmatter("---\nstatus: active # current\nlang: C#\n---\n")
    assert fm["status"] == "active"
    assert fm["lang"] == "C#"  # no space before '#', so it survives


def test_parse_frontmatter_ignores_inline_dataview_keys():
    # `key:: value` (Dataview) must NOT be read as a frontmatter `key:` scalar.
    fm = parse_frontmatter("---\nname: x\n---\nbody\nfoo:: bar\n")
    assert "foo" not in fm


# ---------------------------------------------------------------------------
# Core conflict detection
# ---------------------------------------------------------------------------
def test_detects_same_subject_different_value(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "name: a\nabout: paris\npopulation: 2100000")
    _note(v, "b", "name: b\nabout: paris\npopulation: 2200000")
    report = scan_conflicts(v)
    assert report["scanned_notes"] == 2
    assert len(report["conflicts"]) == 1
    c = report["conflicts"][0]
    assert c["subject"] == "paris"
    assert c["predicate"] == "population"
    assert c["kind"] == "numeric"
    assert {val["value"] for val in c["values"]} == {"2100000", "2200000"}


def test_subject_resolves_through_wikilink_and_slugify(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", 'name: a\nabout: "[[Brain Forge]]"\nlead: julian')
    _note(v, "b", "name: b\nentity: brain-forge\nlead: someone-else")
    report = scan_conflicts(v)
    # about=[[Brain Forge]] and entity=brain-forge both slugify to "brain forge"
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["subject"] == "brain forge"
    assert report["conflicts"][0]["predicate"] == "lead"


def test_no_conflict_when_values_agree(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "name: a\nabout: paris\ncountry: france")
    _note(v, "b", "name: b\nabout: paris\ncountry: france")
    assert scan_conflicts(v)["conflicts"] == []


def test_slug_collision_surfaces_conflict(tmp_path):
    # Two physically distinct files collapse to ONE note identity; no subject key,
    # so subject falls back to the shared slug. They disagree on a field.
    v = tmp_path / "vault"
    v.mkdir()
    (v / "Brain-Forge.md").write_text("---\ncapital: rome\n---\n", encoding="utf-8")
    (v / "brain forge.md").write_text("---\ncapital: milan\n---\n", encoding="utf-8")
    report = scan_conflicts(v)
    assert len(report["conflicts"]) == 1
    c = report["conflicts"][0]
    assert c["subject"] == "brain forge"
    assert c["subject_source"] == "note-slug"
    assert c["predicate"] == "capital"


def test_single_note_never_conflicts_with_itself(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "solo", "name: solo\nabout: x\nfield: one")
    assert scan_conflicts(v)["conflicts"] == []


# ---------------------------------------------------------------------------
# Type-aware tolerance
# ---------------------------------------------------------------------------
def test_numeric_tolerance(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 2100000")
    _note(v, "b", "about: x\npop: 2150000")
    assert len(scan_conflicts(v, num_tol=0.0)["conflicts"]) == 1
    assert scan_conflicts(v, num_tol=100000)["conflicts"] == []


def test_date_tolerance(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nfounded: 2020-01-01")
    _note(v, "b", "about: x\nfounded: 2020-01-03")
    assert len(scan_conflicts(v, date_tol_days=0)["conflicts"]) == 1
    assert scan_conflicts(v, date_tol_days=7)["conflicts"] == []


def test_tolerance_not_applied_to_mixed_kinds(tmp_path):
    # numeric vs string in the same bucket: num_tol must NOT suppress (the kinds
    # differ, so comparison falls back to exact canonical).
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nweight: 5.1")
    _note(v, "b", "about: x\nweight: about-five")
    assert len(scan_conflicts(v, num_tol=100)["conflicts"]) == 1


def test_non_zero_padded_date_does_not_crash(tmp_path):
    # date.fromisoformat('2026-6-1') RAISES; the scanner must treat it as a string,
    # not blow up (regression for the design's self-disclosed bug).
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nwhen: 2026-6-1")
    _note(v, "b", "about: x\nwhen: 2026-06-01")
    report = scan_conflicts(v)  # must not raise
    # "2026-6-1" is not a full ISO date -> string; "2026-06-01" -> date; so mixed.
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["kind"] == "mixed"


def test_date_value_with_trailing_text_is_not_truncated(tmp_path):
    # _as_date must require the WHOLE value to be a date, else "2026-01-01 alpha"
    # and "2026-01-01 beta" both truncate to the same date and the conflict hides.
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nrelease: 2026-01-01 alpha")
    _note(v, "b", "about: x\nrelease: 2026-01-01 beta")
    report = scan_conflicts(v)
    assert len(report["conflicts"]) == 1  # genuinely different strings, surfaced
    assert report["conflicts"][0]["kind"] == "string"


def test_non_finite_float_does_not_crash(tmp_path):
    # _as_num must reject inf/nan (float() accepts them) or int() raises and the
    # whole scan aborts.
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nmetric: inf")
    _note(v, "b", "about: x\nmetric: 5")
    report = scan_conflicts(v)  # must not raise OverflowError/ValueError
    assert len(report["conflicts"]) == 1  # 'inf' string vs 5 numeric -> differ


def test_underscore_number_not_collapsed_to_int(tmp_path):
    # YAML treats 1_000 as a string, not 1000; do not silently merge them.
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\ncount: 1_000")
    _note(v, "b", "about: x\ncount: 1000")
    assert len(scan_conflicts(v)["conflicts"]) == 1


def test_quoted_value_with_trailing_comment(tmp_path):
    # A quoted scalar followed by a comment must unquote to the same value as the
    # bare scalar, not keep the literal quotes (which would be a phantom conflict).
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", 'about: x\ntitle: "Brain Forge" # legacy')
    _note(v, "b", "about: x\ntitle: Brain Forge")
    assert scan_conflicts(v)["conflicts"] == []


def test_mis_indented_line_after_scalar_is_ignored(tmp_path):
    # An over-indented line following a scalar key (no container parent) must not
    # become a top-level predicate.
    fm = parse_frontmatter("---\nstatus: active\n  weight: 5\n---\n")
    assert "weight" not in fm
    assert fm["status"] == "active"


# ---------------------------------------------------------------------------
# Temporal precision (overlapping = real conflict, disjoint = succession)
# ---------------------------------------------------------------------------
def test_valid_time_overlapping_is_high_confidence(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 100\nvalid_from: 2020-01-01\nvalid_to: 2026-01-01")
    _note(v, "b", "about: x\npop: 200\nvalid_from: 2023-01-01")
    c = scan_conflicts(v)["conflicts"][0]
    assert c["valid_time"] == "overlapping"
    assert c["confidence"] == "high"


def test_valid_time_disjoint_is_low_confidence(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 100\nvalid_from: 2020-01-01\nvalid_to: 2023-01-01")
    _note(v, "b", "about: x\npop: 200\nvalid_from: 2023-01-01")
    c = scan_conflicts(v)["conflicts"][0]
    assert c["valid_time"] == "disjoint"  # legitimate succession, likely a supersede
    assert c["confidence"] == "low"


def test_valid_time_unknown_is_medium(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 100")
    _note(v, "b", "about: x\npop: 200")
    c = scan_conflicts(v)["conflicts"][0]
    assert c["valid_time"] == "unknown"
    assert c["confidence"] == "medium"


def test_same_value_overlap_does_not_force_overlapping(tmp_path):
    # Two SAME-value notes overlap, but the DIFFERING values are a disjoint
    # succession: temporal status must judge only differing-value pairs.
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: org\nrole: ceo\nvalid_from: 2020-01-01\nvalid_to: 2022-01-01")
    _note(v, "b", "about: org\nrole: ceo\nvalid_from: 2021-01-01\nvalid_to: 2022-01-01")
    _note(v, "c", "about: org\nrole: cto\nvalid_from: 2022-01-01\nvalid_to: 2024-01-01")
    c = scan_conflicts(v)["conflicts"][0]
    assert len(c["values"]) == 3  # 3-way group exercised
    assert c["valid_time"] == "disjoint"  # ceo/ceo overlap must NOT promote to high
    assert c["confidence"] == "low"


def test_dated_vs_undated_pair_is_unknown_not_high(tmp_path):
    # One side has no valid-time: we cannot prove overlap, so don't claim 'high'.
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: org\npop: 100")  # undated
    _note(v, "b", "about: org\npop: 200\nvalid_from: 2025-01-01")
    c = scan_conflicts(v)["conflicts"][0]
    assert c["valid_time"] == "unknown"
    assert c["confidence"] == "medium"


def test_same_basename_in_subdirs_both_surfaced(tmp_path):
    # Distinct files sharing a basename in different subdirs must NOT collapse:
    # the dedup key is the vault-relative path, not the basename.
    v = tmp_path / "vault"
    v.mkdir()
    (v / "y2024").mkdir()
    (v / "y2025").mkdir()
    (v / "y2024" / "status.md").write_text("---\nabout: proj\nstage: alpha\n---\n", encoding="utf-8")
    (v / "y2025" / "status.md").write_text("---\nabout: proj\nstage: beta\n---\n", encoding="utf-8")
    report = scan_conflicts(v)
    assert report["scanned_notes"] == 2
    assert len(report["conflicts"]) == 1
    files = {val["file"] for val in report["conflicts"][0]["values"]}
    assert files == {"y2024/status.md", "y2025/status.md"}


def test_crlf_note_is_parsed(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_bytes(b"---\r\nabout: x\r\nceo: alice\r\n---\r\nbody\r\n")
    (v / "b.md").write_bytes(b"---\r\nabout: x\r\nceo: bob\r\n---\r\nbody\r\n")
    report = scan_conflicts(v)
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["predicate"] == "ceo"
    assert {val["value"] for val in report["conflicts"][0]["values"]} == {"alice", "bob"}


# ---------------------------------------------------------------------------
# Ignore set / all_keys / inline
# ---------------------------------------------------------------------------
def test_structural_keys_ignored_by_default(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nmetadata:\n  session_id: AAA")
    _note(v, "b", "about: x\nmetadata:\n  session_id: BBB")
    assert scan_conflicts(v)["conflicts"] == []  # session_id is bookkeeping
    assert len(scan_conflicts(v, all_keys=True)["conflicts"]) == 1  # forced in


def test_session_notes_skipped_by_default(tmp_path):
    # session-* notes are operational logs, not curated facts.
    v = tmp_path / "vault"
    v.mkdir()
    (v / "session-2026-01-01-aaaa.md").write_text("---\nabout: x\nstage: alpha\n---\n", encoding="utf-8")
    (v / "session-2026-01-02-bbbb.md").write_text("---\nabout: x\nstage: beta\n---\n", encoding="utf-8")
    rep = scan_conflicts(v)
    assert rep["scanned_notes"] == 0
    assert rep["conflicts"] == []
    rep2 = scan_conflicts(v, include_sessions=True)
    assert rep2["scanned_notes"] == 2
    assert len(rep2["conflicts"]) == 1


def test_case_insensitive_merges_string_values(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nteam: Engineering")
    _note(v, "b", "about: x\nteam: engineering")
    assert len(scan_conflicts(v)["conflicts"]) == 1  # case-sensitive default
    assert scan_conflicts(v, case_insensitive=True)["conflicts"] == []


def test_status_field_not_flagged_by_default(tmp_path):
    # A supersede sets status:superseded; that is expected state, not a conflict.
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\nstatus: active")
    _note(v, "b", "about: x\nstatus: superseded")
    assert scan_conflicts(v)["conflicts"] == []


def test_include_inline_reads_dataview_fields(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x", body="ceo:: alice")
    _note(v, "b", "about: x", body="ceo:: bob")
    assert scan_conflicts(v)["conflicts"] == []  # off by default
    report = scan_conflicts(v, include_inline=True)
    assert len(report["conflicts"]) == 1
    assert report["conflicts"][0]["predicate"] == "ceo"


# ---------------------------------------------------------------------------
# Determinism + formatting + edge cases
# ---------------------------------------------------------------------------
def test_report_is_deterministic(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 1\nceo: alice")
    _note(v, "b", "about: x\npop: 2\nceo: bob")
    assert scan_conflicts(v) == scan_conflicts(v)
    # and the rendered text is stable too
    r = scan_conflicts(v)
    assert format_conflicts(r) == format_conflicts(r)


def test_json_format_is_valid_json(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 1")
    _note(v, "b", "about: x\npop: 2")
    parsed = json.loads(format_conflicts(scan_conflicts(v), fmt="json"))
    assert parsed["conflicts"][0]["subject"] == "x"


def test_missing_vault_is_empty_report(tmp_path):
    report = scan_conflicts(tmp_path / "nope")
    assert report["scanned_notes"] == 0
    assert report["conflicts"] == []
    assert "No contradicting" in format_conflicts(report)


def test_clean_vault_reports_consistent(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "name: a\nabout: x\nfield: same")
    _note(v, "b", "name: b\nabout: x\nfield: same")
    assert "consistent" in format_conflicts(scan_conflicts(v)).lower()


# ---------------------------------------------------------------------------
# CLI path (argparse wiring + the stdout reconfigure guard)
# ---------------------------------------------------------------------------
def test_cli_conflicts_json(tmp_path, capfd):
    from wikimoth.cli import main

    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: acme\nceo: alice")
    _note(v, "b", "about: acme\nceo: bob")
    rc = main(["conflicts", "--vault", str(v), "--format", "json"])
    assert rc == 0
    data = json.loads(capfd.readouterr().out)
    assert data["conflicts"][0]["predicate"] == "ceo"


def test_cli_conflicts_num_tol_flag(tmp_path, capfd):
    from wikimoth.cli import main

    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 2100000")
    _note(v, "b", "about: x\npop: 2150000")
    rc = main(["conflicts", "--vault", str(v), "--format", "json", "--num-tol", "100000"])
    assert rc == 0
    assert json.loads(capfd.readouterr().out)["conflicts"] == []


def test_cli_conflicts_missing_vault_returns_1(tmp_path, capfd):
    from wikimoth.cli import main

    rc = main(["conflicts", "--vault", str(tmp_path / "nope")])
    assert rc == 1


def test_text_scaffolding_is_ascii_safe(tmp_path):
    # ASCII-only note content -> the whole rendered text must encode on cp1252
    # (regression: the formatter's own markers/arrows must never be non-ASCII).
    v = tmp_path / "vault"
    v.mkdir()
    _note(v, "a", "about: x\npop: 1\nvalid_from: 2020-01-01")
    _note(v, "b", "about: x\npop: 2\nvalid_from: 2021-01-01")
    text = format_conflicts(scan_conflicts(v))
    text.encode("cp1252")  # must not raise UnicodeEncodeError
