# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for ``wikimoth supersede`` — the invalidate-don't-delete write op. Offline."""

from __future__ import annotations

from datetime import date

import pytest

from wikimoth.frontmatter import parse_frontmatter
from wikimoth.supersede import SupersedeError, format_result, supersede

AS_OF = date(2026, 6, 20)


def _vault(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "old-project.md").write_text(
        "---\nname: old-project\ndescription: \"the old name\"\nmetadata:\n  type: note\n  date: 2026-01-15\n---\n"
        "# Old project\nbody about the old project\n",
        encoding="utf-8",
    )
    (v / "new-project.md").write_text(
        "---\nname: new-project\n---\n# New project\nbody about the new project\n",
        encoding="utf-8",
    )
    return v


def test_supersede_writes_top_level_keys(tmp_path):
    v = _vault(tmp_path)
    res = supersede(v, "old-project", "new-project", as_of=AS_OF, reason="renamed")
    assert res["changed"] is True
    fm = parse_frontmatter((v / "old-project.md").read_text(encoding="utf-8"))
    assert fm["superseded_by"] == "[[new-project]]"
    assert fm["status"] == "superseded"
    assert fm["valid_to"] == "2026-06-20"
    assert fm["invalidated_at"] == "2026-06-20"
    assert fm["supersession_reason"] == "renamed"
    # nested metadata preserved, NOT clobbered
    assert fm["metadata.type"] == "note"
    assert fm["metadata.date"] == "2026-01-15"


def test_supersede_keeps_the_file_and_body(tmp_path):
    v = _vault(tmp_path)
    supersede(v, "old-project", "new-project", as_of=AS_OF)
    text = (v / "old-project.md").read_text(encoding="utf-8")
    assert (v / "old-project.md").exists()  # never deleted
    assert "body about the old project" in text  # body intact


def test_superseded_by_is_a_real_wikilink_edge(tmp_path):
    # the [[new-project]] must survive as a graph edge (lint sees it resolve).
    from wikimoth.lint import scan_lint

    v = _vault(tmp_path)
    supersede(v, "old-project", "new-project", as_of=AS_OF)
    rep = scan_lint(v)
    assert rep["broken_links"] == []  # the edge resolves to new-project
    # old-project -> new-project is a single edge (1 edge = not a transitive chain)
    assert rep["supersession_chains"] == []


def test_reverse_stamps_replaces_on_new(tmp_path):
    v = _vault(tmp_path)
    supersede(v, "old-project", "new-project", as_of=AS_OF, reverse=True)
    fm_new = parse_frontmatter((v / "new-project.md").read_text(encoding="utf-8"))
    assert fm_new["replaces"] == "[[old-project]]"


def test_undo_removes_supersession(tmp_path):
    v = _vault(tmp_path)
    supersede(v, "old-project", "new-project", as_of=AS_OF, reason="x")
    res = supersede(v, "old-project", "new-project", undo=True)
    assert res["action"] == "undo"
    fm = parse_frontmatter((v / "old-project.md").read_text(encoding="utf-8"))
    for k in ("superseded_by", "status", "valid_to", "invalidated_at", "supersession_reason"):
        assert k not in fm
    assert fm["name"] == "old-project"  # untouched keys remain


def test_undo_on_non_superseded_errors(tmp_path):
    v = _vault(tmp_path)
    with pytest.raises(SupersedeError):
        supersede(v, "old-project", "new-project", undo=True)


def test_dry_run_writes_nothing(tmp_path):
    v = _vault(tmp_path)
    before = (v / "old-project.md").read_text(encoding="utf-8")
    res = supersede(v, "old-project", "new-project", as_of=AS_OF, dry_run=True)
    assert res["dry_run"] is True
    assert "superseded_by" in res["diff"]
    assert (v / "old-project.md").read_text(encoding="utf-8") == before  # unchanged


def test_self_supersede_errors(tmp_path):
    v = _vault(tmp_path)
    with pytest.raises(SupersedeError):
        supersede(v, "old-project", "old-project", as_of=AS_OF)


def test_unresolved_note_errors(tmp_path):
    v = _vault(tmp_path)
    with pytest.raises(SupersedeError):
        supersede(v, "ghost", "new-project", as_of=AS_OF)


def test_session_note_refused(tmp_path):
    v = _vault(tmp_path)
    (v / "session-2026-01-01-aa.md").write_text("---\nname: s\n---\nlog\n", encoding="utf-8")
    with pytest.raises(SupersedeError):
        supersede(v, "session-2026-01-01-aa", "new-project", as_of=AS_OF)


def test_re_supersede_overwrites(tmp_path):
    v = _vault(tmp_path)
    (v / "newer.md").write_text("---\nname: newer\n---\nbody\n", encoding="utf-8")
    supersede(v, "old-project", "new-project", as_of=AS_OF)
    supersede(v, "old-project", "newer", as_of=AS_OF)  # re-point
    fm = parse_frontmatter((v / "old-project.md").read_text(encoding="utf-8"))
    assert fm["superseded_by"] == "[[newer]]"  # exactly one, re-pointed


def test_vault_containment_refused(tmp_path):
    # An OLD ref resolving outside the vault must be refused (no arbitrary writes).
    v = _vault(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: outside\n---\nsecret\n", encoding="utf-8")
    with pytest.raises(SupersedeError):
        supersede(v, str(outside), "new-project", as_of=AS_OF)
    assert "secret" in outside.read_text(encoding="utf-8")  # untouched


def test_tags_block_list_preserved(tmp_path):
    # Managed keys go at the TOP, so a tags: block list is never split/orphaned.
    v = tmp_path / "v"
    v.mkdir()
    (v / "a.md").write_text(
        "---\ntitle: A\ntags:\n  - alpha\n  - beta\nmetadata:\n  type: note\n---\nbody\n",
        encoding="utf-8",
    )
    (v / "b.md").write_text("---\nname: b\n---\nx\n", encoding="utf-8")
    supersede(v, "a", "b", as_of=AS_OF)
    text = (v / "a.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    fm = parse_frontmatter(text)
    assert fm["superseded_by"] == "[[b]]"
    assert fm["metadata.type"] == "note"
    assert "tags:\n  - alpha\n  - beta" in text  # block list contiguous, intact


def test_ambiguous_new_refused(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "old.md").write_text("---\nname: old\n---\nx\n", encoding="utf-8")
    (v / "Dup-Note.md").write_text("---\nname: d1\n---\nx\n", encoding="utf-8")
    (v / "dup note.md").write_text("---\nname: d2\n---\nx\n", encoding="utf-8")
    with pytest.raises(SupersedeError):  # [[Dup-Note]] would resolve ambiguously
        supersede(v, "old", str(v / "Dup-Note.md"), as_of=AS_OF)


def test_note_without_frontmatter_gets_one(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    (v / "bare.md").write_text("just a body, no frontmatter\n", encoding="utf-8")
    (v / "new.md").write_text("---\nname: new\n---\nx\n", encoding="utf-8")
    supersede(v, "bare", "new", as_of=AS_OF)
    fm = parse_frontmatter((v / "bare.md").read_text(encoding="utf-8"))
    assert fm["superseded_by"] == "[[new]]"
    assert "just a body, no frontmatter" in (v / "bare.md").read_text(encoding="utf-8")
