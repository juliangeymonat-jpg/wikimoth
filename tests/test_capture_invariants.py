# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Regression tests for the three adversarial-review findings (2026-06-21).

1. BLOCKER — echoed text (prompt/description/file paths) must not smuggle a
   ``[[wikilink]]`` edge past the code-computed link set.
2. SHOULD-FIX — the hook must exit 0 even on pathological stdin (RecursionError).
3. SHOULD-FIX — install merge must not crash on a non-dict ``hooks`` value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wikimoth.capture.config import CaptureConfig
from wikimoth.capture.buffer import SessionBuffer
from wikimoth.capture.note import build_note, write_session_note, _deactivate_wikilinks
from wikimoth.capture import install as inst

REPO = Path(__file__).resolve().parents[1]          # .


# --------------------------------------------------------------------------
# Finding 1 — no smuggled edges from echoed user/tool text
# --------------------------------------------------------------------------

def test_deactivate_wikilinks_breaks_all_runs():
    assert "[[" not in _deactivate_wikilinks("a [[x]] b")
    assert "]]" not in _deactivate_wikilinks("a [[x]] b")
    # triple+ runs too
    out = _deactivate_wikilinks("[[[x]]] and [[[[y]]]]")
    assert "[[" not in out and "]]" not in out
    # single brackets preserved (arr[i])
    assert _deactivate_wikilinks("arr[i] = x") == "arr[i] = x"


def test_prompt_brackets_do_not_become_edges():
    events = [
        {"event": "SessionStart", "ts": "2026-06-21T10:00:00",
         "session_id": "feedface0001", "cwd": "/tmp", "source": "startup"},
        {"event": "UserPromptSubmit", "ts": "2026-06-21T10:01:00",
         "prompt": "see [[aaa]] now and edit [[bbb|alias]] too"},
        {"event": "PostToolUse", "ts": "2026-06-21T10:02:00",
         "tool": "Edit", "file_path": "/repo/[[ccc]].md"},
    ]
    # compute_links authors NOTHING here (no prev session, no file→note match,
    # single-token titles too short to phrase-match).
    stem, content = build_note(
        events, session_id="feedface0001", note_index={}, session_stems=[],
        config=CaptureConfig(vault_dir=Path(".")),
    )
    # The note must contain NO [[...]] at all (the Links section says "none").
    assert "[[aaa]]" not in content
    assert "[[bbb" not in content
    assert "[[ccc]]" not in content
    assert "_No links resolved this session._" in content


def test_prompt_bracket_edge_not_walked_by_real_graph(tmp_path):
    from wikimoth.retrieval import GraphRetriever  # noqa: F401
    from wikimoth import MemoryRAG

    vault = tmp_path / "vault"
    vault.mkdir()
    # A content note that a smuggled [[aaa]] *would* connect to.
    (vault / "aaa.md").write_text(
        "---\nname: aaa\ndescription: \"target\"\n---\nUnrelated content body.\n",
        encoding="utf-8",
    )
    cfg = CaptureConfig(vault_dir=vault)
    buf = SessionBuffer.for_session(cfg, "feedface0002")
    for e in [
        {"event": "SessionStart", "ts": "2026-06-21T10:00:00",
         "session_id": "feedface0002", "cwd": str(tmp_path), "source": "startup"},
        {"event": "UserPromptSubmit", "ts": "2026-06-21T10:01:00",
         "prompt": "please see [[aaa]] right now"},
    ]:
        buf.append(e)
    write_session_note(cfg, "feedface0002")

    rag = MemoryRAG(exclude_content=())
    rag.index(vault)
    # aaa.md has no links; the session note's [[aaa]] was neutralised → 0 edges.
    assert rag.retriever.edge_count() == 0


# --------------------------------------------------------------------------
# Finding 2 — hook never breaks the session, even on pathological stdin
# --------------------------------------------------------------------------

def _hook_env(vault: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["WIKIMOTH_VAULT"] = str(vault)
    return env


@pytest.mark.parametrize(
    "stdin_text",
    ["[" * 100000, "{" * 50000, "", "not json", "null"],
    ids=["deep_brackets", "deep_braces", "empty", "not_json", "null"],
)
def test_hook_exits_zero_on_pathological_stdin(tmp_path, stdin_text):
    proc = subprocess.run(
        [sys.executable, "-m", "wikimoth.capture.hook", "Stop"],
        input=stdin_text, text=True, capture_output=True,
        env=_hook_env(tmp_path / "vault"), timeout=60,
    )
    assert proc.returncode == 0


# --------------------------------------------------------------------------
# Finding 3 — install merge survives a malformed settings.json shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad_hooks", [[], "weird", 42, None])
def test_merge_hooks_survives_non_dict_hooks(bad_hooks):
    out = inst.merge_hooks({"hooks": bad_hooks}, python_exe="C:/py/python.exe")
    assert isinstance(out["hooks"], dict)
    assert "SessionStart" in out["hooks"]


def test_merge_hooks_survives_non_list_event_value():
    out = inst.merge_hooks({"hooks": {"Stop": "scalar"}}, python_exe="C:/py/python.exe")
    assert isinstance(out["hooks"]["Stop"], list)
    # our Stop hook is present; the bogus scalar is dropped, not char-shredded
    cmds = [h.get("command", "") for g in out["hooks"]["Stop"] for h in g.get("hooks", [])]
    assert any(inst.MARKER in c for c in cmds)


def test_remove_hooks_survives_non_dict_hooks():
    assert inst.remove_hooks({"hooks": "weird"}) == {}  # nothing to keep, hooks dropped


def test_install_preserves_other_env_keys(tmp_path):
    sp = tmp_path / ".claude" / "settings.json"
    inst.save_settings(sp, {"env": {"FOO": "bar"}})
    inst.install(sp, python_exe="C:/py/python.exe", vault_dir=tmp_path / "v")
    data = inst.load_settings(sp)
    assert data["env"]["FOO"] == "bar"
    assert data["env"]["WIKIMOTH_VAULT"] == str((tmp_path / "v").resolve())
