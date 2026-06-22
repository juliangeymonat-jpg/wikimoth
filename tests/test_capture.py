# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for WikiMoth's ``wikimoth.capture`` subpackage — deterministic, API-free.

The subpackage turns Claude Code lifecycle-hook events into ONE deterministic
``[[wikilink]]`` markdown note per session, where **edges (links) are computed
by code, never by an LLM**. These tests pin that contract:

* config resolution is a pure function of (cwd, env);
* the session buffer is a tolerant append-only JSONL log;
* link computation is conservative and reproducible (the heart);
* note building is byte-deterministic and *strips any model-emitted edge*;
* install/uninstall is idempotent + non-destructive;
* the hook entry point never breaks the session (exit 0 on any input);
* a captured note's edges are actually walkable by the real GraphRetriever.

Nothing here calls the network, Claude, or any paid service. WikiMoth is
self-contained, so just put it on the path:

    PYTHONPATH=.  pytest -q
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wikimoth.capture.config import CaptureConfig
from wikimoth.capture.buffer import SessionBuffer, make_record
from wikimoth.capture.links import (
    compute_links,
    file_note_links,
    is_session_stem,
    match_notes,
    previous_session,
)
from wikimoth.capture.note import build_note, write_session_note
from wikimoth.capture.recall import build_recall_block
from wikimoth.capture import install as inst


_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _linkset(text: str) -> set[str]:
    """The set of ``[[...]]`` targets appearing in note text."""
    return set(re.findall(r"\[\[([^\]]+)\]\]", text))


def _seed_content_note(vault: Path, stem: str, *, description: str = "a note", body: str = "Body.") -> Path:
    p = vault / f"{stem}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\nname: {stem}\ndescription: \"{description}\"\n---\n{body}\n",
        encoding="utf-8",
    )
    return p


def _seed_session_note(vault: Path, stem: str, *, description: str = "prior", links=()) -> Path:
    link_lines = "\n".join(f"[[{l}]]" for l in links)
    p = vault / f"{stem}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\nname: {stem}\ndescription: \"{description}\"\nmetadata:\n  type: session\n---\n"
        f"# prior\n{link_lines}\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# 1. CaptureConfig.resolve
# ---------------------------------------------------------------------------

def test_resolve_default_vault(tmp_path):
    cfg = CaptureConfig.resolve(cwd=tmp_path, env={})
    assert cfg.vault_dir == tmp_path / ".wikimoth" / "vault"
    assert cfg.sessions_dir == tmp_path / ".wikimoth" / "vault" / ".sessions"


def test_resolve_vault_env_override(tmp_path):
    custom = tmp_path / "elsewhere" / "myvault"
    cfg = CaptureConfig.resolve(cwd=tmp_path, env={"WIKIMOTH_VAULT": str(custom)})
    assert cfg.vault_dir == custom


def test_resolve_llm_prose_flag(tmp_path):
    assert CaptureConfig.resolve(cwd=tmp_path, env={}).enable_llm_prose is False
    cfg = CaptureConfig.resolve(cwd=tmp_path, env={"WIKIMOTH_LLM_PROSE": "1"})
    assert cfg.enable_llm_prose is True


def test_resolve_capture_prompts_flag(tmp_path):
    assert CaptureConfig.resolve(cwd=tmp_path, env={}).capture_prompts is True
    cfg = CaptureConfig.resolve(cwd=tmp_path, env={"WIKIMOTH_CAPTURE_PROMPTS": "0"})
    assert cfg.capture_prompts is False


def test_buffer_path_sanitises_traversal(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    sessions = cfg.sessions_dir.resolve()
    for evil in ("../../etc/passwd", "a/b/c", "..", "x/../../y"):
        bp = cfg.buffer_path(evil).resolve()
        # The buffer file must stay strictly inside the sessions dir (no escape):
        # path separators are stripped, so the id can never climb a directory.
        assert sessions == bp.parent, f"{evil!r} escaped to {bp}"
        assert "/" not in bp.name and "\\" not in bp.name
        # the resolved path is genuinely under the sessions dir
        assert sessions in bp.parents or sessions == bp.parent


# ---------------------------------------------------------------------------
# 2. make_record
# ---------------------------------------------------------------------------

def test_make_record_session_start():
    rec = make_record({
        "hook_event_name": "SessionStart",
        "session_id": "deadbeef1234",
        "cwd": "/tmp/x",
        "source": "startup",
    })
    assert rec["event"] == "SessionStart"
    assert rec["session_id"] == "deadbeef1234"
    assert rec["cwd"] == "/tmp/x"
    assert rec["source"] == "startup"
    assert rec.get("ts")


def test_make_record_posttooluse_file_and_command():
    edit = make_record({
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": "notes/foo.md"},
    })
    assert edit["tool"] == "Edit"
    assert edit["file_path"] == "notes/foo.md"
    assert "command" not in edit

    bash = make_record({
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "pytest -q"},
    })
    assert bash["tool"] == "Bash"
    assert bash["command"] == "pytest -q"


def test_make_record_prompt_suppressed_when_off():
    assert make_record(
        {"hook_event_name": "UserPromptSubmit", "prompt": "hi"},
        capture_prompts=False,
    ) is None
    on = make_record({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}, capture_prompts=True)
    assert on["prompt"] == "hi"


def test_make_record_always_has_event_and_ts():
    for payload in (
        {"hook_event_name": "SessionStart", "session_id": "x"},
        {"hook_event_name": "PostToolUse", "tool_name": "Read", "tool_input": {}},
        {"hook_event_name": "Stop"},
        {"hook_event_name": "Whatever"},
    ):
        rec = make_record(payload)
        assert rec is not None
        assert "event" in rec and "ts" in rec


# ---------------------------------------------------------------------------
# 3. SessionBuffer
# ---------------------------------------------------------------------------

def test_buffer_append_read_roundtrip_preserves_order(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    buf = SessionBuffer.for_session(cfg, "sess1")
    recs = [
        {"event": "SessionStart", "ts": "2026-06-21T10:00:00"},
        {"event": "UserPromptSubmit", "ts": "2026-06-21T10:01:00", "prompt": "p"},
        {"event": "Stop", "ts": "2026-06-21T10:02:00"},
    ]
    for r in recs:
        buf.append(r)
    read = buf.read()
    assert read == recs  # order + content preserved


def test_buffer_read_missing_file_is_empty(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    buf = SessionBuffer.for_session(cfg, "nope")
    assert buf.exists() is False
    assert buf.read() == []


def test_buffer_read_skips_corrupt_line(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    buf = SessionBuffer.for_session(cfg, "sess2")
    buf.append({"event": "A", "ts": "t1"})
    # inject a corrupt (non-JSON) line directly into the buffer file
    with buf.path.open("a", encoding="utf-8") as fh:
        fh.write("this is not json{{{\n")
    buf.append({"event": "B", "ts": "t2"})
    read = buf.read()
    assert [r["event"] for r in read] == ["A", "B"]


def test_buffer_delete_safe_when_absent(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    buf = SessionBuffer.for_session(cfg, "ghost")
    buf.delete()  # must not raise
    buf.append({"event": "X", "ts": "t"})
    assert buf.exists()
    buf.delete()
    assert buf.exists() is False


# ---------------------------------------------------------------------------
# 4. links — the heart: conservatism + all three sources
# ---------------------------------------------------------------------------

def test_match_notes_multi_token_verbatim():
    idx = {"project alpha beta": "project-alpha-beta"}
    assert match_notes("we worked on project alpha beta today", idx) == ["project-alpha-beta"]
    # not present verbatim → no match
    assert match_notes("project gamma beta", idx) == []


def test_match_notes_rejects_short_single_token():
    idx = {"rag": "rag-note"}  # 'rag' is 3 chars < default min_token_len=4
    assert match_notes("a rag pipeline", idx) == []


def test_match_notes_rejects_stopword_title():
    idx = {"memory": "memory-note", "test": "test-note"}
    assert match_notes("the memory bank test run", idx) == []


def test_match_notes_no_substring_match():
    # 'mothragx' must NOT match the note 'mothrag' — phrase match is word-bounded,
    # not substring. (mothrag is 7 chars so length is not what blocks it.)
    idx = {"mothrag": "mothrag-note"}
    assert match_notes("the mothragx tool", idx) == []
    # but the real word does match
    assert match_notes("the mothrag tool", idx) == ["mothrag-note"]


def test_match_notes_returns_sorted_stems():
    idx = {
        "zebra topic": "zebra-topic",
        "alpha topic": "alpha-topic",
        "middle topic": "middle-topic",
    }
    corpus = "discussed zebra topic then alpha topic and middle topic"
    assert match_notes(corpus, idx) == ["alpha-topic", "middle-topic", "zebra-topic"]


def test_file_note_links_basename_slug_match():
    idx = {"my note": "my-note", "other": "other-note"}
    # a touched file whose basename slug == an existing note slug
    assert file_note_links(["some/dir/my-note.md"], idx) == ["my-note"]
    # unrelated file → nothing
    assert file_note_links(["some/dir/unrelated.py"], idx) == []


def test_previous_session_largest_strictly_less():
    stems = [
        "session-2026-06-18-aaaa",
        "session-2026-06-20-cccc",
        "session-2026-06-19-bbbb",
    ]
    assert previous_session("session-2026-06-21-zzzz", stems) == "session-2026-06-20-cccc"
    # strictly-less: an equal stem is not "previous"
    assert previous_session("session-2026-06-18-aaaa", stems) == None  # noqa: E711


def test_previous_session_none_when_none_earlier():
    assert previous_session("session-2026-06-01-aaaa", ["session-2026-06-02-bbbb"]) is None
    assert previous_session("session-2026-06-01-aaaa", []) is None


def test_compute_links_order_dedup_no_self():
    self_stem = "session-2026-06-21-deadbeef"
    note_index = {"mothrag canonical repo": "mothrag-canonical-repo"}
    session_stems = [self_stem, "session-2026-06-20-aaaa"]
    events = [
        {"event": "SessionStart", "ts": "2026-06-21T10:00:00"},
        {"event": "UserPromptSubmit", "ts": "2026-06-21T10:01:00",
         "prompt": "work on the mothrag canonical repo"},
        {"event": "PostToolUse", "ts": "2026-06-21T10:02:00",
         "tool": "Edit", "file_path": "mothrag-canonical-repo.md"},
    ]
    links = compute_links(
        self_stem=self_stem,
        events=events,
        note_index=note_index,
        session_stems=session_stems,
    )
    # prev-session first, then file→note / phrase (which resolve to the same
    # note → deduped to a single entry). Self never appears.
    assert links[0] == "session-2026-06-20-aaaa"
    assert "mothrag-canonical-repo" in links
    assert self_stem not in links
    assert len(links) == len(set(links))  # deduped


def test_is_session_stem():
    assert is_session_stem("session-2026-06-21-deadbeef") is True
    assert is_session_stem("mothrag-canonical-repo") is False
    assert is_session_stem("") is False


# ---------------------------------------------------------------------------
# 5. build_note determinism
# ---------------------------------------------------------------------------

_EVENTS = [
    {"event": "SessionStart", "ts": "2026-06-21T10:00:00",
     "session_id": "deadbeef1234", "cwd": "/work/dir", "source": "startup"},
    {"event": "UserPromptSubmit", "ts": "2026-06-21T10:01:00",
     "prompt": "work on the mothrag canonical repo please"},
    {"event": "PostToolUse", "ts": "2026-06-21T10:02:00",
     "tool": "Edit", "file_path": "mothrag-canonical-repo.md"},
    {"event": "PostToolUse", "ts": "2026-06-21T10:03:00",
     "tool": "Bash", "command": "pytest"},
    {"event": "Stop", "ts": "2026-06-21T10:04:00"},
]
_NOTE_INDEX = {"mothrag canonical repo": "mothrag-canonical-repo"}
_SESSION_STEMS = ["session-2026-06-20-aaaaaaaa"]


def test_build_note_is_byte_identical():
    stem1, c1 = build_note(_EVENTS, session_id="deadbeef1234",
                           note_index=_NOTE_INDEX, session_stems=_SESSION_STEMS)
    stem2, c2 = build_note(_EVENTS, session_id="deadbeef1234",
                           note_index=_NOTE_INDEX, session_stems=_SESSION_STEMS)
    assert stem1 == stem2
    assert c1 == c2


def test_build_note_stem_format_and_frontmatter():
    stem, content = build_note(_EVENTS, session_id="deadbeef1234",
                               note_index=_NOTE_INDEX, session_stems=_SESSION_STEMS)
    assert stem == "session-2026-06-21-deadbeef"  # session-YYYY-MM-DD-<first8ofid>
    assert content.startswith("---\n")
    assert "metadata:" in content
    assert "  type: session" in content


def test_build_note_emits_all_three_edge_kinds():
    _, content = build_note(_EVENTS, session_id="deadbeef1234",
                            note_index=_NOTE_INDEX, session_stems=_SESSION_STEMS)
    links = _linkset(content)
    assert "session-2026-06-20-aaaaaaaa" in links  # prev-session chain
    assert "mothrag-canonical-repo" in links       # file→note / phrase
    assert "session-2026-06-21-deadbeef" not in links  # never self


# ---------------------------------------------------------------------------
# 6. build_note LLM-edge invariant (CRITICAL — WikiMoth's core guarantee)
# ---------------------------------------------------------------------------

def test_build_note_strips_llm_injected_edges():
    """An LLM that emits [[EVIL]] must NOT inject an edge into the graph."""
    def evil_llm(_prompt: str) -> str:
        return "Did the thing. [[EVIL]] then more [[another]] stuff here."

    cfg_det = CaptureConfig(vault_dir=Path("."))
    cfg_llm = CaptureConfig(vault_dir=Path("."), enable_llm_prose=True)

    _, det = build_note(_EVENTS, session_id="deadbeef1234", note_index=_NOTE_INDEX,
                        session_stems=_SESSION_STEMS, config=cfg_det)
    _, llm = build_note(_EVENTS, session_id="deadbeef1234", note_index=_NOTE_INDEX,
                        session_stems=_SESSION_STEMS, config=cfg_llm, llm=evil_llm)

    # 1. the injected tokens are gone entirely
    assert "[[EVIL]]" not in llm
    assert "[[another]]" not in llm
    assert "EVIL" not in _linkset(llm)
    assert "another" not in _linkset(llm)
    # 2. the prose text (minus the links) is retained
    assert "Did the thing." in llm
    assert "then more" in llm and "stuff here." in llm
    # 3. the link set is EXACTLY the deterministic one — code owns the edges
    assert _linkset(llm) == _linkset(det)


def test_build_note_llm_disabled_ignores_llm():
    """With enable_llm_prose False, an evil llm is never even called."""
    def boom(_prompt: str) -> str:  # pragma: no cover - must not be invoked
        raise AssertionError("llm must not be called when prose disabled")

    cfg = CaptureConfig(vault_dir=Path("."), enable_llm_prose=False)
    _, content = build_note(_EVENTS, session_id="deadbeef1234", note_index=_NOTE_INDEX,
                            session_stems=_SESSION_STEMS, config=cfg, llm=boom)
    assert "deterministic summary" in content


# ---------------------------------------------------------------------------
# 7. write_session_note
# ---------------------------------------------------------------------------

def test_write_session_note_writes_file(tmp_path):
    vault = tmp_path / "vault"
    cfg = CaptureConfig(vault_dir=vault)
    buf = SessionBuffer.for_session(cfg, "deadbeef1234")
    for e in _EVENTS:
        buf.append(e)
    path = write_session_note(cfg, "deadbeef1234")
    assert path is not None
    assert path == cfg.note_path("session-2026-06-21-deadbeef")
    assert path.exists()


def test_write_session_note_none_on_empty_buffer(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    assert write_session_note(cfg, "never-buffered") is None


def test_write_session_note_idempotent(tmp_path):
    cfg = CaptureConfig(vault_dir=tmp_path / "vault")
    buf = SessionBuffer.for_session(cfg, "deadbeef1234")
    for e in _EVENTS:
        buf.append(e)
    p1 = write_session_note(cfg, "deadbeef1234")
    c1 = p1.read_text(encoding="utf-8")
    p2 = write_session_note(cfg, "deadbeef1234")
    c2 = p2.read_text(encoding="utf-8")
    assert p1 == p2
    assert c1 == c2


def test_write_session_note_picks_up_prior_session_as_link(tmp_path):
    """A pre-existing session note becomes the prev-session link target."""
    vault = tmp_path / "vault"
    cfg = CaptureConfig(vault_dir=vault)
    _seed_session_note(vault, "session-2026-06-20-aaaaaaaa")

    buf = SessionBuffer.for_session(cfg, "deadbeef1234")
    for e in _EVENTS:
        buf.append(e)
    path = write_session_note(cfg, "deadbeef1234")
    text = path.read_text(encoding="utf-8")
    assert "[[session-2026-06-20-aaaaaaaa]]" in text


# ---------------------------------------------------------------------------
# 8. recall
# ---------------------------------------------------------------------------

def test_recall_empty_when_no_session_notes(tmp_path):
    vault = tmp_path / "vault"
    cfg = CaptureConfig(vault_dir=vault)
    assert build_recall_block(cfg) == ""  # vault doesn't exist
    vault.mkdir(parents=True)
    _seed_content_note(vault, "just-a-content-note")  # not a session note
    assert build_recall_block(cfg) == ""


def test_recall_newest_first_capped(tmp_path):
    vault = tmp_path / "vault"
    cfg = CaptureConfig(vault_dir=vault, recall_max_notes=2)
    for d in ("18", "19", "20"):
        _seed_session_note(vault, f"session-2026-06-{d}-aaaa", description=f"day {d}")
    block = build_recall_block(cfg)
    # capped to 2, newest first → 20 then 19, 18 dropped
    assert "session-2026-06-20-aaaa" in block
    assert "session-2026-06-19-aaaa" in block
    assert "session-2026-06-18-aaaa" not in block
    assert block.index("session-2026-06-20-aaaa") < block.index("session-2026-06-19-aaaa")


def test_recall_includes_description_and_links(tmp_path):
    vault = tmp_path / "vault"
    cfg = CaptureConfig(vault_dir=vault)
    _seed_content_note(vault, "topic-note")
    _seed_session_note(vault, "session-2026-06-20-aaaa",
                       description="did some topic work", links=["topic-note"])
    block = build_recall_block(cfg)
    assert "did some topic work" in block
    assert "[[topic-note]]" in block


def test_recall_respects_max_chars(tmp_path):
    vault = tmp_path / "vault"
    cfg = CaptureConfig(vault_dir=vault, recall_max_notes=50, recall_max_chars=200)
    for i in range(40):
        _seed_session_note(vault, f"session-2026-06-{i:02d}-aaaa",
                           description="x" * 80)
    block = build_recall_block(cfg)
    assert len(block) <= 200 + len("\n…(truncated)")
    assert block.endswith("…(truncated)")


# ---------------------------------------------------------------------------
# 9. install / uninstall
# ---------------------------------------------------------------------------

def test_merge_hooks_idempotent():
    once = inst.merge_hooks({}, python_exe="C:/py/python.exe")
    twice = inst.merge_hooks(once, python_exe="C:/py/python.exe")
    assert once == twice


def test_merge_hooks_non_destructive():
    pre = {"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo other"}]}
    ]}}
    merged = inst.merge_hooks(pre, python_exe="C:/py/python.exe")
    ptu = merged["hooks"]["PostToolUse"]
    has_other = any("echo other" in h.get("command", "") for g in ptu for h in g.get("hooks", []))
    has_ours = any(inst.MARKER in h.get("command", "") for g in ptu for h in g.get("hooks", []))
    assert has_other and has_ours


def test_build_hooks_block_wires_all_five_events():
    block = inst.build_hooks_block(python_exe="C:/py/python.exe")
    for ev in ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "SessionEnd"):
        assert ev in block
        assert inst.MARKER in block[ev][0]["hooks"][0]["command"]


def test_posttooluse_group_matcher_is_star():
    block = inst.build_hooks_block(python_exe="C:/py/python.exe")
    assert block["PostToolUse"][0]["matcher"] == "*"
    # session/prompt/stop groups carry no matcher key
    assert "matcher" not in block["SessionStart"][0]


def test_remove_hooks_keeps_others():
    merged = inst.merge_hooks(
        {"hooks": {"PostToolUse": [
            {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo other"}]}
        ]}},
        python_exe="C:/py/python.exe",
    )
    cleaned = inst.remove_hooks(merged)
    ptu = cleaned.get("hooks", {}).get("PostToolUse", [])
    assert any("echo other" in h.get("command", "") for g in ptu for h in g.get("hooks", []))
    assert not any(inst.MARKER in h.get("command", "") for g in ptu for h in g.get("hooks", []))


def test_is_wikimoth_group():
    ours = {"hooks": [{"type": "command", "command": f'"py" -m {inst.MARKER} Stop'}]}
    theirs = {"hooks": [{"type": "command", "command": "echo other"}]}
    assert inst._is_wikimoth_group(ours) is True
    assert inst._is_wikimoth_group(theirs) is False
    assert inst._is_wikimoth_group("not a dict") is False


def test_install_writes_backup_when_file_preexists(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    inst.save_settings(settings, {"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo other"}]}
    ]}})
    summary = inst.install(settings, python_exe="C:/py/python.exe")
    assert summary["backup"] is not None
    assert Path(summary["backup"]).exists()
    assert Path(summary["backup"]) == settings.with_suffix(".json.bak")


def test_install_no_backup_when_fresh(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    summary = inst.install(settings, python_exe="C:/py/python.exe")
    assert summary["backup"] is None
    assert settings.exists()


def test_install_writes_vault_env(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    vault = tmp_path / "myvault"
    inst.install(settings, python_exe="C:/py/python.exe", vault_dir=vault)
    data = inst.load_settings(settings)
    assert data["env"]["WIKIMOTH_VAULT"] == str(vault.resolve())


def test_install_then_uninstall_roundtrip_keeps_others(tmp_path):
    settings = tmp_path / ".claude" / "settings.json"
    inst.save_settings(settings, {"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo other"}]}
    ]}})
    inst.install(settings, python_exe="C:/py/python.exe")
    inst.uninstall(settings)
    data = inst.load_settings(settings)
    ptu = data.get("hooks", {}).get("PostToolUse", [])
    assert any("echo other" in h.get("command", "") for g in ptu for h in g.get("hooks", []))
    assert not any(inst.MARKER in h.get("command", "") for g in ptu for h in g.get("hooks", []))


# ---------------------------------------------------------------------------
# 10. hook dispatch via subprocess
# ---------------------------------------------------------------------------

def _hook_env(vault: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT
    env["WIKIMOTH_VAULT"] = str(vault)
    return env


def _run_hook(event: str, payload: dict, vault: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "wikimoth.capture.hook", event],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=_hook_env(vault),
        timeout=60,
    )


def test_hook_subprocess_lifecycle(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    sid = "feedface5678"
    cwd = str(tmp_path)

    # SessionStart (no prior notes yet) → exit 0
    r = _run_hook("SessionStart", {
        "hook_event_name": "SessionStart", "session_id": sid, "cwd": cwd, "source": "startup",
    }, vault)
    assert r.returncode == 0, r.stderr

    # a prompt + a tool use buffered
    assert _run_hook("UserPromptSubmit", {
        "hook_event_name": "UserPromptSubmit", "session_id": sid, "cwd": cwd,
        "prompt": "do some work",
    }, vault).returncode == 0
    assert _run_hook("PostToolUse", {
        "hook_event_name": "PostToolUse", "session_id": sid, "cwd": cwd,
        "tool_name": "Edit", "tool_input": {"file_path": "foo.md"},
    }, vault).returncode == 0

    # Stop → writes the session note
    r_stop = _run_hook("Stop", {
        "hook_event_name": "Stop", "session_id": sid, "cwd": cwd,
    }, vault)
    assert r_stop.returncode == 0
    notes = list(vault.glob("session-*.md"))
    assert notes, "Stop must have written a session note"

    # SessionEnd → idempotent re-write, still exit 0
    assert _run_hook("SessionEnd", {
        "hook_event_name": "SessionEnd", "session_id": sid, "cwd": cwd,
    }, vault).returncode == 0


def test_hook_sessionstart_emits_recall_envelope(tmp_path):
    vault = tmp_path / "vault"
    _seed_session_note(vault, "session-2026-06-20-aaaaaaaa", description="prior work")

    r = _run_hook("SessionStart", {
        "hook_event_name": "SessionStart", "session_id": "feedface5678",
        "cwd": str(tmp_path), "source": "startup",
    }, vault)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip(), "SessionStart with a prior note must emit an envelope"
    env = json.loads(r.stdout)
    assert env["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "session-2026-06-20-aaaaaaaa" in env["hookSpecificOutput"]["additionalContext"]


def test_hook_malformed_stdin_exits_zero(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    # non-JSON stdin
    bad = subprocess.run(
        [sys.executable, "-m", "wikimoth.capture.hook", "Stop"],
        input="this is not json {{{",
        capture_output=True, text=True, env=_hook_env(vault), timeout=60,
    )
    assert bad.returncode == 0
    # empty stdin
    empty = subprocess.run(
        [sys.executable, "-m", "wikimoth.capture.hook", "SessionStart"],
        input="",
        capture_output=True, text=True, env=_hook_env(vault), timeout=60,
    )
    assert empty.returncode == 0


# ---------------------------------------------------------------------------
# 11. graph integration (mirrors wikimoth_graph_smoke.py)
# ---------------------------------------------------------------------------

def test_captured_note_edge_is_walkable_by_graph_retriever(tmp_path):
    """Capture a note linking to a lexically-disjoint content note, then prove
    the real GraphRetriever walks that [[wikilink]] edge (capture→retrieve)."""
    from wikimoth.retrieval import GraphRetriever  # noqa: F401
    from wikimoth import MemoryRAG

    vault = tmp_path / "vault"
    # content note shares ZERO tokens with the later query → reachable only by hop
    _seed_content_note(
        vault, "falkordb-partnership",
        description="graph db partnership",
        body="Sparse-matrix substrate collaboration; sender is a database vendor.",
    )

    cfg = CaptureConfig(vault_dir=vault)
    buf = SessionBuffer.for_session(cfg, "cafebabe9999")
    for e in [
        {"event": "SessionStart", "ts": "2026-06-21T09:00:00",
         "session_id": "cafebabe9999", "cwd": str(tmp_path), "source": "startup"},
        {"event": "UserPromptSubmit", "ts": "2026-06-21T09:01:00",
         "prompt": "onboarding summary write-up about the falkordb partnership"},
        {"event": "PostToolUse", "ts": "2026-06-21T09:02:00",
         "tool": "Edit", "file_path": "falkordb-partnership.md"},
        {"event": "SessionEnd", "ts": "2026-06-21T09:05:00"},
    ]:
        buf.append(e)

    path = write_session_note(cfg, "cafebabe9999")
    assert "[[falkordb-partnership]]" in path.read_text(encoding="utf-8")

    rag = MemoryRAG(exclude_content=())
    rag.index(vault)
    assert rag.retriever.edge_count() > 0, "captured note must contribute graph edges"

    pairs = rag.retrieve_with_hops("onboarding summary write-up", top_k=8)
    by_slug: dict[str, int] = {}
    for c, hop in pairs:
        slug = c.metadata.get("note_slug", "")
        by_slug[slug] = min(hop, by_slug.get(slug, hop))
    falkor_hop = min((v for k, v in by_slug.items() if "falkordb" in k), default=99)
    assert falkor_hop >= 1, (
        "content note must be reached via the wikilink hop (not lexically); "
        f"slugs={by_slug}"
    )
