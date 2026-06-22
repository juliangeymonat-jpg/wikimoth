# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""SessionBuffer — the append-only, per-session event log.

Each Claude Code session gets one JSONL file under
``<vault>/.sessions/<session_id>.jsonl``. Lifecycle hooks append one **lean**
record per event (tool name + file path, a truncated prompt, never the full tool
output — capturing the dump we exist to avoid would defeat the point). The note
builder later reads the whole buffer back and turns it into one deterministic
note; keeping the buffer is what makes note-generation a *pure function of the
buffer* (and re-derivable).

Records are written with ``sort_keys=True`` so a given event always serialises
the same way. Timestamps are the one inherently-wall-clock value, and they enter
*here* at capture time (not at note-build time), so "same buffer → same note"
holds.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# How much of a prompt / command we keep. Enough to drive link-matching and a
# readable summary; not the whole thing.
_PROMPT_CAP = 600
_CMD_CAP = 300

# Tool calls whose ``file_path`` we treat as "a file was touched".
_FILE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Read"}
# Tools that mutate a file (used downstream to distinguish touched-vs-read; we
# record the tool name so the note builder can decide).


def _now_iso() -> str:
    """Capture-time timestamp (seconds precision). The only wall-clock read."""
    return datetime.now().isoformat(timespec="seconds")


def _trunc(text: object, cap: int) -> str:
    s = str(text or "")
    return s if len(s) <= cap else s[:cap] + "…"


def make_record(payload: Mapping[str, Any], *, capture_prompts: bool = True) -> dict | None:
    """Map a Claude Code hook stdin payload to a lean buffer record.

    Returns ``None`` for events we don't persist (e.g. a prompt when
    ``capture_prompts`` is off). The ``ts`` is stamped here, at capture time.
    """
    event = str(payload.get("hook_event_name") or "").strip()
    rec: dict[str, Any] = {"event": event, "ts": _now_iso()}

    if event == "SessionStart":
        rec["session_id"] = str(payload.get("session_id") or "unknown")
        rec["cwd"] = str(payload.get("cwd") or "")
        rec["source"] = str(payload.get("source") or "")
        if payload.get("model"):
            rec["model"] = str(payload.get("model"))
        return rec

    if event == "UserPromptSubmit":
        if not capture_prompts:
            return None
        rec["prompt"] = _trunc(payload.get("prompt"), _PROMPT_CAP)
        return rec

    if event == "PostToolUse":
        tool = str(payload.get("tool_name") or "")
        rec["tool"] = tool
        tool_input = payload.get("tool_input") or {}
        if isinstance(tool_input, Mapping):
            fp = tool_input.get("file_path") or tool_input.get("notebook_path")
            if fp:
                rec["file_path"] = str(fp)
            if tool == "Bash" and tool_input.get("command"):
                rec["command"] = _trunc(tool_input.get("command"), _CMD_CAP)
        return rec

    if event in ("Stop", "SessionEnd"):
        if payload.get("reason"):
            rec["reason"] = str(payload.get("reason"))
        return rec

    # Unknown/uninteresting event: still record the bare event so the buffer
    # reflects what happened, but nothing else.
    return rec


class SessionBuffer:
    """Append-only JSONL log for one session.

    Construct directly with a path, or via :meth:`for_session`. All reads/writes
    are tolerant of a missing file (a session may flush before any event was
    buffered).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_session(cls, config, session_id: str) -> "SessionBuffer":
        return cls(config.buffer_path(session_id))

    def exists(self) -> bool:
        return self.path.exists()

    def append(self, record: Mapping[str, Any] | None) -> None:
        """Append one record (no-op if ``record`` is ``None``)."""
        if record is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def read(self) -> list[dict]:
        """Return all records in order. Skips any corrupt line defensively."""
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


__all__ = ["SessionBuffer", "make_record"]
