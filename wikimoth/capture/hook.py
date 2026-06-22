# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Hook entry point — ``python -m wikimoth.capture.hook [EVENT]``.

Claude Code invokes this once per lifecycle event, passing the event payload as
JSON on stdin (the install command wires the five events below). The dispatcher:

- **SessionStart** — record the start, then print the recall block as
  ``hookSpecificOutput.additionalContext`` (boot with memory).
- **UserPromptSubmit / PostToolUse** — append a lean record to the buffer.
- **Stop / SessionEnd** — append, then (re)write the one session note from the
  full buffer.

Hard rule: a hook must never break the user's session. Every path is wrapped so
that on *any* error we log to ``<vault>/.sessions/hook-errors.log`` and exit 0
with no output. The only stdout we ever emit is the SessionStart JSON envelope.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from wikimoth.capture.buffer import SessionBuffer, make_record
from wikimoth.capture.config import CaptureConfig
from wikimoth.capture.note import write_session_note
from wikimoth.capture.recall import build_recall_block

_BUFFERED = {"SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "SessionEnd"}


def _read_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        # Not just JSONDecodeError: pathological input (e.g. deeply-nested
        # arrays) raises RecursionError. A hook must never crash on stdin.
        return {}
    return obj if isinstance(obj, dict) else {}


def _maybe_llm(config: CaptureConfig):
    """Optional prose drafter. ``None`` by default (capture stays API-free).

    When ``WIKIMOTH_LLM_PROSE`` is set the note builder is *allowed* to draft a
    summary sentence, but wiring a real model is intentionally out of the
    default path — edges never depend on it. Returns ``None`` for now; the
    ``llm`` param is plumbed through :func:`write_session_note` for when a
    deterministic-friendly drafter is added.
    """
    return None


def _log_error(config: CaptureConfig | None, message: str) -> None:
    try:
        if config is None:
            return
        path = config.error_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(message.rstrip() + "\n")
    except Exception:
        pass  # logging must never raise either


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    config: CaptureConfig | None = None
    try:
        # Everything that could fail (stdin read/parse, field access, config
        # resolution, dispatch) lives inside the guard so the hook ALWAYS
        # exits 0 and never breaks the user's session.
        payload = _read_payload()
        event = (payload.get("hook_event_name") or (argv[0] if argv else "") or "").strip()
        payload.setdefault("hook_event_name", event)
        cwd = str(payload.get("cwd") or os.getcwd())
        session_id = str(payload.get("session_id") or "unknown")

        config = CaptureConfig.resolve(cwd=cwd, env=os.environ)

        if event in _BUFFERED:
            buf = SessionBuffer.for_session(config, session_id)
            buf.append(make_record(payload, capture_prompts=config.capture_prompts))

        if event == "SessionStart":
            block = build_recall_block(config)
            if block:
                envelope = {
                    "continue": True,
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": block,
                    },
                }
                sys.stdout.write(json.dumps(envelope))
        elif event in ("Stop", "SessionEnd"):
            write_session_note(config, session_id, llm=_maybe_llm(config))
    except Exception as exc:  # never break the session
        import traceback

        _log_error(config, f"[hook] {exc!r}\n{traceback.format_exc()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
