# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Install / uninstall WikiMoth's hooks into a Claude Code ``settings.json``.

Non-destructive and idempotent. We tag our hook entries by the ``wikimoth.capture
.hook`` substring in their command, so :func:`install` can be re-run safely
(it strips our prior entries first) and :func:`uninstall` removes only ours,
leaving any other hooks untouched.

The five wired events mirror :data:`wikimoth.capture.hook._BUFFERED`. The command
uses the *current* interpreter's absolute path (``sys.executable``) so the hook
runs in the same environment WikiMoth was installed into (venv-correct), and is
double-quoted for Windows paths with spaces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MARKER = "wikimoth.capture.hook"

# (event, matcher) — matcher None means "no matcher key" (session/prompt/stop
# events aren't tool-scoped). PostToolUse matches all tools.
_EVENTS: tuple[tuple[str, str | None], ...] = (
    ("SessionStart", None),
    ("UserPromptSubmit", None),
    ("PostToolUse", "*"),
    ("Stop", None),
    ("SessionEnd", None),
)


def hook_command(event: str, python_exe: str | None = None) -> str:
    """The shell command Claude Code runs for ``event``."""
    py = python_exe or sys.executable or "python"
    return f'"{py}" -m {MARKER} {event}'


def build_hooks_block(python_exe: str | None = None) -> dict[str, list]:
    """Build the ``hooks`` mapping (one group per wired event)."""
    block: dict[str, list] = {}
    for event, matcher in _EVENTS:
        group: dict[str, Any] = {"hooks": [{"type": "command", "command": hook_command(event, python_exe)}]}
        if matcher is not None:
            group = {"matcher": matcher, **group}
        block[event] = [group]
    return block


def _is_wikimoth_group(group: Any) -> bool:
    if not isinstance(group, dict):
        return False
    for h in group.get("hooks", []) or []:
        if isinstance(h, dict) and MARKER in str(h.get("command", "")):
            return True
    return False


def merge_hooks(settings: dict, python_exe: str | None = None) -> dict:
    """Return ``settings`` with WikiMoth's hook groups merged in (idempotent)."""
    out = dict(settings)
    raw = out.get("hooks")
    hooks = dict(raw) if isinstance(raw, dict) else {}
    block = build_hooks_block(python_exe)
    for event, groups in block.items():
        prev = hooks.get(event)
        prev = prev if isinstance(prev, list) else []
        existing = [g for g in prev if not _is_wikimoth_group(g)]
        hooks[event] = existing + groups
    out["hooks"] = hooks
    return out


def remove_hooks(settings: dict) -> dict:
    """Return ``settings`` with all WikiMoth hook groups removed."""
    out = dict(settings)
    raw = out.get("hooks")
    hooks = dict(raw) if isinstance(raw, dict) else {}
    cleaned: dict[str, list] = {}
    for event, groups in hooks.items():
        glist = groups if isinstance(groups, list) else []
        kept = [g for g in glist if not _is_wikimoth_group(g)]
        if kept:
            cleaned[event] = kept
    if cleaned:
        out["hooks"] = cleaned
    else:
        out.pop("hooks", None)
    return out


def load_settings(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return obj if isinstance(obj, dict) else {}


def save_settings(path: str | Path, settings: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def install(
    settings_path: str | Path,
    *,
    python_exe: str | None = None,
    vault_dir: str | Path | None = None,
) -> dict:
    """Install hooks into ``settings_path`` (backing up any existing file).

    If ``vault_dir`` is given it is written to ``settings['env']['WIKIMOTH_VAULT']``
    so the hook resolves to that vault. Returns a small summary dict.
    """
    p = Path(settings_path)
    existing = load_settings(p)
    if p.exists():
        backup = p.with_suffix(p.suffix + ".bak")
        try:
            backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            backup = None  # type: ignore[assignment]
    else:
        backup = None  # type: ignore[assignment]

    merged = merge_hooks(existing, python_exe)
    if vault_dir is not None:
        env = dict(merged.get("env") or {})
        env["WIKIMOTH_VAULT"] = str(Path(vault_dir).resolve())
        merged["env"] = env

    save_settings(p, merged)
    return {
        "settings_path": str(p),
        "backup": str(backup) if backup else None,
        "events": [e for e, _ in _EVENTS],
        "vault_env": merged.get("env", {}).get("WIKIMOTH_VAULT"),
    }


def uninstall(settings_path: str | Path) -> dict:
    """Remove WikiMoth hooks from ``settings_path``. Returns a summary dict."""
    p = Path(settings_path)
    existing = load_settings(p)
    cleaned = remove_hooks(existing)
    save_settings(p, cleaned)
    return {"settings_path": str(p), "removed_marker": MARKER}


__all__ = [
    "MARKER",
    "hook_command",
    "build_hooks_block",
    "merge_hooks",
    "remove_hooks",
    "load_settings",
    "save_settings",
    "install",
    "uninstall",
]
