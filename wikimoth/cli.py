# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth`` command line — install/uninstall capture hooks, inspect state.

Subcommands::

    wikimoth install [--user|--project] [--dir DIR] [--vault PATH]
    wikimoth uninstall [--user|--project] [--dir DIR]
    wikimoth status [--vault PATH]
    wikimoth serve [--vault PATH] [--host H] [--port N] [--top-k K]
    wikimoth capture EVENT            # manual hook invocation (reads stdin)

``install`` writes the five lifecycle hooks into a Claude Code ``settings.json``
(project ``./.claude/settings.json`` by default, or user ``~/.claude/settings
.json`` with ``--user``). It is the WikiMoth equivalent of ``npx <tool> install`` —
one command and sessions start being captured into a deterministic
``[[wikilink]]`` vault.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from wikimoth.capture.config import CaptureConfig
from wikimoth.capture import install as _install


def _settings_path(args) -> Path:
    base = Path(args.dir) if args.dir else (Path.home() if args.user else Path.cwd())
    return base / ".claude" / "settings.json"


def _cmd_install(args) -> int:
    path = _settings_path(args)
    summary = _install.install(path, vault_dir=args.vault)
    print(f"WikiMoth capture installed → {summary['settings_path']}")
    if summary.get("backup"):
        print(f"  backed up previous settings → {summary['backup']}")
    print(f"  events: {', '.join(summary['events'])}")
    if summary.get("vault_env"):
        print(f"  WIKIMOTH_VAULT (settings env) → {summary['vault_env']}")
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        print(f"  vault (default): {cfg.vault_dir}")
        print("  (override with --vault PATH or the WIKIMOTH_VAULT env var)")
    print("\nStart a new Claude Code session to begin capturing.")
    return 0


def _cmd_uninstall(args) -> int:
    path = _settings_path(args)
    summary = _install.uninstall(path)
    print(f"WikiMoth capture removed from {summary['settings_path']}")
    return 0


def _cmd_status(args) -> int:
    env = dict(os.environ)
    if args.vault:
        env["WIKIMOTH_VAULT"] = str(args.vault)
    cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=env)
    notes = sessions = 0
    if cfg.vault_dir.exists():
        for p in cfg.vault_dir.rglob("*.md"):
            if cfg.sessions_dir.resolve() in p.resolve().parents:
                continue
            if p.stem.startswith("session-"):
                sessions += 1
            else:
                notes += 1
    buffers = len(list(cfg.sessions_dir.glob("*.jsonl"))) if cfg.sessions_dir.exists() else 0

    project_installed = _has_hooks(Path.cwd() / ".claude" / "settings.json")
    user_installed = _has_hooks(Path.home() / ".claude" / "settings.json")

    print("WikiMoth capture status")
    print(f"  vault_dir:        {cfg.vault_dir}")
    print(f"  content notes:    {notes}")
    print(f"  session notes:    {sessions}")
    print(f"  live buffers:     {buffers}")
    print(f"  llm prose:        {'on' if cfg.enable_llm_prose else 'off (deterministic)'}")
    print(f"  hooks (project):  {'installed' if project_installed else 'not installed'}")
    print(f"  hooks (user):     {'installed' if user_installed else 'not installed'}")
    return 0


def _has_hooks(settings_path: Path) -> bool:
    settings = _install.load_settings(settings_path)
    for groups in (settings.get("hooks") or {}).values():
        for g in groups or []:
            if _install._is_wikimoth_group(g):
                return True
    return False


def _cmd_serve(args) -> int:
    from wikimoth.capture.config import CaptureConfig
    from wikimoth import viewer

    if args.top_k < 1:
        print("--top-k must be a positive integer.")
        return 1
    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    if not vault.is_dir():
        print(f"vault not found (or not a directory): {vault}")
        print("  capture some sessions first (wikimoth install), or pass --vault PATH.")
        return 1
    return viewer.serve(vault, host=args.host, port=args.port, top_k=args.top_k)


def _cmd_capture(args) -> int:
    from wikimoth.capture.hook import main as hook_main

    return hook_main([args.event])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wikimoth", description="WikiMoth deterministic memory capture")
    sub = p.add_subparsers(dest="command", required=True)

    def add_scope(sp):
        sp.add_argument("--user", action="store_true", help="write ~/.claude/settings.json")
        sp.add_argument("--project", action="store_true", help="write ./.claude/settings.json (default)")
        sp.add_argument("--dir", help="settings base dir override (expects a .claude/ under it)")

    sp_i = sub.add_parser("install", help="install capture hooks")
    add_scope(sp_i)
    sp_i.add_argument("--vault", help="vault dir for captured notes (sets WIKIMOTH_VAULT)")
    sp_i.set_defaults(func=_cmd_install)

    sp_u = sub.add_parser("uninstall", help="remove capture hooks")
    add_scope(sp_u)
    sp_u.set_defaults(func=_cmd_uninstall)

    sp_s = sub.add_parser("status", help="show capture status")
    sp_s.add_argument("--vault", help="vault dir to inspect")
    sp_s.set_defaults(func=_cmd_status)

    sp_v = sub.add_parser("serve", help="open the local web viewer for the vault")
    sp_v.add_argument("--vault", help="vault dir to serve (default: resolved capture vault)")
    sp_v.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1, local-only)")
    sp_v.add_argument("--port", type=int, default=8765, help="bind port (default 8765)")
    sp_v.add_argument("--top-k", type=int, default=8, dest="top_k", help="chunks per question (default 8)")
    sp_v.set_defaults(func=_cmd_serve)

    sp_c = sub.add_parser("capture", help="run a hook event manually (reads stdin JSON)")
    sp_c.add_argument("event", help="hook event name, e.g. SessionStart")
    sp_c.set_defaults(func=_cmd_capture)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
