# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth`` command line — install/uninstall capture hooks, inspect state.

Subcommands::

    wikimoth install [--user|--project] [--dir DIR] [--vault PATH]
    wikimoth uninstall [--user|--project] [--dir DIR]
    wikimoth status [--vault PATH]
    wikimoth serve [--vault PATH] [--host H] [--port N] [--top-k K]
    wikimoth mcp [--vault PATH] [--top-k K]   # MCP server (stdio): Claude calls recall itself
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
from datetime import date
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


def _cmd_mcp(args) -> int:
    from wikimoth import mcp

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    # stdout is the MCP protocol channel: diagnostics go to stderr ONLY, never stdout.
    print(f"wikimoth mcp: serving vault {vault} over stdio (top_k={args.top_k})", file=sys.stderr)
    if not vault.is_dir():
        print(f"  note: {vault} does not exist yet; recall reports this until it does.", file=sys.stderr)
    return mcp.serve_stdio(vault, default_top_k=args.top_k)


def _cmd_conflicts(args) -> int:
    from wikimoth import conflicts as _conflicts

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    if not vault.is_dir():
        print(f"vault not found (or not a directory): {vault}")
        print("  capture some notes first (wikimoth install), or pass --vault PATH.")
        return 1
    report = _conflicts.scan_conflicts(
        vault,
        subject_keys=args.subject_key or _conflicts.DEFAULT_SUBJECT_KEYS,
        all_keys=args.all_keys,
        num_tol=args.num_tol,
        date_tol_days=args.date_tol_days,
        include_inline=args.include_inline,
        include_sessions=args.include_sessions,
        case_insensitive=args.case_insensitive,
    )
    print(_conflicts.format_conflicts(report, fmt=args.format))
    return 0


def _cmd_lint(args) -> int:
    from wikimoth import lint as _lint

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    if not vault.is_dir():
        print(f"vault not found (or not a directory): {vault}")
        print("  capture some notes first (wikimoth install), or pass --vault PATH.")
        return 1
    report = _lint.scan_lint(
        vault,
        stale_days=args.stale,
        min_stub_chars=args.stub_chars,
        include_sessions=args.include_sessions,
    )
    print(_lint.format_lint(report, fmt=args.format))
    return 0


def _cmd_recall(args) -> int:
    from wikimoth.mcp import _build_index, _format_recall
    from wikimoth.tokens import count_passage_tokens

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    if not vault.is_dir():
        print(f"vault not found (or not a directory): {vault}")
        print("  capture some notes first (wikimoth install), or pass --vault PATH.")
        return 1
    as_of = None
    if args.as_of:
        try:
            as_of = date.fromisoformat(args.as_of)
        except ValueError:
            print(f"recall: --as-of must be YYYY-MM-DD, got {args.as_of!r}")
            return 1
    rag, total, _, _ = _build_index(vault)
    rag.as_of = as_of
    rag.show_superseded = args.show_superseded
    pairs = rag.retrieve_with_hops(args.query, top_k=args.top_k)
    per_chunk = [count_passage_tokens([getattr(c, "text", "") or ""]) for c, _ in pairs]
    fed = sum(per_chunk)
    pct = (100.0 * (1 - fed / total)) if total else 0.0
    print(_format_recall(args.query, pairs, per_chunk, fed, total, pct))
    return 0


def _cmd_supersede(args) -> int:
    from wikimoth import supersede as _s

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    as_of = None
    if args.as_of:
        try:
            as_of = date.fromisoformat(args.as_of)
        except ValueError:
            print(f"supersede: --as-of must be YYYY-MM-DD, got {args.as_of!r}")
            return 1
    try:
        res = _s.supersede(
            vault, args.old, args.new,
            as_of=as_of, reason=args.reason or "",
            undo=args.undo, reverse=args.reverse, dry_run=args.dry_run,
        )
    except _s.SupersedeError as e:
        print(f"supersede: {e}")
        return 1
    print(_s.format_result(res))
    return 0


def _cmd_dedup(args) -> int:
    from wikimoth import dedup as _dedup

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    if not vault.is_dir():
        print(f"vault not found (or not a directory): {vault}")
        print("  capture some notes first (wikimoth install), or pass --vault PATH.")
        return 1
    report = _dedup.scan_dedup(
        vault,
        threshold=args.threshold,
        shingle_size=args.shingle_size,
        include_sessions=args.include_sessions,
    )
    print(_dedup.format_dedup(report, fmt=args.format))
    return 0


def _cmd_decay(args) -> int:
    from wikimoth import decay as _decay

    if args.vault:
        vault = Path(args.vault)
    else:
        cfg = CaptureConfig.resolve(cwd=Path.cwd(), env=os.environ)
        vault = cfg.vault_dir
    if not vault.is_dir():
        print(f"vault not found (or not a directory): {vault}")
        print("  capture some notes first (wikimoth install), or pass --vault PATH.")
        return 1
    report = _decay.scan_decay(
        vault,
        tau_days=args.tau_days,
        threshold=args.threshold,
        limit=args.limit,
        include_sessions=args.include_sessions,
    )
    print(_decay.format_decay(report, fmt=args.format))
    return 0


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

    sp_m = sub.add_parser("mcp", help="run the MCP server (stdio) so Claude calls WikiMoth recall in the loop")
    sp_m.add_argument("--vault", help="vault dir to serve (default: resolved capture vault)")
    sp_m.add_argument("--top-k", type=int, default=8, dest="top_k", help="default chunks per recall (default 8)")
    sp_m.set_defaults(func=_cmd_mcp)

    sp_x = sub.add_parser(
        "conflicts",
        help="surface deterministic contradiction candidates (same subject+predicate, different value)",
    )
    sp_x.add_argument("--vault", help="vault dir to scan (default: resolved capture vault)")
    sp_x.add_argument("--format", choices=("text", "json"), default="text", help="output format")
    sp_x.add_argument(
        "--subject-key", action="append", dest="subject_key",
        help="frontmatter key naming the subject (repeatable; default: subject/entity/about/topic)",
    )
    sp_x.add_argument(
        "--all-keys", action="store_true",
        help="compare every non-subject frontmatter key (default skips structural/bookkeeping keys)",
    )
    sp_x.add_argument("--num-tol", type=float, default=0.0, help="treat numeric values within this spread as equal")
    sp_x.add_argument("--date-tol-days", type=int, default=0, help="treat dates within this many days as equal")
    sp_x.add_argument("--include-inline", action="store_true", help="also read Dataview 'key:: value' inline body fields")
    sp_x.add_argument("--include-sessions", action="store_true", help="also scan session-* notes (skipped by default)")
    sp_x.add_argument("--case-insensitive", action="store_true", help="compare string values case-insensitively")
    sp_x.set_defaults(func=_cmd_conflicts)

    sp_l = sub.add_parser(
        "lint",
        help="report deterministic vault-hygiene issues (broken links, orphans, duplicates, stale)",
    )
    sp_l.add_argument("--vault", help="vault dir to scan (default: resolved capture vault)")
    sp_l.add_argument("--format", choices=("text", "json"), default="text", help="output format")
    sp_l.add_argument("--stale", type=int, default=0, help="flag notes whose mtime is older than N days (0=off)")
    sp_l.add_argument("--stub-chars", type=int, default=0, dest="stub_chars", help="flag notes whose body is <= N chars (default 0 = empty only)")
    sp_l.add_argument("--include-sessions", action="store_true", help="include session-* notes in orphan/stub/stale checks")
    sp_l.set_defaults(func=_cmd_lint)

    sp_r = sub.add_parser("recall", help="recall the note-chain that answers a query (supersession-aware; --as-of time-travel)")
    sp_r.add_argument("query", help="the question or topic to recall")
    sp_r.add_argument("--vault", help="vault dir (default: resolved capture vault)")
    sp_r.add_argument("--as-of", dest="as_of", help="show the vault as it was valid at this date (YYYY-MM-DD)")
    sp_r.add_argument("--show-superseded", action="store_true", dest="show_superseded", help="include superseded note bodies")
    sp_r.add_argument("--top-k", type=int, default=8, dest="top_k", help="max note chunks (default 8)")
    sp_r.set_defaults(func=_cmd_recall)

    sp_su = sub.add_parser("supersede", help="mark OLD note as replaced by NEW (invalidate-don't-delete)")
    sp_su.add_argument("old", help="the superseded note (stem, slug, or path)")
    sp_su.add_argument("new", help="the current note that replaces it (stem, slug, or path)")
    sp_su.add_argument("--vault", help="vault dir (default: resolved capture vault)")
    sp_su.add_argument("--reason", default="", help="audit note for why OLD was superseded")
    sp_su.add_argument("--as-of", dest="as_of", help="date the fact stopped being true (YYYY-MM-DD, default today)")
    sp_su.add_argument("--reverse", action="store_true", help="also stamp replaces: [[OLD]] on NEW")
    sp_su.add_argument("--undo", action="store_true", help="remove a prior supersession from OLD")
    sp_su.add_argument("--dry-run", action="store_true", dest="dry_run", help="show the frontmatter diff, write nothing")
    sp_su.set_defaults(func=_cmd_supersede)

    sp_d = sub.add_parser("dedup", help="find exact + near-duplicate notes (Jaccard/MinHash, deterministic)")
    sp_d.add_argument("--vault", help="vault dir to scan (default: resolved capture vault)")
    sp_d.add_argument("--format", choices=("text", "json"), default="text", help="output format")
    sp_d.add_argument("--threshold", type=float, default=0.8, help="near-duplicate Jaccard threshold (default 0.8)")
    sp_d.add_argument("--shingle-size", type=int, default=5, dest="shingle_size", help="word k-gram size (default 5)")
    sp_d.add_argument("--include-sessions", action="store_true", help="also scan session-* notes (skipped by default)")
    sp_d.set_defaults(func=_cmd_dedup)

    sp_de = sub.add_parser("decay", help="list the fading review queue (cold notes by decay+access+links; read-only, never deletes)")
    sp_de.add_argument("--vault", help="vault dir to scan (default: resolved capture vault)")
    sp_de.add_argument("--format", choices=("text", "json"), default="text", help="output format")
    sp_de.add_argument("--tau-days", type=float, default=90.0, dest="tau_days", help="decay time constant in days (default 90)")
    sp_de.add_argument("--threshold", type=float, default=0.25, help="fading strength threshold (default 0.25)")
    sp_de.add_argument("--limit", type=int, default=50, help="max fading notes to show (default 50)")
    sp_de.add_argument("--include-sessions", action="store_true", help="also scan session-* notes (skipped by default)")
    sp_de.set_defaults(func=_cmd_decay)

    sp_c = sub.add_parser("capture", help="run a hook event manually (reads stdin JSON)")
    sp_c.add_argument("event", help="hook event name, e.g. SessionStart")
    sp_c.set_defaults(func=_cmd_capture)

    return p


def main(argv: list[str] | None = None) -> int:
    # Vault note values may carry non-ASCII; never let a cp1252 console raise on
    # print. Safe for every command (the MCP/capture stdout is ASCII JSON anyway).
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover - already-detached stream
            pass
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
