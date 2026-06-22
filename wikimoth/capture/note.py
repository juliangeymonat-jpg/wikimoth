# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Note builder — turn a session buffer into one deterministic markdown note.

:func:`build_note` is a **pure function** of (events, vault index): same inputs →
byte-identical note. The body's prose may optionally be drafted by an LLM, but
its ``[[wikilinks]]`` are stripped (see :func:`_strip_wikilinks`) and the real
links are computed by :mod:`wikimoth.capture.links`. So the graph the retriever
walks is reproducible regardless of whether prose was model-written.

:func:`write_session_note` is the thin on-disk wrapper the hook calls: it reads
the buffer, scans the vault for the link index, builds the note, and writes it.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from wikimoth.capture.config import CaptureConfig, _safe_id
from wikimoth.capture.links import (
    _slugify,
    compute_links,
    files_touched,
    is_session_stem,
)

# Matches the same three ``[[...]]`` forms MOTHRAG's GraphRetriever parses, so we
# can guarantee no model-drafted prose smuggles an edge into the graph.
_WIKILINK = re.compile(r"\[\[[^\]]*\]\]")


def _strip_wikilinks(text: str) -> str:
    """Remove any ``[[...]]`` token (and stray brackets) from model prose."""
    s = _WIKILINK.sub("", text or "")
    s = s.replace("[[", "").replace("]]", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def _deactivate_wikilinks(text: str) -> str:
    """Neutralise ``[[`` / ``]]`` in echoed user/tool text so it can never
    become a graph edge, while keeping the text readable.

    The note body echoes the prompt (description + Prompts) and touched file
    paths verbatim. If any of those contained ``[[x]]`` the retriever would
    parse a *real* edge that :func:`compute_links` never authored — silently
    breaking WikiMoth's core invariant (edges are computed by code, not by a
    model or by arbitrary input). We break any run of 2+ ``[``/``]`` by
    interleaving a space, so no ``[[``/``]]`` survives anywhere outside the
    code-computed ``## Links`` section, while single brackets (``arr[i]``) stay
    intact.
    """
    s = str(text or "")
    s = re.sub(r"\[{2,}", lambda m: " ".join("[" * len(m.group(0))), s)
    s = re.sub(r"\]{2,}", lambda m: " ".join("]" * len(m.group(0))), s)
    return s


def _yaml_str(s: str) -> str:
    """Quote a scalar for YAML frontmatter (double-quoted, escaped)."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _first_line(s: str, cap: int = 80) -> str:
    one = " ".join(str(s or "").split())
    return one if len(one) <= cap else one[:cap].rstrip() + "…"


def _session_start(events: Sequence[Mapping]) -> Mapping:
    for rec in events:
        if isinstance(rec, Mapping) and rec.get("event") == "SessionStart":
            return rec
    return {}


def _earliest_ts(events: Sequence[Mapping]) -> str:
    ts = [str(r.get("ts")) for r in events if isinstance(r, Mapping) and r.get("ts")]
    return min(ts) if ts else ""


def build_note(
    events: Sequence[Mapping],
    *,
    session_id: str,
    note_index: Mapping[str, str],
    session_stems: Sequence[str],
    config: CaptureConfig | None = None,
    llm: Callable[[str], str] | None = None,
) -> tuple[str, str]:
    """Build ``(stem, content)`` for the session note. Pure given its inputs.

    Parameters
    ----------
    events
        The session buffer records (see :class:`SessionBuffer`).
    session_id
        The session id (drives the deterministic filename suffix).
    note_index
        ``slug -> filename_stem`` for the vault's **non-session** notes (the
        link targets). Session notes are passed separately via ``session_stems``.
    session_stems
        Existing session-note stems (for the previous-session chain edge).
    llm
        Optional ``str -> str`` prose drafter. Only used when
        ``config.enable_llm_prose``; its output is stripped of wikilinks.
    """
    cfg = config or CaptureConfig(vault_dir=Path("."))
    start = _session_start(events)
    ts = str(start.get("ts") or _earliest_ts(events))
    date = ts[:10] if ts else "0000-00-00"
    short_id = _safe_id(session_id)[:8]
    self_stem = f"session-{date}-{short_id}"

    cwd = str(start.get("cwd") or "")
    cwd_safe = _deactivate_wikilinks(cwd)
    source = str(start.get("source") or "")

    prompts = [str(r["prompt"]) for r in events
               if isinstance(r, Mapping) and r.get("event") == "UserPromptSubmit" and r.get("prompt")]
    tool_recs = [r for r in events
                 if isinstance(r, Mapping) and r.get("event") == "PostToolUse"]
    tools = Counter(str(r.get("tool") or "?") for r in tool_recs)
    files = files_touched(events)

    links = compute_links(
        self_stem=self_stem,
        events=events,
        note_index=note_index,
        session_stems=session_stems,
        max_match_chars=cfg.max_match_chars,
        min_token_len=cfg.min_link_token_len,
    )

    description = _first_line(prompts[0]) if prompts else (
        f"{len(tool_recs)} tool calls, {len(files)} files touched"
    )
    description = _deactivate_wikilinks(description)

    # ---- frontmatter ----
    tools_inline = "{" + ", ".join(f"{k}: {v}" for k, v in sorted(tools.items())) + "}"
    fm = [
        "---",
        f"name: {self_stem}",
        f"description: {_yaml_str(description)}",
        "metadata:",
        "  type: session",
        f"  session_id: {_yaml_str(session_id)}",
        f"  date: {date}",
        f"  cwd: {_yaml_str(cwd_safe)}",
        f"  source: {_yaml_str(source)}",
        f"  tools: {tools_inline}",
        f"  files: {len(files)}",
        "---",
    ]

    # ---- body ----
    tool_summary = ""
    if tools:
        tool_summary = " across " + ", ".join(sorted(tools))
    lines = [
        "",
        f"# Session {date} ({short_id})",
        "",
        f"- **When:** {ts or '—'}  ·  **Source:** {source or '—'}",
        f"- **Where:** {cwd_safe or '—'}",
        f"- {len(prompts)} prompt(s), {len(tool_recs)} tool call(s)"
        f"{tool_summary}, {len(files)} file(s) touched",
        "",
        "## Summary",
    ]

    prose = ""
    if cfg.enable_llm_prose and llm is not None:
        try:
            facts = (
                f"Session prompts: {prompts}\n"
                f"Tools: {dict(tools)}\nFiles touched: {files}\n"
                "Write ONE plain sentence summarising what this session did. "
                "Do not use [[ ]] links."
            )
            prose = _strip_wikilinks(llm(facts))
        except Exception:
            prose = ""
    if prose:
        lines += [prose, ""]
    else:
        lines += ["_(deterministic summary; no model prose)_", ""]

    if prompts:
        lines.append("## Prompts")
        lines += [f"- {_deactivate_wikilinks(_first_line(p, 200))}" for p in prompts]
        lines.append("")

    if files:
        lines.append("## Files touched")
        lines += [f"- `{_deactivate_wikilinks(f)}`" for f in files]
        lines.append("")

    lines.append("## Links")
    if links:
        lines += [f"[[{stem}]]" for stem in links]
    else:
        lines.append("_No links resolved this session._")
    lines.append("")

    content = "\n".join(fm + lines)
    if not content.endswith("\n"):
        content += "\n"
    return self_stem, content


def _scan_vault(config: CaptureConfig) -> tuple[dict[str, str], list[str]]:
    """Return ``(note_index, session_stems)`` from the vault on disk.

    ``note_index`` is ``slug -> stem`` for non-session notes (link targets);
    ``session_stems`` is the list of existing session-note stems. The
    ``.sessions`` buffer dir is skipped.
    """
    note_index: dict[str, str] = {}
    session_stems: list[str] = []
    vault = config.vault_dir
    if not vault.exists():
        return note_index, session_stems
    sessions_dir = config.sessions_dir.resolve()
    for p in sorted(vault.rglob("*.md")):
        try:
            if sessions_dir in p.resolve().parents:
                continue
        except OSError:  # pragma: no cover - exotic path
            pass
        stem = p.stem
        if is_session_stem(stem):
            session_stems.append(stem)
        else:
            note_index[_slugify(stem)] = stem
    return note_index, session_stems


def write_session_note(
    config: CaptureConfig,
    session_id: str,
    *,
    llm: Callable[[str], str] | None = None,
) -> Path | None:
    """Read the session buffer, build the note, write it. Returns the path.

    Returns ``None`` if the buffer is empty/absent. Idempotent: safe to call on
    every Stop and again on SessionEnd — it overwrites the same file from the
    full buffer each time.
    """
    from wikimoth.capture.buffer import SessionBuffer

    events = SessionBuffer.for_session(config, session_id).read()
    if not events:
        return None

    note_index, session_stems = _scan_vault(config)
    stem, content = build_note(
        events,
        session_id=session_id,
        note_index=note_index,
        session_stems=session_stems,
        config=config,
        llm=llm,
    )
    config.vault_dir.mkdir(parents=True, exist_ok=True)
    path = config.note_path(stem)
    path.write_text(content, encoding="utf-8")
    return path


__all__ = ["build_note", "write_session_note"]
