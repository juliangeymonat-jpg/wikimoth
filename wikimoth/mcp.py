# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MCP server — put WikiMoth's deterministic recall *in the agent loop*.

``wikimoth serve`` shows a human the note-chain behind an answer; this module
exposes the same retrieval to a model over the Model Context Protocol, so Claude
(or any MCP client) calls ``recall(query)`` itself before answering, pulls the
exact ``[[wikilink]]`` note-chain with no LLM call to retrieve, and can show its
work. It is a hand-rolled, pure-stdlib JSON-RPC 2.0 server over stdio (newline
delimited, the MCP stdio transport), so the core stays dependency-free.

Two tools:

* ``recall(query, top_k?)`` -> the deterministic note-chain that answers
  ``query``: the actual note text, per-chunk hop distance and token count, and
  the ``-N%`` vs dumping the whole vault. This is what the agent reads to answer.
* ``status()`` -> the connected vault, note/chunk counts, token backend.

The protocol layer is :meth:`_Server.handle` (one JSON-RPC message in, one
response dict or ``None`` out): pure and unit-testable with no subprocess or
socket. :func:`serve_stdio` is the thin stdin/stdout pump around it.

stdout is the protocol channel: nothing here writes anything but JSON-RPC to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from wikimoth import __version__
from wikimoth.conflicts import format_conflicts, scan_conflicts
from wikimoth.decay import format_decay, scan_decay
from wikimoth.dedup import format_dedup, scan_dedup
from wikimoth.lint import format_lint, scan_lint
from wikimoth.pipeline import MemoryRAG, _load_vault_chunks, _parse_iso_date
from wikimoth.supersede import SupersedeError, format_result, supersede
from wikimoth.tokens import count_passage_tokens, token_backend

# The MCP protocol revisions this server actually implements. Per the MCP
# lifecycle spec, initialize MUST respond with a version the server supports:
# a supported request is echoed, anything else gets our default. 2025-03-26 is
# deliberately absent: that revision requires JSON-RPC batch support, which
# this server rejects by design.
_PROTOCOL_VERSION = "2025-06-18"
_SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2024-11-05", _PROTOCOL_VERSION})

TOOLS = [
    {
        "name": "recall",
        "description": (
            "Deterministically recall the note-chain from the user's WikiMoth "
            "[[wikilink]] memory that answers a question. Walks the authored links "
            "(multi-hop), returns the exact notes with NO LLM call to retrieve, far "
            "fewer tokens than dumping the whole vault, and the same result every "
            "time. Call this before answering anything that might live in the user's "
            "notes, memory, or past sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "the question or topic to recall from memory",
                },
                "top_k": {
                    "type": "integer",
                    "description": "max note chunks to return (default 8)",
                },
                "as_of": {
                    "type": "string",
                    "description": "time-travel: show the vault as it was valid at this ISO date (YYYY-MM-DD)",
                },
                "show_superseded": {
                    "type": "boolean",
                    "description": "include superseded note bodies (default false hides them, keeping the edge)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "status",
        "description": (
            "Report the connected WikiMoth memory vault: path, note and chunk "
            "counts, whole-vault token size, and the token backend. Use to confirm "
            "memory is wired up."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_conflicts",
        "description": (
            "Deterministically list contradiction CANDIDATES in the WikiMoth vault: "
            "notes that assert different values for the same (subject, predicate). "
            "No model finds them. Each candidate is for YOU to adjudicate: decide if "
            "it is a real contradiction and which note is current. Notes tagged "
            "valid-time 'disjoint' are likely a legitimate succession (consider "
            "superseding), 'overlapping' is a real conflict. Use before trusting a "
            "possibly-stale fact recalled from memory."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_inline": {
                    "type": "boolean",
                    "description": "also read Dataview 'key:: value' inline body fields",
                },
                "all_keys": {
                    "type": "boolean",
                    "description": "compare every non-subject key, not just domain facts",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_lint",
        "description": (
            "Deterministically report vault-hygiene issues: broken [[links]], "
            "orphan notes, duplicate identities, empty stubs, stale/expired notes, "
            "and supersession chains/cycles. No model. Read-only. Use to check the "
            "memory's structural health or before trusting a possibly-broken link."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stale_days": {
                    "type": "integer",
                    "description": "also flag notes whose mtime is older than N days (0 = off)",
                },
                "include_sessions": {
                    "type": "boolean",
                    "description": "include session-* notes in orphan/stub/stale checks",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_duplicates",
        "description": (
            "Deterministically find exact and near-duplicate notes (MinHash/Jaccard, "
            "no model). WikiMoth capture is append-only and never merges, so content "
            "gets restated; this surfaces it. Candidates only: decide which to keep "
            "(consider superseding the older one)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "near-duplicate Jaccard threshold in (0,1] (default 0.8)",
                },
                "include_sessions": {
                    "type": "boolean",
                    "description": "also scan session-* notes (skipped by default)",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_fading",
        "description": (
            "Deterministically list the 'fading' review queue: notes going cold "
            "(old, rarely linked, rarely recalled), scored by a decay + access + "
            "connectivity strength. Read-only, nothing is deleted. Use to suggest "
            "what the user might archive, refresh, or supersede."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "fading strength threshold (default 0.25)",
                },
                "tau_days": {
                    "type": "number",
                    "description": "decay time constant in days (default 90)",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "supersede",
        "description": (
            "Mark an OLD note as replaced by a NEW one, WITHOUT deleting it "
            "(invalidate-don't-delete: the file stays, its frontmatter records "
            "superseded_by/valid_to/status). Call this AFTER you have adjudicated "
            "that NEW genuinely replaces OLD (e.g. from a list_conflicts candidate). "
            "OLD/NEW are note stems, slugs, or paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "old": {"type": "string", "description": "the superseded note (stem/slug/path)"},
                "new": {"type": "string", "description": "the current note that replaces it"},
                "reason": {"type": "string", "description": "optional audit note"},
                "reverse": {"type": "boolean", "description": "also stamp replaces: on NEW"},
            },
            "required": ["old", "new"],
            "additionalProperties": False,
        },
    },
]


def _build_index(vault_dir: Path) -> tuple[MemoryRAG, int, int, int]:
    """Index ``vault_dir`` once; return ``(rag, total_tokens, n_notes, n_chunks)``.

    ``total_tokens`` is the whole-vault token count (the dump baseline for the
    ``-N%`` figure), computed over the same chunks that are indexed so the two
    never disagree.
    """
    chunks = _load_vault_chunks(vault_dir)
    total = count_passage_tokens([getattr(c, "text", "") or "" for c in chunks])
    rag = MemoryRAG()
    rag.index_chunks(chunks)
    n_notes = len({getattr(c, "doc_id", "") for c in chunks if getattr(c, "doc_id", "")})
    return rag, total, n_notes, len(chunks)


def _tool_text(text: str, *, is_error: bool = False) -> dict:
    """A ``tools/call`` result carrying a single text block."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _err(mid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _format_recall(query: str, pairs, per_chunk, fed: int, total: int, pct: float) -> str:
    header = f'WikiMoth recall for: "{query}"\n'
    if not pairs:
        return header + (
            "No notes matched. Try different words, or this vault has no [[link]] "
            "path to an answer."
        )
    stats = (
        f"{len(pairs)} note chunk(s), ~{fed:,} tokens fed "
        f"(-{pct:.1f}% vs dumping the whole vault, ~{total:,}). "
        "Deterministic, no LLM call to retrieve.\n"
    )
    blocks = []
    for i, ((c, hop), tok) in enumerate(zip(pairs, per_chunk), 1):
        meta = getattr(c, "metadata", None) or {}
        name = meta.get("filename") or (getattr(c, "doc_id", "") or "note")
        hop_label = "seed" if hop == 0 else f"hop {hop}"
        body = getattr(c, "text", "") or ""
        blocks.append(f"[{i}] {name} ({hop_label}, {tok} tok)\n{body}")
    return header + stats + "\n" + "\n\n".join(blocks)


class _Server:
    """Protocol core: ``handle(message) -> response dict | None``. Pure + testable.

    Holds a lazily-built, change-detecting index of the vault so repeated recalls
    are cheap but new captures are picked up: the index is rebuilt only when the
    set of ``.md`` files or their newest mtime changes.
    """

    def __init__(self, vault_dir: str | Path, *, default_top_k: int = 8) -> None:
        self.vault_dir = Path(vault_dir)
        self.default_top_k = max(1, int(default_top_k))
        self._cache: tuple | None = None  # (fingerprint, rag, total, n_notes, n_chunks)

    # -- indexing -------------------------------------------------------
    def _fingerprint(self) -> tuple[int, int]:
        if not self.vault_dir.is_dir():
            return (0, 0)
        files = list(self.vault_dir.rglob("*.md"))
        newest = max((p.stat().st_mtime_ns for p in files), default=0)
        return (len(files), newest)

    def _index(self) -> tuple:
        fp = self._fingerprint()
        if self._cache is None or self._cache[0] != fp:
            rag, total, n_notes, n_chunks = _build_index(self.vault_dir)
            self._cache = (fp, rag, total, n_notes, n_chunks)
        return self._cache

    # -- JSON-RPC dispatch ---------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        if not isinstance(msg, dict):
            # Valid JSON but not a JSON-RPC object (a bare array, scalar, or null).
            # We don't support batches; reject without crashing the loop.
            return _err(None, -32600, "invalid request: expected a JSON-RPC object")
        method = msg.get("method")
        mid = msg.get("id")  # absent on notifications
        if not isinstance(method, str):
            return None  # a response or garbage; ignore
        params = msg.get("params")
        if not isinstance(params, dict):
            params = {}  # params MUST be an object; coerce malformed ones
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._tools_call(params)
            elif method.startswith("notifications/"):
                return None  # notifications get no response
            else:
                return None if mid is None else _err(mid, -32601, f"method not found: {method}")
        except Exception as e:  # noqa: BLE001 - never crash the loop on one message
            return None if mid is None else _err(mid, -32603, f"{type(e).__name__}: {e}")
        if mid is None:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        negotiated = requested if requested in _SUPPORTED_PROTOCOL_VERSIONS else _PROTOCOL_VERSION
        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "wikimoth", "version": __version__},
        }

    def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments")
        if not isinstance(args, dict):
            args = {}
        if name == "recall":
            return self._recall(args)
        if name == "status":
            return self._status()
        if name == "list_conflicts":
            return self._list_conflicts(args)
        if name == "list_lint":
            return self._list_lint(args)
        if name == "list_duplicates":
            return self._list_duplicates(args)
        if name == "list_fading":
            return self._list_fading(args)
        if name == "supersede":
            return self._supersede(args)
        return _tool_text(f"Unknown tool: {name!r}", is_error=True)

    def _recall(self, args: dict) -> dict:
        query = args.get("query")
        query = query.strip() if isinstance(query, str) else ""
        if not query:
            return _tool_text("recall needs a non-empty 'query' string.", is_error=True)
        top_k = args.get("top_k")
        # bool is a subclass of int (True==1), so reject it explicitly.
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            top_k = self.default_top_k
        if not self.vault_dir.is_dir():
            return _tool_text(
                f"No WikiMoth vault at {self.vault_dir}. Run `wikimoth install` to "
                "start capturing one, or start the server with --vault PATH.",
                is_error=True,
            )
        _, rag, total, _, _ = self._index()
        # Per-request supersession-aware / as-of view (the index is shared/cached).
        as_of = args.get("as_of")
        if isinstance(as_of, str) and as_of.strip():
            parsed = _parse_iso_date(as_of)
            if parsed is None:
                # Don't silently fall back to the present: an agent asking for a
                # historical view must not be handed current-view notes.
                return _tool_text(f"recall: as_of must be YYYY-MM-DD, got {as_of!r}", is_error=True)
            rag.as_of = parsed
        else:
            rag.as_of = None
        rag.show_superseded = bool(args.get("show_superseded"))
        pairs = rag.retrieve_with_hops(query, top_k=top_k)
        per_chunk = [count_passage_tokens([getattr(c, "text", "") or ""]) for c, _ in pairs]
        fed = sum(per_chunk)
        pct = (100.0 * (1 - fed / total)) if total else 0.0
        return _tool_text(_format_recall(query, pairs, per_chunk, fed, total, pct))

    def _list_conflicts(self, args: dict) -> dict:
        if not self.vault_dir.is_dir():
            return _tool_text(
                f"No WikiMoth vault at {self.vault_dir}. Run `wikimoth install` to "
                "start capturing one, or start the server with --vault PATH.",
                is_error=True,
            )
        report = scan_conflicts(
            self.vault_dir,
            include_inline=bool(args.get("include_inline")),
            all_keys=bool(args.get("all_keys")),
        )
        return _tool_text(format_conflicts(report, fmt="text"))

    def _list_lint(self, args: dict) -> dict:
        if not self.vault_dir.is_dir():
            return _tool_text(
                f"No WikiMoth vault at {self.vault_dir}. Run `wikimoth install` to "
                "start capturing one, or start the server with --vault PATH.",
                is_error=True,
            )
        stale = args.get("stale_days")
        if isinstance(stale, bool) or not isinstance(stale, int) or stale < 0:
            stale = 0
        report = scan_lint(
            self.vault_dir,
            stale_days=stale,
            include_sessions=bool(args.get("include_sessions")),
        )
        return _tool_text(format_lint(report, fmt="text"))

    def _list_duplicates(self, args: dict) -> dict:
        if not self.vault_dir.is_dir():
            return _tool_text(
                f"No WikiMoth vault at {self.vault_dir}. Run `wikimoth install` to "
                "start capturing one, or start the server with --vault PATH.",
                is_error=True,
            )
        threshold = args.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not (0 < threshold <= 1):
            threshold = 0.8
        report = scan_dedup(
            self.vault_dir,
            threshold=float(threshold),
            include_sessions=bool(args.get("include_sessions")),
        )
        return _tool_text(format_dedup(report, fmt="text"))

    def _list_fading(self, args: dict) -> dict:
        if not self.vault_dir.is_dir():
            return _tool_text(
                f"No WikiMoth vault at {self.vault_dir}. Run `wikimoth install` to "
                "start capturing one, or start the server with --vault PATH.",
                is_error=True,
            )
        kw: dict = {}
        thr = args.get("threshold")
        if isinstance(thr, (int, float)) and not isinstance(thr, bool) and thr > 0:
            kw["threshold"] = float(thr)
        tau = args.get("tau_days")
        if isinstance(tau, (int, float)) and not isinstance(tau, bool) and tau > 0:
            kw["tau_days"] = float(tau)
        report = scan_decay(self.vault_dir, **kw)
        return _tool_text(format_decay(report, fmt="text"))

    def _supersede(self, args: dict) -> dict:
        old = args.get("old")
        new = args.get("new")
        if not isinstance(old, str) or not old.strip() or not isinstance(new, str) or not new.strip():
            return _tool_text("supersede needs non-empty 'old' and 'new' note refs.", is_error=True)
        reason = args.get("reason")
        try:
            result = supersede(
                self.vault_dir, old.strip(), new.strip(),
                reason=reason.strip() if isinstance(reason, str) else "",
                reverse=bool(args.get("reverse")),
            )
        except SupersedeError as e:
            return _tool_text(f"supersede: {e}", is_error=True)
        return _tool_text(format_result(result))

    def _status(self) -> dict:
        if not self.vault_dir.is_dir():
            return _tool_text(
                f"vault: {self.vault_dir} (not found)\n"
                "Run `wikimoth install` to start capturing, or restart with --vault PATH."
            )
        _, _, total, n_notes, n_chunks = self._index()
        text = (
            "WikiMoth memory\n"
            f"vault: {self.vault_dir}\n"
            f"notes: {n_notes}\n"
            f"chunks: {n_chunks}\n"
            f"whole-vault tokens: {total:,}\n"
            f"token backend: {token_backend()}\n"
            f"default top_k: {self.default_top_k}"
        )
        return _tool_text(text)


def _write(stdout: TextIO, obj: dict) -> None:
    stdout.write(json.dumps(obj) + "\n")
    stdout.flush()


def serve_stdio(
    vault_dir: str | Path,
    *,
    default_top_k: int = 8,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the MCP server over stdio until stdin closes.

    Reads one newline-delimited JSON-RPC message per line, dispatches it through
    :class:`_Server`, and writes each response back as one line. ``stdin``/``stdout``
    are injectable so the full loop is testable in-process (no subprocess).
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    server = _Server(vault_dir, default_top_k=default_top_k)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp: dict | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            # Defense in depth: handle() already guards itself, but one message
            # must never be able to kill the loop.
            try:
                resp = server.handle(msg)
            except Exception as e:  # noqa: BLE001
                resp = _err(None, -32603, f"{type(e).__name__}: {e}")
        if resp is None:
            continue
        try:
            _write(stdout, resp)
        except (BrokenPipeError, OSError, ValueError):
            # Client disconnected / stdout closed: stop cleanly, do not raise.
            break
    return 0


__all__ = ["TOOLS", "serve_stdio"]
