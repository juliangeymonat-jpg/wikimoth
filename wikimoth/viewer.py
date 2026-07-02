# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth serve`` -- a zero-dependency local web viewer for a WikiMoth vault.

Pure Python standard library (``http.server`` + ``html`` + ``urllib``): no Flask,
no JS framework, no CDN, no network. It renders the *plain-markdown* vault WikiMoth
already produces, so it is a thin, honest window onto files you can equally open
in Obsidian or VS Code -- not a lock-in store.

Three views:

* **/** -- note browser + search box + vault stats (notes, edges, total tokens).
* **/note?slug=...** -- one note, with its ``[[wikilinks]]`` rendered as links
  and its backlinks listed.
* **/graph** -- the authored ``[[wikilink]]`` graph as a deterministic SVG
  (same edges the retriever walks -- ``_WIKILINK_RE`` + ``_slugify`` are reused
  from :mod:`wikimoth.retrieval.graph`).
* **/ask?q=...** -- the money view: runs *retrieval only* (no reader, no API key)
  and shows the exact note-chain WikiMoth would feed a reader for that question,
  with per-chunk hop distance + token accounting and the ``-N%`` vs dumping the
  whole vault. This is "what memory fed this answer", made visible and
  deterministic.

Design notes:
- The render functions are pure (``model``/``rag`` in, HTML string out) so they
  are unit-testable without opening a socket; :func:`route` dispatches to them.
- All note-derived text is escaped with :func:`html.escape`.
- The server binds to ``127.0.0.1`` by default -- your memory stays local.
"""

from __future__ import annotations

import html
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from wikimoth.frontmatter import parse_frontmatter
from wikimoth.retrieval.graph import _WIKILINK_RE, _slugify
from wikimoth.tokens import count_passage_tokens, token_backend

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TOP_K = 8

HTML_CTYPE = "text/html; charset=utf-8"


def _q(slug: str) -> str:
    """URL-encode a slug for an ``?slug=`` href value.

    ``quote(safe="")`` percent-encodes spaces and ``&`` so a slug like
    ``"rock & roll"`` round-trips through the browser and :func:`parse_qs`
    intact (a bare ``&`` would otherwise start a new query parameter). The
    output contains only ``%XX`` + ``[A-Za-z0-9_.-~]``, so it is also safe
    inside a single-quoted HTML attribute without further escaping.
    """
    return quote(slug, safe="")


# ----------------------------------------------------------------------
# Vault model (scanned once, deterministic)
# ----------------------------------------------------------------------


class Note:
    """One markdown note: its identity, text, resolved links and backlinks."""

    __slots__ = ("slug", "filename", "path", "text", "tokens", "links", "backlinks")

    def __init__(self, slug: str, filename: str, path: str, text: str) -> None:
        self.slug = slug
        self.filename = filename
        self.path = path
        self.text = text
        self.tokens = count_passage_tokens([text])
        self.links: list[str] = []  # resolved target slugs (present in vault)
        self.backlinks: list[str] = []  # slugs that link to this note


class VaultModel:
    """Deterministic view of a ``[[wikilink]]`` vault for the viewer.

    Built by scanning ``*.md`` under ``vault_dir`` and parsing ``[[wikilinks]]``
    with the *same* regex/slugify the retriever uses, so the displayed graph
    matches the retrieved graph.
    """

    def __init__(self, notes: dict[str, Note]) -> None:
        self.notes = notes
        # The model is built once and never mutated, so fold the stats here
        # instead of recomputing them on every request.
        self.n_notes = len(notes)
        self.n_edges = sum(len(n.links) for n in notes.values())
        self.total_tokens = sum(n.tokens for n in notes.values())

    # -- construction --------------------------------------------------

    @classmethod
    def from_dir(cls, vault_dir: str | Path) -> "VaultModel":
        vault = Path(vault_dir)
        if not vault.is_dir():
            raise FileNotFoundError(f"vault_dir not found (or not a directory): {vault}")

        notes: dict[str, Note] = {}
        for p in sorted(vault.rglob("*.md")):
            slug = _slugify(p.name)
            if not slug or slug in notes:
                # Skip unslugifiable names and duplicate slugs. First file wins;
                # deterministic because rglob output is sorted.
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            notes[slug] = Note(slug, p.name, str(p), text)

        cls._wire_edges(notes)
        return cls(notes)

    @staticmethod
    def _wire_edges(notes: dict[str, Note]) -> None:
        """Resolve ``[[wikilinks]]`` to in-vault notes and compute backlinks."""
        for slug, note in notes.items():
            seen: set[str] = set()
            for raw in _WIKILINK_RE.findall(note.text):
                target = _slugify(raw)
                if target and target != slug and target in notes and target not in seen:
                    seen.add(target)
            note.links = sorted(seen)
        for slug, note in notes.items():
            for target in note.links:
                notes[target].backlinks.append(slug)
        for note in notes.values():
            # Each (source, target) pair is appended at most once (links are
            # deduped per note above), so backlinks need only sorting.
            note.backlinks = sorted(note.backlinks)

# ----------------------------------------------------------------------
# HTML rendering (pure functions)
# ----------------------------------------------------------------------

_CSS = """
:root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--good:#3fb950;--line:#30363d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
header .brand{font-weight:700;font-size:18px;letter-spacing:.3px}
header nav a{margin-right:14px;color:var(--muted)}
main{max-width:980px;margin:0 auto;padding:24px}
.stats{display:flex;gap:24px;flex-wrap:wrap;margin:0 0 20px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;min-width:120px}
.stat b{display:block;font-size:24px}.stat span{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.5px}
form.ask{display:flex;gap:8px;margin:0 0 24px}
form.ask input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:10px 12px;font-size:15px}
form.ask button{background:var(--accent);color:#0d1117;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer}
ul.notes{list-style:none;padding:0;columns:2;gap:24px}
ul.notes li{margin:0 0 6px;break-inside:avoid}
.muted{color:var(--muted)}
.note-body{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;white-space:pre-wrap;word-wrap:break-word;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px}
.chunk{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 14px}
.chunk .meta{display:flex;gap:10px;align-items:center;margin:0 0 8px;flex-wrap:wrap}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--muted)}
.badge.hop{color:var(--good);border-color:var(--good)}
.chunk pre{white-space:pre-wrap;word-wrap:break-word;margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;color:var(--ink)}
.savings{font-size:15px;margin:0 0 18px}.savings b{color:var(--good);font-size:20px}
svg{max-width:100%;height:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
.foot{margin:28px 0 0;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
"""


def _page(title: str, body: str) -> str:
    """Wrap ``body`` in the shared shell (escaped title)."""
    return (
        "<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        "<header><span class=brand>WikiMoth</span>"
        "<nav><a href='/'>notes</a><a href='/graph'>graph</a></nav>"
        "<span class=muted style='margin-left:auto'>deterministic, plain-markdown memory</span>"
        "</header><main>" + body +
        "<div class=foot>Served from plain-markdown files &mdash; open the same vault in Obsidian or VS Code. "
        "Nothing here calls an LLM or the network.</div>"
        "</main></body></html>"
    )


def _ask_form(q: str = "") -> str:
    return (
        "<form class=ask method=get action='/ask'>"
        f"<input type=text name=q placeholder='Ask your memory a question…' value=\"{html.escape(q)}\" autofocus>"
        "<button type=submit>What memory feeds this?</button></form>"
    )


def render_home(model: VaultModel) -> str:
    items = "".join(
        f"<li><a href='/note?slug={_q(n.slug)}'>{html.escape(n.filename)}</a> "
        f"<span class=muted>· {n.tokens} tok · {len(n.links)} links</span></li>"
        for n in sorted(model.notes.values(), key=lambda x: x.slug)
    )
    body = (
        "<div class=stats>"
        f"<div class=stat><b>{model.n_notes}</b><span>notes</span></div>"
        f"<div class=stat><b>{model.n_edges}</b><span>wikilink edges</span></div>"
        f"<div class=stat><b>{model.total_tokens:,}</b><span>total tokens</span></div>"
        "</div>"
        + _ask_form()
        + f"<p class=muted>{model.n_notes} notes in this vault. Click one, explore the graph, "
        "or ask a question to see the exact note-chain it would feed a reader.</p>"
        f"<ul class=notes>{items}</ul>"
    )
    return _page("WikiMoth — vault", body)


def _linkify(text: str) -> str:
    """Turn ``[[target]]`` tokens into anchors; escape everything else.

    The match is run on the **raw** text (not pre-escaped), so the link target
    is slugified from the real characters (``[[Rock & Roll]]`` -> ``rock & roll``,
    not the HTML entity) and the href slug is URL-encoded. Non-link spans and the
    visible label are HTML-escaped exactly once.
    """
    out: list[str] = []
    last = 0
    for m in _WIKILINK_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        raw = m.group(1)
        out.append(
            f"[[<a href='/note?slug={_q(_slugify(raw))}'>{html.escape(raw)}</a>]]"
        )
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def render_note(model: VaultModel, slug: str) -> tuple[int, str]:
    note = model.notes.get(_slugify(slug))
    if note is None:
        return 404, _page(
            "not found",
            f"<p>No note named <code>{html.escape(slug)}</code> in this vault. "
            "<a href='/'>Back to all notes</a>.</p>",
        )
    backlinks = (
        "".join(
            f"<li><a href='/note?slug={_q(s)}'>{html.escape(model.notes[s].filename)}</a></li>"
            for s in note.backlinks
        )
        or "<li class=muted>none</li>"
    )
    # Browser shows every note, but flag the ones recall now hides so the view
    # is not silently inconsistent with /ask.
    fm = parse_frontmatter(note.text)
    banner = ""
    sup = fm.get("superseded_by", "")
    if sup or fm.get("status") == "superseded":
        m = _WIKILINK_RE.search(sup)
        link = (
            f" by <a href='/note?slug={_q(_slugify(m.group(1)))}'>{html.escape(m.group(1))}</a>"
            if m else ""
        )
        banner = (
            "<div class=banner style='background:#fee;border:1px solid #f99;"
            "padding:8px;border-radius:6px;margin-bottom:10px'>"
            f"Superseded{link}. Hidden from recall in the current view; the file is "
            "kept as the audit trail.</div>"
        )
    body = (
        f"<p><a href='/'>&larr; notes</a></p>"
        f"{banner}"
        f"<h2>{html.escape(note.filename)}</h2>"
        f"<p class=muted>{note.tokens} tokens · {len(note.links)} outgoing · {len(note.backlinks)} backlinks</p>"
        f"<div class=note-body>{_linkify(note.text)}</div>"
        f"<h3>Backlinks</h3><ul>{backlinks}</ul>"
    )
    return 200, _page(f"WikiMoth note: {note.filename}", body)


def render_graph(model: VaultModel, *, max_nodes: int = 200) -> str:
    """Deterministic circular SVG of the authored wikilink graph."""
    # Show the most-connected notes first so a big vault stays legible.
    ranked = sorted(
        model.notes.values(),
        key=lambda n: (-(len(n.links) + len(n.backlinks)), n.slug),
    )
    shown = ranked[:max_nodes]
    index = {n.slug: i for i, n in enumerate(shown)}
    k = len(shown)
    if k == 0:
        return _page("graph", "<p class=muted>No notes to graph.</p>")

    size, pad = 760, 60
    r = (size / 2) - pad
    cx = cy = size / 2
    pos: dict[str, tuple[float, float]] = {}
    for i, n in enumerate(shown):
        ang = (2 * math.pi * i / k) - math.pi / 2
        pos[n.slug] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

    edges_svg = []
    drawn: set[tuple[str, str]] = set()
    for n in shown:
        for t in n.links:
            if t in index:
                key = tuple(sorted((n.slug, t)))
                if key in drawn:
                    continue
                drawn.add(key)
                x1, y1 = pos[n.slug]
                x2, y2 = pos[t]
                edges_svg.append(
                    f"<line x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f} "
                    "stroke='#58a6ff' stroke-opacity='0.22' stroke-width='1'/>"
                )

    nodes_svg = []
    for n in shown:
        x, y = pos[n.slug]
        deg = len(n.links) + len(n.backlinks)
        rad = 3 + min(7, deg)
        right = x >= cx
        anchor = "start" if right else "end"
        lx = x + (rad + 4) * (1 if right else -1)
        label = html.escape(n.slug if len(n.slug) <= 22 else n.slug[:21] + "…")
        nodes_svg.append(
            f"<a href='/note?slug={_q(n.slug)}'>"
            f"<circle cx={x:.1f} cy={y:.1f} r={rad} fill='#3fb950'/>"
            f"<text x={lx:.1f} y={y + 3:.1f} text-anchor='{anchor}' "
            f"fill='#8b949e' font-size='10'>{label}</text></a>"
        )

    omitted = ""
    if len(ranked) > k:
        omitted = f"<p class=muted>Showing the {k} most-connected of {len(ranked)} notes.</p>"

    svg = (
        f"<svg viewBox='0 0 {size} {size}' width='{size}' role='img' "
        "aria-label='wikilink graph'>"
        + "".join(edges_svg) + "".join(nodes_svg) + "</svg>"
    )
    body = (
        f"<p><a href='/'>&larr; notes</a></p><h2>Wikilink graph</h2>"
        f"<p class=muted>{model.n_notes} notes · {model.n_edges} authored edges "
        "(the same edges the retriever walks). Click a node to open the note.</p>"
        + omitted + svg
    )
    return _page("WikiMoth — graph", body)


def _note_slug_of(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    return meta.get("note_slug") or _slugify(getattr(chunk, "doc_id", "") or "")


def render_ask(rag: Any, model: VaultModel, q: str, *, top_k: int = DEFAULT_TOP_K) -> str:
    q = (q or "").strip()
    top_k = max(1, int(top_k))  # a non-positive top_k would slice from the end
    if not q:
        return _page("WikiMoth — ask", _ask_form() + "<p class=muted>Type a question above.</p>")

    pairs = rag.retrieve_with_hops(q, top_k=top_k)
    # Count each chunk's tokens once and reuse the per-chunk figure below.
    per_chunk = [count_passage_tokens([getattr(c, "text", "") or ""]) for c, _ in pairs]
    fed = sum(per_chunk)
    total = model.total_tokens
    pct = (100.0 * (1 - fed / total)) if total else 0.0

    if not pairs:
        chunks_html = (
            "<p class=muted>No notes matched — try different words, or this vault "
            "has no link-path to an answer.</p>"
        )
    else:
        rows = []
        for (c, hop), tok in zip(pairs, per_chunk):
            slug = _note_slug_of(c)
            fname = model.notes[slug].filename if slug in model.notes else slug
            cid = html.escape(getattr(c, "chunk_id", "") or slug)
            # hop_label is built from constants + an int, so it needs no escape.
            hop_label = "seed (hop 0)" if hop == 0 else (f"hop {hop}" if hop and hop > 0 else "—")
            rows.append(
                "<div class=chunk><div class=meta>"
                f"<a href='/note?slug={_q(slug)}'><b>{html.escape(fname)}</b></a>"
                f"<span class='badge hop'>{hop_label}</span>"
                f"<span class=badge>{tok} tok</span>"
                f"<span class=badge>{cid}</span>"
                "</div>"
                f"<pre>{html.escape(getattr(c, 'text', '') or '')}</pre></div>"
            )
        chunks_html = "".join(rows)

    body = (
        _ask_form(q)
        + f"<p class=savings>This question feeds <b>{fed:,} tokens</b> to the reader "
        f"vs <b>{total:,}</b> for dumping the whole vault — "
        f"<b>&minus;{pct:.1f}%</b>, deterministically, no LLM call to pick them.</p>"
        + f"<h3>The note-chain WikiMoth would feed for: <em>{html.escape(q)}</em></h3>"
        + chunks_html
    )
    return _page(f"WikiMoth — ask: {q}", body)


# ----------------------------------------------------------------------
# Routing (pure) + HTTP server
# ----------------------------------------------------------------------


def route(path: str, query: dict[str, list[str]], model: VaultModel, rag: Any,
          *, top_k: int = DEFAULT_TOP_K) -> tuple[int, str, str]:
    """Map a request to ``(status, content_type, body)``. Pure / testable."""
    if path in ("/", "/index.html"):
        return 200, HTML_CTYPE, render_home(model)
    if path == "/note":
        slug = (query.get("slug") or [""])[0]
        status, html_body = render_note(model, slug)
        return status, HTML_CTYPE, html_body
    if path == "/graph":
        return 200, HTML_CTYPE, render_graph(model)
    if path == "/ask":
        q = (query.get("q") or [""])[0]
        return 200, HTML_CTYPE, render_ask(rag, model, q, top_k=top_k)
    if path == "/favicon.ico":
        return 204, "image/x-icon", ""
    return 404, HTML_CTYPE, _page("not found", "<p>Not found. <a href='/'>Home</a>.</p>")


def _make_handler(model: VaultModel, rag: Any, top_k: int):
    class _Handler(BaseHTTPRequestHandler):
        # Quiet by default; one startup line is printed by serve().
        def log_message(self, *args: Any) -> None:  # noqa: D401
            return

        def do_GET(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            query = parse_qs(parts.query)
            try:
                status, ctype, body = route(parts.path, query, model, rag, top_k=top_k)
            except Exception as exc:  # never leak a stack trace to the browser
                status, ctype, body = 500, HTML_CTYPE, _page(
                    "error", f"<p>Internal error: {html.escape(type(exc).__name__)}</p>"
                )
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

    return _Handler


def serve(
    vault_dir: str | Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    top_k: int = DEFAULT_TOP_K,
) -> int:
    """Build the model + retriever and serve the viewer until Ctrl-C."""
    from wikimoth.pipeline import MemoryRAG

    top_k = max(1, int(top_k))
    model = VaultModel.from_dir(vault_dir)
    rag = MemoryRAG().index(vault_dir)

    handler = _make_handler(model, rag, top_k)
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        print(f"could not bind {host}:{port}: {exc}")
        print("  the port may be in use — try a different --port.")
        return 1
    url = f"http://{host}:{port}/"
    print(f"WikiMoth viewer → {url}")
    print(f"  vault: {Path(vault_dir).resolve()}  ({model.n_notes} notes, {model.n_edges} edges)")
    print("  Ctrl-C to stop. Nothing leaves this machine.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return 0


__all__ = ["VaultModel", "Note", "route", "serve", "render_home", "render_note",
           "render_graph", "render_ask", "DEFAULT_HOST", "DEFAULT_PORT", "DEFAULT_TOP_K"]
