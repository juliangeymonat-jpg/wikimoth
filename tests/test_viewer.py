# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the zero-dependency ``wikimoth serve`` web viewer.

The render functions are pure (model/rag in, HTML out) so they are tested here
without ever opening a socket.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wikimoth.pipeline import MemoryRAG
from wikimoth.viewer import (
    VaultModel,
    _linkify,
    render_ask,
    render_note,
    route,
)


def _write_vault(root: Path) -> Path:
    """A tiny 3-note vault: alpha -> beta -> gamma (multi-hop chain)."""
    (root / "alpha.md").write_text(
        "# Alpha\nThe quokka project lives here.\nSee [[beta]] for the build.\n",
        encoding="utf-8",
    )
    (root / "beta.md").write_text(
        "# Beta\nBuild details.\nThe answer continues in [[gamma]].\n",
        encoding="utf-8",
    )
    (root / "gamma.md").write_text(
        "# Gamma\nThe deployment runs on a raspberry pi cluster.\n",
        encoding="utf-8",
    )
    return root


# ----------------------------------------------------------------------
# VaultModel
# ----------------------------------------------------------------------


def test_vault_model_scans_notes_links_backlinks(tmp_path):
    model = VaultModel.from_dir(_write_vault(tmp_path))
    assert model.n_notes == 3
    assert set(model.notes) == {"alpha", "beta", "gamma"}
    # resolved outgoing links
    assert model.notes["alpha"].links == ["beta"]
    assert model.notes["beta"].links == ["gamma"]
    assert model.notes["gamma"].links == []
    # backlinks are the inverse
    assert model.notes["beta"].backlinks == ["alpha"]
    assert model.notes["gamma"].backlinks == ["beta"]
    # 2 authored edges, positive token total
    assert model.n_edges == 2
    assert model.total_tokens > 0


def test_vault_model_ignores_dangling_links(tmp_path):
    (tmp_path / "solo.md").write_text("Links to [[does-not-exist]].\n", encoding="utf-8")
    model = VaultModel.from_dir(tmp_path)
    assert model.notes["solo"].links == []  # dangling target dropped
    assert model.n_edges == 0


def test_vault_model_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VaultModel.from_dir(tmp_path / "nope")


def test_vault_model_rejects_a_file(tmp_path):
    # Regression: a file path must be rejected, not silently scanned as an
    # empty vault (was .exists(), now .is_dir()).
    f = tmp_path / "not-a-vault.md"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        VaultModel.from_dir(f)


# ----------------------------------------------------------------------
# linkify / escaping
# ----------------------------------------------------------------------


def test_linkify_escapes_html_and_renders_wikilinks():
    out = _linkify("danger <script>alert(1)</script> and [[Some Note]] here")
    assert "<script>" not in out  # raw tag escaped
    assert "&lt;script&gt;" in out
    assert "/note?slug=some%20note" in out  # anchor slug URL-encoded
    assert ">Some Note<" in out  # original label preserved


def test_linkify_ampersand_slug_and_no_double_escape():
    # Regression: the wikilink target is slugified from RAW text (not the HTML
    # entity) and URL-encoded; the label is escaped exactly once.
    out = _linkify("see [[Rock & Roll]] now")
    assert "/note?slug=rock%20%26%20roll" in out  # & encoded as %26, space as %20
    assert ">Rock &amp; Roll<" in out  # label escaped once
    assert "&amp;amp;" not in out  # NOT double-escaped


# ----------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------


@pytest.fixture()
def served(tmp_path):
    vault = _write_vault(tmp_path)
    model = VaultModel.from_dir(vault)
    rag = MemoryRAG().index(vault)
    return model, rag


def test_route_home_ok(served):
    model, rag = served
    status, ctype, body = route("/", {}, model, rag)
    assert status == 200
    assert "text/html" in ctype
    assert "WikiMoth" in body
    assert "alpha.md" in body


def test_route_note_found_and_404(served):
    model, rag = served
    status, _, body = route("/note", {"slug": ["beta"]}, model, rag)
    assert status == 200
    assert "beta.md" in body
    assert "alpha.md" in body  # backlink shown

    status, _, body = route("/note", {"slug": ["ghost"]}, model, rag)
    assert status == 404


def test_route_note_with_ampersand_slug_resolves(tmp_path):
    # Regression: a note whose slug contains '&' must resolve. The browser
    # sends the decoded slug, which parse_qs hands us as "rock & roll".
    (tmp_path / "Rock & Roll.md").write_text("[[Jazz & Blues]]\n", encoding="utf-8")
    (tmp_path / "Jazz & Blues.md").write_text("music notes\n", encoding="utf-8")
    model = VaultModel.from_dir(tmp_path)
    rag = MemoryRAG().index(tmp_path)
    status, _, body = route("/note", {"slug": ["rock & roll"]}, model, rag)
    assert status == 200
    assert "Rock &amp; Roll.md" in body
    # its wikilink to the other &-note is a correctly-encoded anchor
    assert "/note?slug=jazz%20%26%20blues" in body


def test_route_note_escapes_content(tmp_path):
    (tmp_path / "evil.md").write_text("<script>steal()</script>\n", encoding="utf-8")
    model = VaultModel.from_dir(tmp_path)
    rag = MemoryRAG().index(tmp_path)
    status, _, body = route("/note", {"slug": ["evil"]}, model, rag)
    assert status == 200
    assert "<script>steal()</script>" not in body
    assert "&lt;script&gt;" in body


def test_route_graph_ok(served):
    model, rag = served
    status, _, body = route("/graph", {}, model, rag)
    assert status == 200
    assert "<svg" in body


def test_route_favicon_and_unknown(served):
    model, rag = served
    status, _, _ = route("/favicon.ico", {}, model, rag)
    assert status == 204
    status, _, _ = route("/totally-unknown", {}, model, rag)
    assert status == 404


# ----------------------------------------------------------------------
# the money view: /ask
# ----------------------------------------------------------------------


def test_ask_empty_query_shows_form(served):
    model, rag = served
    body = render_ask(rag, model, "")
    assert "Ask your memory" in body or "Type a question" in body


def test_ask_shows_chain_and_token_savings(served):
    model, rag = served
    # "quokka" only appears in alpha; gamma is reachable only via the [[links]].
    body = render_ask(rag, model, "quokka project", top_k=8)
    assert "alpha.md" in body          # the lexical seed
    assert "&minus;" in body           # the -N% token-savings line rendered
    assert "tokens" in body
    # at least one hop badge present (seed at hop 0)
    assert "hop" in body


def test_route_ask_dispatches(served):
    model, rag = served
    status, ctype, body = route("/ask", {"q": ["quokka"]}, model, rag)
    assert status == 200
    assert "text/html" in ctype
    assert "quokka" in body


def test_ask_clamps_nonpositive_top_k(served):
    # Regression: a non-positive top_k must not become a negative slice
    # (scored[:-k] returns "all but the last k"); it is clamped to >= 1.
    model, rag = served
    body = render_ask(rag, model, "quokka", top_k=0)
    assert "alpha.md" in body  # still returns the seed, no crash, no over-return
