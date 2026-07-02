# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for as-of / supersession-aware retrieval (the read side of supersede).

Drives the REAL retrieve path (retrieve_with_hops, the one all consumers use), so
a filter mistakenly wired only into the unused _retrieve_filtered would fail here.
Fully offline.
"""

from __future__ import annotations

from datetime import date

from wikimoth.mcp import _build_index
from wikimoth.supersede import supersede


def _w(v, name, content):
    (v / name).write_text(content, encoding="utf-8")


def _texts(pairs):
    return " ".join(getattr(c, "text", "") or "" for c, _ in pairs)


def test_superseded_body_dropped_by_default(tmp_path):
    # The filter must FIRE on retrieve_with_hops (not be a silent no-op).
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", '---\nname: old\nsuperseded_by: "[[new]]"\nstatus: superseded\nvalid_to: 2026-06-20\n---\nThe zzqq spec is OLDVALUE.\n')
    _w(v, "new.md", "---\nname: new\n---\nThe zzqq spec is NEWVALUE.\n")
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("zzqq", top_k=8))
    assert "OLDVALUE" not in t  # superseded body hidden
    assert "NEWVALUE" in t      # current note visible


def test_hop_to_current_surfaces_new(tmp_path):
    # A query matching ONLY the superseded note still surfaces the current one via
    # the live superseded_by edge (the body is dropped, the edge is not).
    v = tmp_path / "v"
    v.mkdir()
    supersede_src = '---\nname: old\n---\nUnique token zzqq lives here. Old details.\n'
    _w(v, "old.md", supersede_src)
    _w(v, "new.md", "---\nname: new\n---\nThe current answer is NEWVAL.\n")
    supersede(v, "old", "new", as_of=date(2026, 6, 20))
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("zzqq", top_k=8))
    assert "NEWVAL" in t          # reached via the OLD->NEW edge
    assert "Old details" not in t  # old body dropped


def test_superseded_multichunk_no_chunk0_leak(tmp_path):
    # A >400-token superseded note: EVERY chunk must drop, not just chunk 0.
    v = tmp_path / "v"
    v.mkdir()
    big = "widgetword " * 700  # well over one ~400-token chunk
    _w(v, "old.md", f'---\nname: old\nsuperseded_by: "[[new]]"\nstatus: superseded\nvalid_to: 2026-06-20\n---\n{big}\nTAILMARKER end.\n')
    _w(v, "new.md", "---\nname: new\n---\ncurrent widgetword note.\n")
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("widgetword", top_k=30))
    assert "TAILMARKER" not in t  # the LAST chunk of the superseded note is also hidden


def test_garbage_valid_to_fails_open(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\nvalid_to: not-a-date\n---\nKEEPME content here.\n")
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("KEEPME", top_k=8))
    assert "KEEPME" in t  # unparseable date never hides a note


def test_expired_valid_to_dropped_by_default(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\nvalid_to: 2000-01-01\n---\nEXPIREDFACT content.\n")
    _w(v, "b.md", "---\nname: b\n---\nLIVEFACT content.\n")
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("content", top_k=8))
    assert "EXPIREDFACT" not in t
    assert "LIVEFACT" in t


def test_as_of_time_travel(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", '---\nname: old\nsuperseded_by: "[[new]]"\nstatus: superseded\nvalid_from: 2020-01-01\nvalid_to: 2022-01-01\n---\nOLDFACT about thing.\n')
    _w(v, "new.md", "---\nname: new\nvalid_from: 2022-01-01\n---\nNEWFACT about thing.\n")
    rag, _, _, _ = _build_index(v)
    # current view: old hidden, new shown
    t_now = _texts(rag.retrieve_with_hops("thing", top_k=8))
    assert "OLDFACT" not in t_now and "NEWFACT" in t_now
    # as-of 2021: old was valid, new not yet
    rag.as_of = date(2021, 1, 1)
    t_then = _texts(rag.retrieve_with_hops("thing", top_k=8))
    assert "OLDFACT" in t_then and "NEWFACT" not in t_then


def test_show_superseded_includes_body(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "old.md", '---\nname: old\nsuperseded_by: "[[new]]"\nstatus: superseded\nvalid_to: 2026-06-20\n---\nOLDFACT here.\n')
    _w(v, "new.md", "---\nname: new\n---\nNEWFACT here.\n")
    rag, _, _, _ = _build_index(v)
    rag.show_superseded = True
    t = _texts(rag.retrieve_with_hops("OLDFACT", top_k=8))
    assert "OLDFACT" in t  # forensic view shows everything


def test_top_k_preserved_despite_drops(tmp_path):
    # Over-fetch: superseded notes that rank high must not starve top_k.
    v = tmp_path / "v"
    v.mkdir()
    for i in range(3):
        _w(v, f"vis{i}.md", f"---\nname: vis{i}\n---\nmatchword visible VIS{i}.\n")
    for i in range(3):
        _w(v, f"sup{i}.md", f"---\nname: sup{i}\nstatus: superseded\n---\nmatchword stale SUP{i}.\n")
    rag, _, _, _ = _build_index(v)
    pairs = rag.retrieve_with_hops("matchword", top_k=3)
    t = _texts(pairs)
    assert len(pairs) == 3            # not starved below top_k
    assert "SUP" not in t            # no superseded body leaked
    assert "VIS" in t                # visible notes fill the result


def test_future_valid_from_hidden_in_current_view(tmp_path):
    # A fact dated to become valid in the future is not yet current.
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\nvalid_from: 2099-01-01\n---\nFUTUREFACT content.\n")
    _w(v, "b.md", "---\nname: b\n---\nNOWFACT content.\n")
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("content", top_k=8))
    assert "FUTUREFACT" not in t
    assert "NOWFACT" in t


def test_plain_vault_unaffected(tmp_path):
    # No hygiene metadata anywhere -> behaviour identical to before (all visible).
    v = tmp_path / "v"
    v.mkdir()
    _w(v, "a.md", "---\nname: a\n---\nplain ALPHA note.\n")
    _w(v, "b.md", "---\nname: b\n---\nplain BETA note.\n")
    rag, _, _, _ = _build_index(v)
    t = _texts(rag.retrieve_with_hops("plain", top_k=8))
    assert "ALPHA" in t and "BETA" in t
