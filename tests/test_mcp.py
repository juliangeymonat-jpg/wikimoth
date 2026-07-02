# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the MCP server (stdio JSON-RPC) that puts recall in the agent loop.

Everything is driven in-process: the protocol core ``_Server.handle`` takes one
JSON-RPC dict and returns one, and ``serve_stdio`` is pumped with ``StringIO``,
so there is no subprocess, socket, or network. Matches the repo invariant that
the suite is fully offline.
"""

from __future__ import annotations

import io
import json

from wikimoth.mcp import TOOLS, serve_stdio
from wikimoth.mcp import _Server  # type: ignore[attr-defined]


def _vault(tmp_path):
    """A 3-note [[wikilink]] chain (answer is link-only) plus a distractor."""
    v = tmp_path / "vault"
    v.mkdir()
    (v / "alpha.md").write_text(
        "---\nname: alpha\n---\nNotes on topicZ9. See the next file. [[beta]]\n",
        encoding="utf-8",
    )
    (v / "beta.md").write_text(
        "---\nname: beta\n---\nIntermediate waypoint. Continue. [[gamma]]\n",
        encoding="utf-8",
    )
    (v / "gamma.md").write_text(
        "---\nname: gamma\n---\nThe final answer is FOUNDIT42.\n",
        encoding="utf-8",
    )
    (v / "noise.md").write_text(
        "---\nname: noise\n---\nUnrelated routine filing record, no reference.\n",
        encoding="utf-8",
    )
    return v


def _call(server, **params):
    return server.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": params})


# ---------------------------------------------------------------------------
# Protocol handshake
# ---------------------------------------------------------------------------
def test_initialize(tmp_path):
    s = _Server(_vault(tmp_path))
    resp = s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2024-11-05"}})
    assert resp["id"] == 1
    r = resp["result"]
    assert r["protocolVersion"] == "2024-11-05"
    assert r["serverInfo"]["name"] == "wikimoth"
    assert "tools" in r["capabilities"]


def test_initialize_defaults_protocol(tmp_path):
    s = _Server(_vault(tmp_path))
    resp = s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["protocolVersion"]  # a non-empty default


def test_tools_list(tmp_path):
    s = _Server(_vault(tmp_path))
    resp = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert names == {
        "recall", "status", "list_conflicts", "list_lint", "list_duplicates", "list_fading", "supersede",
    }
    recall = next(t for t in TOOLS if t["name"] == "recall")
    assert recall["inputSchema"]["required"] == ["query"]


def test_notification_gets_no_response(tmp_path):
    s = _Server(_vault(tmp_path))
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_errors(tmp_path):
    s = _Server(_vault(tmp_path))
    resp = s.handle({"jsonrpc": "2.0", "id": 5, "method": "no/such"})
    assert resp["error"]["code"] == -32601


def test_unknown_notification_silent(tmp_path):
    s = _Server(_vault(tmp_path))
    # a notification (no id) for an unknown method must not produce a response
    assert s.handle({"jsonrpc": "2.0", "method": "no/such"}) is None


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------
def test_recall_reaches_multihop_answer(tmp_path):
    s = _Server(_vault(tmp_path))
    resp = _call(s, name="recall", arguments={"query": "topicZ9"})
    res = resp["result"]
    assert res["isError"] is False
    text = res["content"][0]["text"]
    # walked alpha -> beta -> gamma and pulled the link-only answer
    assert "FOUNDIT42" in text
    assert "Deterministic, no LLM call to retrieve" in text
    assert "vs dumping the whole vault" in text


def test_recall_is_deterministic(tmp_path):
    s = _Server(_vault(tmp_path))
    a = _call(s, name="recall", arguments={"query": "topicZ9"})["result"]["content"][0]["text"]
    b = _call(s, name="recall", arguments={"query": "topicZ9"})["result"]["content"][0]["text"]
    assert a == b


def test_recall_empty_query_is_error(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="recall", arguments={"query": "   "})["result"]
    assert res["isError"] is True
    assert "query" in res["content"][0]["text"]


def test_recall_missing_vault_is_error(tmp_path):
    s = _Server(tmp_path / "does-not-exist")
    res = _call(s, name="recall", arguments={"query": "anything"})["result"]
    assert res["isError"] is True
    assert "No WikiMoth vault" in res["content"][0]["text"]


def test_recall_bad_as_of_is_error(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="recall", arguments={"query": "topicZ9", "as_of": "nope"})["result"]
    assert res["isError"] is True
    assert "YYYY-MM-DD" in res["content"][0]["text"]


def test_recall_default_view_hides_superseded(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "old.md").write_text(
        '---\nname: old\nsuperseded_by: "[[new]]"\nstatus: superseded\nvalid_to: 2026-06-20\n---\ntopicZ9 OLDBODY.\n',
        encoding="utf-8",
    )
    (v / "new.md").write_text("---\nname: new\n---\ntopicZ9 NEWBODY.\n", encoding="utf-8")
    s = _Server(v)
    txt = _call(s, name="recall", arguments={"query": "topicZ9"})["result"]["content"][0]["text"]
    assert "OLDBODY" not in txt and "NEWBODY" in txt


def test_recall_respects_top_k(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="recall", arguments={"query": "topicZ9", "top_k": 1})["result"]
    # top_k=1 returns a single chunk (the seed), so the link-only answer is absent
    assert res["content"][0]["text"].count("[1]") == 1
    assert "[2]" not in res["content"][0]["text"]


def test_unknown_tool_is_error(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="frobnicate", arguments={})["result"]
    assert res["isError"] is True


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def test_status_reports_vault(tmp_path):
    v = _vault(tmp_path)
    s = _Server(v)
    text = _call(s, name="status", arguments={})["result"]["content"][0]["text"]
    assert str(v) in text
    assert "notes: 4" in text  # alpha, beta, gamma, noise
    assert "token backend:" in text


# ---------------------------------------------------------------------------
# list_conflicts
# ---------------------------------------------------------------------------
def test_list_conflicts_tool_surfaces_candidate(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("---\nname: a\nabout: acme\nceo: alice\n---\n", encoding="utf-8")
    (v / "b.md").write_text("---\nname: b\nabout: acme\nceo: bob\n---\n", encoding="utf-8")
    s = _Server(v)
    res = _call(s, name="list_conflicts", arguments={})["result"]
    assert res["isError"] is False
    text = res["content"][0]["text"]
    assert "acme" in text and "ceo" in text
    assert "alice" in text and "bob" in text


def test_list_conflicts_clean_vault(tmp_path):
    # The shared _vault has no conflicting frontmatter -> a clean, consistent report.
    s = _Server(_vault(tmp_path))
    res = _call(s, name="list_conflicts", arguments={})["result"]
    assert res["isError"] is False
    assert "consistent" in res["content"][0]["text"].lower()


def test_list_conflicts_include_inline_arg(tmp_path):
    # The include_inline arg must thread through to scan_conflicts.
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("---\nabout: org\n---\nceo:: alice\n", encoding="utf-8")
    (v / "b.md").write_text("---\nabout: org\n---\nceo:: bob\n", encoding="utf-8")
    s = _Server(v)
    off = _call(s, name="list_conflicts", arguments={})["result"]
    assert "consistent" in off["content"][0]["text"].lower()  # inline off by default
    on = _call(s, name="list_conflicts", arguments={"include_inline": True})["result"]
    assert "ceo" in on["content"][0]["text"] and "alice" in on["content"][0]["text"]


def test_list_conflicts_missing_vault_is_error(tmp_path):
    s = _Server(tmp_path / "does-not-exist")
    res = _call(s, name="list_conflicts", arguments={})["result"]
    assert res["isError"] is True


# ---------------------------------------------------------------------------
# list_lint
# ---------------------------------------------------------------------------
def test_list_lint_surfaces_broken_link(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("---\nname: a\n---\nSee [[ghost]].\n", encoding="utf-8")
    s = _Server(v)
    res = _call(s, name="list_lint", arguments={})["result"]
    assert res["isError"] is False
    assert "ghost" in res["content"][0]["text"]


def test_list_lint_clean_vault(tmp_path):
    s = _Server(_vault(tmp_path))  # alpha->beta->gamma chain, gamma is a leaf
    res = _call(s, name="list_lint", arguments={})["result"]
    assert res["isError"] is False


def test_list_lint_missing_vault_is_error(tmp_path):
    s = _Server(tmp_path / "does-not-exist")
    res = _call(s, name="list_lint", arguments={})["result"]
    assert res["isError"] is True


# ---------------------------------------------------------------------------
# list_duplicates
# ---------------------------------------------------------------------------
def test_list_duplicates_tool(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    dup = "The shared body text that appears verbatim in two different note files here."
    (v / "a.md").write_text(f"---\nname: a\n---\n{dup}\n", encoding="utf-8")
    (v / "b.md").write_text(f"---\nname: b\n---\n{dup}\n", encoding="utf-8")
    s = _Server(v)
    res = _call(s, name="list_duplicates", arguments={})["result"]
    assert res["isError"] is False
    assert "a.md" in res["content"][0]["text"] and "b.md" in res["content"][0]["text"]


def test_list_duplicates_clean_vault(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="list_duplicates", arguments={})["result"]
    assert res["isError"] is False


# ---------------------------------------------------------------------------
# list_fading
# ---------------------------------------------------------------------------
def test_list_fading_tool(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "old.md").write_text("---\nname: old\nlast_access: 2000-01-01\n---\nancient note.\n", encoding="utf-8")
    s = _Server(v)
    res = _call(s, name="list_fading", arguments={})["result"]
    assert res["isError"] is False
    assert "old.md" in res["content"][0]["text"]


def test_list_fading_missing_vault_is_error(tmp_path):
    s = _Server(tmp_path / "does-not-exist")
    res = _call(s, name="list_fading", arguments={})["result"]
    assert res["isError"] is True


# ---------------------------------------------------------------------------
# supersede (write tool)
# ---------------------------------------------------------------------------
def test_supersede_tool_marks_old(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "old.md").write_text("---\nname: old\n---\nbody\n", encoding="utf-8")
    (v / "new.md").write_text("---\nname: new\n---\nbody\n", encoding="utf-8")
    s = _Server(v)
    res = _call(s, name="supersede", arguments={"old": "old", "new": "new"})["result"]
    assert res["isError"] is False
    from wikimoth.frontmatter import parse_frontmatter

    fm = parse_frontmatter((v / "old.md").read_text(encoding="utf-8"))
    assert fm["superseded_by"] == "[[new]]"


def test_supersede_tool_missing_args_is_error(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    s = _Server(v)
    res = _call(s, name="supersede", arguments={"old": "x"})["result"]
    assert res["isError"] is True


def test_supersede_tool_unresolved_is_error(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "new.md").write_text("---\nname: new\n---\nbody\n", encoding="utf-8")
    s = _Server(v)
    res = _call(s, name="supersede", arguments={"old": "ghost", "new": "new"})["result"]
    assert res["isError"] is True


# ---------------------------------------------------------------------------
# Full stdio pump
# ---------------------------------------------------------------------------
def test_serve_stdio_end_to_end(tmp_path):
    v = _vault(tmp_path)
    lines_in = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "recall", "arguments": {"query": "topicZ9"}}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in lines_in) + "\n")
    stdout = io.StringIO()
    rc = serve_stdio(v, stdin=stdin, stdout=stdout)
    assert rc == 0
    out = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
    # the notification gets no reply -> exactly two responses
    assert [r["id"] for r in out] == [1, 2]
    assert out[0]["result"]["serverInfo"]["name"] == "wikimoth"
    assert "FOUNDIT42" in out[1]["result"]["content"][0]["text"]


def test_serve_stdio_bad_json_reports_parse_error(tmp_path):
    stdin = io.StringIO("{not json}\n")
    stdout = io.StringIO()
    serve_stdio(tmp_path, stdin=stdin, stdout=stdout)
    out = json.loads(stdout.getvalue().splitlines()[0])
    assert out["error"]["code"] == -32700


# ---------------------------------------------------------------------------
# Malformed-but-valid-JSON hardening (regression: one bad line must not kill the loop)
# ---------------------------------------------------------------------------
def test_handle_non_dict_message(tmp_path):
    s = _Server(_vault(tmp_path))
    for garbage in ([1, 2, 3], 42, "hello", True, None):
        resp = s.handle(garbage)
        assert resp["error"]["code"] == -32600  # invalid request, not a crash


def test_serve_stdio_non_dict_line_survives(tmp_path):
    # A bare JSON array (also a batch, which we don't support) must not drop later messages.
    stdin = io.StringIO(
        '[1,2,3]\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n'
    )
    stdout = io.StringIO()
    rc = serve_stdio(_vault(tmp_path), stdin=stdin, stdout=stdout)
    assert rc == 0
    out = [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]
    assert out[0]["error"]["code"] == -32600       # the bad line got an error
    assert out[1] == {"jsonrpc": "2.0", "id": 7, "result": {}}  # ping still served


def test_recall_non_string_query_is_error(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="recall", arguments={"query": 123})["result"]
    assert res["isError"] is True


def test_recall_bool_top_k_falls_back(tmp_path):
    # bool is a subclass of int; True must NOT be treated as top_k=1.
    s = _Server(_vault(tmp_path))
    text = _call(s, name="recall", arguments={"query": "topicZ9", "top_k": True})["result"]["content"][0]["text"]
    assert "FOUNDIT42" in text  # full chain reached, not truncated to the seed


def test_tools_call_non_dict_arguments_no_crash(tmp_path):
    s = _Server(_vault(tmp_path))
    res = _call(s, name="recall", arguments=[1, 2, 3])["result"]
    assert res["isError"] is True  # empty query, gracefully, not a -32603 crash


def test_initialize_non_dict_params_ok(tmp_path):
    s = _Server(_vault(tmp_path))
    resp = s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": [1, 2, 3]})
    assert resp["result"]["serverInfo"]["name"] == "wikimoth"


def test_python_dash_m_wikimoth_runs():
    """`python -m wikimoth ...` must work (the documented MCP command does not rely
    on the `wikimoth` console script being on PATH). Regression guard."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, "-m", "wikimoth", "--help"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert r.returncode == 0, r.stderr
    assert "mcp" in r.stdout  # the mcp subcommand is reachable via `python -m wikimoth`


def test_serve_stdio_broken_stdout_breaks_clean(tmp_path):
    class _BrokenStdout:
        def write(self, s):
            raise BrokenPipeError("client gone")

        def flush(self):
            pass

    stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    # must return cleanly, not propagate BrokenPipeError
    assert serve_stdio(tmp_path, stdin=stdin, stdout=_BrokenStdout()) == 0
