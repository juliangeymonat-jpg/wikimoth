# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Standalone MCP acceptance check against the INSTALLED wikimoth wheel.

Spawns ``python -m wikimoth mcp`` over stdio, performs the initialize +
tools/list JSON-RPC handshake, and asserts the server reports the right
serverInfo and a non-empty tool list. The release acceptance gate runs this
against the freshly-built wheel in a clean venv (not the source tree). Exits
nonzero on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile


def main() -> int:
    vault = tempfile.mkdtemp(prefix="wikimoth-acc-")
    proc = subprocess.Popen(
        [sys.executable, "-m", "wikimoth", "mcp", "--vault", vault],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )

    def send(obj: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv() -> dict | None:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        return json.loads(line) if line.strip() else None

    init: dict | None = None
    tools: dict | None = None
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "acceptance", "version": "0"}}})
        init = recv()
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = recv()
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()

    problems: list[str] = []
    server_info = (init or {}).get("result", {}).get("serverInfo", {})
    if server_info.get("name") != "wikimoth":
        problems.append(f"initialize serverInfo.name != 'wikimoth': {server_info!r}")
    names = [t.get("name") for t in (tools or {}).get("result", {}).get("tools", [])]
    if not names:
        problems.append("tools/list returned no tools")

    if problems:
        print("MCP ACCEPTANCE FAILED:")
        for p in problems:
            print("  -", p)
        if proc.stderr is not None:
            print("server stderr:\n" + proc.stderr.read())
        return 1
    print(f"OK: MCP server responds. serverInfo={server_info}; "
          f"{len(names)} tools: {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
