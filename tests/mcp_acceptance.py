# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Standalone MCP acceptance check against the INSTALLED wikimoth wheel.

Spawns ``python -m wikimoth mcp`` over stdio, performs the initialize +
tools/list JSON-RPC handshake, and asserts the server reports the right
serverInfo and a non-empty tool list. The release acceptance gate runs this
against the freshly-built wheel in a clean venv. Exits nonzero on any failure.

Hardened for gate duty: the child is spawned with a NEUTRAL cwd (so the
source tree can never shadow the installed wheel via ``-m``'s sys.path[0]),
every read is bounded by a watchdog (a hung server fails in seconds, not at
the CI job timeout), stderr goes to a file (an unread PIPE can deadlock a
chatty child), and the temp vault is removed afterwards.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading

WATCHDOG_SECONDS = 60


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="wikimoth-acc-")
    stderr_path = f"{workdir}/server-stderr.log"
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "wikimoth", "mcp", "--vault", workdir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
        cwd=workdir,  # neutral: -m must resolve wikimoth from site-packages
    )
    watchdog = threading.Timer(WATCHDOG_SECONDS, proc.kill)
    watchdog.start()

    def send(obj: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv() -> dict | None:
        assert proc.stdout is not None
        line = proc.stdout.readline()  # bounded by the watchdog
        return json.loads(line) if line.strip() else None

    init: dict | None = None
    tools: dict | None = None
    timed_out = False
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "acceptance", "version": "0"}}})
        init = recv()
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = recv()
    except Exception:  # noqa: BLE001 — a killed pipe surfaces here; report below
        timed_out = not watchdog.is_alive()
    finally:
        watchdog.cancel()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
        stderr_file.close()

    problems: list[str] = []
    if timed_out or (init is None and tools is None):
        problems.append(f"no handshake within {WATCHDOG_SECONDS}s (server hung or died)")
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
        try:
            with open(stderr_path, encoding="utf-8") as fh:
                print("server stderr:\n" + fh.read())
        except OSError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    print(f"OK: MCP server responds. serverInfo={server_info}; "
          f"{len(names)} tools: {names}")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
