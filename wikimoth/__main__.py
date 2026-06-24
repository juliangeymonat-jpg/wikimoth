# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Run the WikiMoth CLI as ``python -m wikimoth ...``.

This makes every subcommand (notably the MCP server) runnable without relying on
the ``wikimoth`` console script being on PATH: ``python -m wikimoth mcp`` works as
long as the package is importable, which is the robust form to register with an
MCP client (``claude mcp add wikimoth -- python -m wikimoth mcp``).
"""

from wikimoth.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
