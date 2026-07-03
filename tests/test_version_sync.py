# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Version single-source-of-truth guard.

Fails when wikimoth.__version__, the installed distribution metadata, and the
version-bearing manifests disagree -- the drift that left the Claude Code
plugin manifests advertising 0.1.0 while the package shipped 0.2.0. Covered
manifests (when present on the checkout): both plugin JSONs, the npm launcher
package.json, the MCP registry server.json, and CITATION.cff.
Runnable standalone::

    python tests/test_version_sync.py
"""
from __future__ import annotations

import json
import re
from importlib.metadata import version as dist_version
from pathlib import Path

import wikimoth

_ROOT = Path(__file__).resolve().parent.parent


def _json_version(relpath: str) -> str | None:
    p = _ROOT / relpath
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("version")


def _marketplace_version() -> str | None:
    p = _ROOT / ".claude-plugin" / "marketplace.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    for plug in data.get("plugins", []):
        if plug.get("name") == "wikimoth":
            return plug.get("version")
    return None


def _citation_version() -> str | None:
    p = _ROOT / "CITATION.cff"
    if not p.exists():
        return None
    m = re.search(r'^version:\s*["\']?([^"\'\n]+)', p.read_text(encoding="utf-8"), re.M)
    return m.group(1).strip() if m else None


# (label, getter) for every version-bearing file; getters return None when the
# file is not on this checkout (e.g. running from the installed wheel).
_MANIFESTS = (
    ("plugin/.claude-plugin/plugin.json",
     lambda: _json_version("plugin/.claude-plugin/plugin.json")),
    (".claude-plugin/marketplace.json", _marketplace_version),
    ("npm/package.json", lambda: _json_version("npm/package.json")),
    ("server.json", lambda: _json_version("server.json")),
    ("CITATION.cff", _citation_version),
)


def test_dunder_matches_installed_metadata():
    assert wikimoth.__version__ == dist_version("wikimoth"), (
        f"wikimoth.__version__ {wikimoth.__version__!r} != installed metadata "
        f"{dist_version('wikimoth')!r}"
    )


def test_manifests_match_version():
    for label, getter in _MANIFESTS:
        got = getter()
        if got is None:
            continue
        assert got == wikimoth.__version__, (
            f"{label} version {got!r} != wikimoth.__version__ "
            f"{wikimoth.__version__!r}"
        )


if __name__ == "__main__":
    # Standalone runner = the SAME assertions as pytest, not a parallel copy.
    import sys

    problems: list[str] = []
    for check in (test_dunder_matches_installed_metadata, test_manifests_match_version):
        try:
            check()
        except AssertionError as exc:
            problems.append(str(exc).splitlines()[0])
    if problems:
        print("VERSION DRIFT:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: version {wikimoth.__version__} consistent")
    sys.exit(0)
