# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Version single-source-of-truth guard.

Fails when wikimoth.__version__, the installed distribution metadata, and the
Claude Code plugin manifests disagree -- the drift that left the plugin
manifests advertising 0.1.0 while the package shipped 0.2.0. Runnable
standalone::

    python tests/test_version_sync.py
"""
from __future__ import annotations

import json
from importlib.metadata import version as dist_version
from pathlib import Path

import wikimoth

_ROOT = Path(__file__).resolve().parent.parent


def _plugin_json_version() -> str | None:
    p = _ROOT / "plugin" / ".claude-plugin" / "plugin.json"
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


def test_dunder_matches_installed_metadata():
    assert wikimoth.__version__ == dist_version("wikimoth"), (
        f"wikimoth.__version__ {wikimoth.__version__!r} != installed metadata "
        f"{dist_version('wikimoth')!r}"
    )


def test_plugin_manifests_match_version():
    for label, got in (("plugin.json", _plugin_json_version()),
                       ("marketplace.json", _marketplace_version())):
        if got is None:
            continue  # manifest not on this path (e.g. installed wheel) -> skip
        assert got == wikimoth.__version__, (
            f"{label} version {got!r} != wikimoth.__version__ "
            f"{wikimoth.__version__!r}"
        )


if __name__ == "__main__":
    import sys

    problems: list[str] = []
    try:
        meta = dist_version("wikimoth")
    except Exception as exc:  # noqa: BLE001
        meta = f"<error: {exc}>"
    if wikimoth.__version__ != meta:
        problems.append(f"__version__ {wikimoth.__version__!r} != installed metadata {meta!r}")
    for _label, _got in (("plugin.json", _plugin_json_version()),
                         ("marketplace.json", _marketplace_version())):
        if _got is not None and _got != wikimoth.__version__:
            problems.append(f"{_label} {_got!r} != __version__ {wikimoth.__version__!r}")
    if problems:
        print("VERSION DRIFT:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: version {wikimoth.__version__} consistent (metadata + plugin manifests)")
    sys.exit(0)
