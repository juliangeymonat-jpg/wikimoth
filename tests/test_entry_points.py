# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Entry-point completeness guard.

Every console_script the INSTALLED wikimoth distribution declares must
resolve: importing its target module (and looking up the attribute) must
succeed. A regression guard against the packaging class of defect (a declared
console_script whose target module does not ship). Runnable standalone::

    python tests/test_entry_points.py
"""
from __future__ import annotations

import importlib
from importlib.metadata import entry_points


def _wikimoth_console_scripts():
    eps = entry_points()
    group = (eps.select(group="console_scripts")
             if hasattr(eps, "select") else eps.get("console_scripts", []))
    return [ep for ep in group if ep.value.split(":")[0].split(".")[0] == "wikimoth"]


def check_console_scripts() -> list[str]:
    failures: list[str] = []
    scripts = _wikimoth_console_scripts()
    if not scripts:
        return ["no wikimoth console_scripts found in the installed distribution"]
    for ep in scripts:
        module_path, _, attr = ep.value.partition(":")
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 — any import failure is a real defect
            failures.append(f"{ep.name} = {ep.value}: import failed: "
                            f"{type(exc).__name__}: {exc}")
            continue
        if attr and not hasattr(mod, attr.split(".")[0]):
            failures.append(f"{ep.name} = {ep.value}: {module_path} "
                            f"has no attribute {attr!r}")
    return failures


def test_all_console_scripts_resolve():
    failures = check_console_scripts()
    assert not failures, "Broken console scripts:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    import sys

    fails = check_console_scripts()
    if fails:
        print("BROKEN console scripts:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"OK: {len(_wikimoth_console_scripts())} wikimoth console script(s) resolve")
    sys.exit(0)
