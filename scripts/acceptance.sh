#!/usr/bin/env bash
# Released-artifact acceptance gate, shared by ci.yml and release.yml so the
# two can never drift. Installs the BUILT WHEEL into a clean venv and runs the
# documented surface from a NEUTRAL working directory, so the source tree can
# never shadow the installed package (python -m prepends cwd to sys.path).
#
# Usage: scripts/acceptance.sh [venv-dir]   (run after `python -m build`)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$(mktemp -d)/venv}"

python -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$ROOT"/dist/*.whl

cd "$(mktemp -d)"

"$VENV/bin/wikimoth" --version
"$VENV/bin/python" -m wikimoth --help
# Guards take absolute paths; cwd stays neutral so they exercise the WHEEL.
"$VENV/bin/python" "$ROOT/tests/mcp_acceptance.py"
"$VENV/bin/python" "$ROOT/tests/test_entry_points.py"
"$VENV/bin/python" "$ROOT/tests/test_version_sync.py"

echo "acceptance: OK"
