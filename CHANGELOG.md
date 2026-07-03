# Changelog

All notable changes to the `wikimoth` package and the `wikimoth-mcp` npm launcher.

## 0.2.2 - 2026-07-03

Release-infrastructure and hygiene release. No functional changes to retrieval, capture, the MCP server, or the launcher.

- **npm Trusted Publishing**: the tag-triggered release workflow now publishes the `wikimoth-mcp` launcher to npm via OIDC (with automatic provenance), alongside the existing PyPI Trusted Publishing. No stored registry tokens anywhere in the release path.
- **Repo hygiene**: added `SECURITY.md` (private vulnerability reporting via GitHub Security Advisories), `CONTRIBUTING.md` (stdlib-core and determinism constraints, version-bump protocol), and `CITATION.cff`.
- **Version-drift guard extended**: `tests/test_version_sync.py` now also covers `CITATION.cff`.
- **README**: CI / PyPI / npm / license badges.

## 0.2.1 - 2026-07-02

- **MCP protocol-version negotiation fixed**: the server now answers `initialize` with a protocol version it actually supports (`2025-06-18`, or `2024-11-05` when the client requests it) instead of echoing the client's requested version verbatim.
- **`wikimoth --version`** flag on the CLI.
- **`wikimoth-mcp` npm launcher** (first npm release): `npx -y wikimoth-mcp` finds a Python with `wikimoth` installed (or falls back to `uvx`), with Windows-safe executable resolution and spawning, explicit-vault validation, and clean signal forwarding.
- **MCP registry**: published `io.github.juliangeymonat-jpg/wikimoth` (`server.json`) to the official MCP registry.
- **Release gates**: CI + tag-triggered release workflow; both run the released-artifact acceptance gate (`scripts/acceptance.sh`: build the wheel, install it in a fresh venv, run the documented commands from a neutral directory) and the MCP handshake acceptance (`tests/mcp_acceptance.py`). npm launcher CI on Node 18/20/22.
- **Version-drift guard**: `tests/test_version_sync.py` keeps `pyproject.toml`, `wikimoth/__init__.py`, both plugin manifests, `npm/package.json` and `server.json` in lockstep.

Older releases predate this changelog.
