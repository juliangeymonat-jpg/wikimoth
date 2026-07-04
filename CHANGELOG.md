# Changelog

All notable changes to the `wikimoth` package and the `wikimoth-mcp` npm launcher.

## 0.2.3 - 2026-07-03

Adoption and robustness release, from an adversarial review of the first-run surface.

- **`wikimoth demo`**: instant multi-hop recall over a bundled demo vault (shipped in the wheel). The answer is reached by following `[[wikilinks]]` two hops out from the notes the question matches, with the token saving shown. No capture, no key, no setup: the first command after `pip install` returns a real note-chain.
- **npm launcher**: a found-but-too-old wikimoth (predating the `mcp` server subcommand) is now skipped so the `uvx` self-heal runs instead of crashing; a `WIKIMOTH_PYTHON` pin is probed once and fails with an actionable error instead of a raw Python traceback; the vault is passed via the `WIKIMOTH_VAULT` env var (immune to cmd.exe `%VAR%` expansion on the `.cmd`/`.bat` shell path); on Windows the child is tree-killed on shutdown so the server is not orphaned through a shim.
- **MCP server**: a malformed request (id present, no `method`) now gets a JSON-RPC `-32600` error keyed to its id instead of silently hanging a strict client; `recall`/`status` distinguish an empty-but-existing vault ("no notes yet") from a genuine miss; every no-vault message names `WIKIMOTH_VAULT` (the fix that actually works for a GUI MCP client whose cwd is the app directory).
- **CLI**: bare `wikimoth` prints help (exit 0) instead of an argparse error; `--user`/`--project` are mutually exclusive; `--top-k` is validated uniformly across `serve`/`mcp`/`recall`.
- **README**: an "Already have a vault?" one-command on-ramp for existing Obsidian/notes vaults; the hero leads with `wikimoth demo`; the Python quickstart flags that `EchoReader` is a deterministic stub, not a prose answer.
- **Release**: npm (reversible) publishes before the immutable PyPI upload, so a broken npm publish can no longer strand a PyPI-only release; `skip-existing` on PyPI lets a re-run recover; least-privilege top-level `permissions`; a per-tag `concurrency` guard; the npm-upgrade step is pinned (`npm@^11.5.1`) and only runs when below the OIDC minimum.

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
