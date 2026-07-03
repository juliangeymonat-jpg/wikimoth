# Contributing to WikiMoth

Thank you for considering a contribution. WikiMoth is a personal engineering project by Julian Geymonat, released under Apache 2.0 and developed in the open.

The design constraint that shapes every review: **the core stays pure stdlib and deterministic**. Retrieval, the wikilink graph, chunking, capture, the MCP server: no hard runtime dependency, no LLM in the retrieval loop, same input, same output. Anything that needs a third-party package goes behind an import-guarded extra (see `[project.optional-dependencies]` in `pyproject.toml`).

## Before you start

1. **Open an issue first** for any change larger than a typo or a docs nit. Aligning on scope before code makes a PR far more likely to land.
2. **Check existing issues** to avoid duplicate work.
3. **Respect the hard constraints**: pure-stdlib core (new dependencies only as opt-in extras), deterministic retrieval (no randomness, no wall-clock dependence in results), Apache 2.0 header on every new source file, conventional commits.

## Development setup

```bash
git clone https://github.com/juliangeymonat-jpg/wikimoth.git
cd wikimoth
pip install -e ".[dev]"
pytest
```

The npm launcher lives in `npm/` (a single `bin.js`, no dependencies). If you touch it:

```bash
node --check npm/bin.js
```

CI (`.github/workflows/npm-ci.yml`) also runs an MCP handshake smoke against the launcher on Node 18/20/22.

## Layout

- `wikimoth/`: the package. CLI (`cli.py`), MCP server (`mcp.py`), retrieval (`retrieval/`), capture, viewer.
- `tests/`: pytest suite plus two standalone gates: `tests/mcp_acceptance.py` (spawns the real server, full JSON-RPC handshake) and `tests/test_version_sync.py` (version drift guard).
- `scripts/acceptance.sh`: the release gate. Builds the wheel, installs it in a fresh venv, and runs the documented commands from a neutral directory. Run it before proposing release-adjacent changes.
- `npm/`: the `wikimoth-mcp` npx launcher.
- `plugin/` and `.claude-plugin/`: Claude Code plugin manifests.

## Version bumps

The version lives in more than one file by necessity (Python package, plugin manifests, npm launcher, MCP registry record, citation file). `tests/test_version_sync.py` is the single guard: it fails if `wikimoth/__init__.py`, `pyproject.toml`, `plugin/.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `npm/package.json`, `server.json`, and `CITATION.cff` disagree. If you add a new version-bearing file, add it to `_MANIFESTS` in that test in the same PR.

## Code style

- **Python**: ≥ 3.10.
- **License header**: every new Python source file starts with:
  ```python
  # Copyright 2026 Julian Geymonat
  # Licensed under the Apache License, Version 2.0
  ```
- **Determinism is a feature.** No `random`, no `set` iteration order leaking into output, no time-dependent ranking. If your change can produce two different outputs for the same vault and query, it will not land.
- **Token budget is a feature.** WikiMoth's value is feeding the reader less. Changes that grow the retrieved context need a measured justification (the benchmark script below).

## Benchmarks and claims

`scripts/run_agentic_benchmark.py` reproduces the README's agent-vs-WikiMoth comparison. Any PR that changes retrieval behavior should include a before/after run. Claims in README or docs must be reproducible from a committed script; qualitative claims about other systems belong in issues, not docs.

## Pull request process

1. Branch from `main`.
2. Use **conventional commits**: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
3. Run `pytest` and (if you touched the launcher) `node --check npm/bin.js` locally. CI runs the same plus the MCP acceptance handshake.
4. PR review is solo (single maintainer). Reviews focus on: determinism, stdlib-core discipline, version sync, and test coverage on the new code path.

## Reporting bugs

Please include:

- **Environment**: Python version, OS, how you run it (CLI, `npx wikimoth-mcp`, Claude Code plugin).
- **Reproduction**: the smallest vault + query that triggers it (a few markdown files inline in the issue is perfect).
- **Expected vs observed**, including the note-chain WikiMoth returned.
- **Version**: `wikimoth --version`.

For security problems, do not open an issue: see [SECURITY.md](SECURITY.md).

## License

By contributing you agree that your contributions are licensed under Apache 2.0, the same license as the project.
