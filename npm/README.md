# wikimoth-mcp

`npx` launcher for the [WikiMoth](https://github.com/juliangeymonat-jpg/wikimoth) MCP server: deterministic, token-minimal memory for Claude and agents over a `[[wikilink]]` markdown vault, with **no LLM call in the retrieval loop**.

This is a thin, zero-dependency wrapper. It spawns the Python `wikimoth mcp` stdio server so any MCP client can add WikiMoth with the `npx` pattern it already uses. The retrieval engine stays Python (that is where it lives); this package only makes it launchable from the Node/MCP world.

## Requirements

- **Node.js >= 18**
- Either **WikiMoth installed on a Python 3.10+** (`pip install wikimoth`), **or** [`uv`](https://docs.astral.sh/uv/) available (the launcher will `uvx --from wikimoth` an ephemeral install for you).

## Use it in an MCP client

Claude Code:

```bash
claude mcp add wikimoth -- npx -y wikimoth-mcp
```

Claude Desktop / Cursor / Windsurf (`mcpServers` config):

```json
{
  "mcpServers": {
    "wikimoth": {
      "command": "npx",
      "args": ["-y", "wikimoth-mcp"],
      "env": { "WIKIMOTH_VAULT": "/absolute/path/to/your/vault" }
    }
  }
}
```

The server exposes `recall`, `status`, and the memory-hygiene tools (`list_conflicts`, `list_lint`, `list_duplicates`, `list_fading`, `supersede`).

## Vault resolution

An MCP client launches the server from **its own** working directory, so a relative default would read the wrong (usually empty) folder. The launcher resolves the vault, in order:

1. `--vault PATH` passed through to the server
2. `WIKIMOTH_VAULT` environment variable
3. `<cwd>/.wikimoth/vault` (fallback)

and injects the resolved absolute path into the server's environment, printing it to stderr on startup. **Set `WIKIMOTH_VAULT` to the folder you actually capture into** (the same one `wikimoth install` uses) so recall reads your real notes.

## Environment variables

- `WIKIMOTH_VAULT` — the vault directory to serve (recommended).
- `WIKIMOTH_PYTHON` — pin the exact Python interpreter to run (skips auto-detection).

## How it works

The launcher looks for a Python that can `import wikimoth` (`WIKIMOTH_PYTHON`, then `python3`, `python`, and on Windows the `py -3` launcher) and runs `python -m wikimoth mcp`. If none is found it falls back to `uvx --from 'wikimoth>=0.2,<0.3' wikimoth mcp` (an ephemeral install; the **first** such run downloads WikiMoth, so it may exceed a strict client startup timeout — `pip install wikimoth` ahead of time avoids this). `stdin`/`stdout` are passed through untouched (the MCP JSON-RPC channel); all launcher diagnostics go to `stderr`.

This launcher is version-independent from the Python package: it runs whatever WikiMoth is installed (or `uvx`-installs a compatible one), so its own npm version does not track the Python release.

Apache-2.0.
