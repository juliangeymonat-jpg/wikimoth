# Security Policy

## Supported versions

Only the **latest release** receives security fixes (PyPI `wikimoth`, npm `wikimoth-mcp`). Older releases are not patched; upgrade with `pip install wikimoth` (add `-U` to force the upgrade); the `npx wikimoth-mcp` launcher installs the latest published version on first run (`npx wikimoth-mcp@latest` forces a re-resolve).

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub Security Advisories:
**[Report a vulnerability](https://github.com/juliangeymonat-jpg/wikimoth/security/advisories/new)** (repository → Security → Report a vulnerability).

You will normally get a first response within **7 days**. Once a fix ships, the advisory is published and you are credited (unless you prefer otherwise).

## Scope notes

WikiMoth is a **local-first** tool: the MCP server speaks JSON-RPC over stdin/stdout, opens no network ports, and makes no outbound network calls at query time. The optional `wikimoth serve` viewer binds `127.0.0.1` by default (a user can opt into another interface with `--host`). Given that model, what is definitely in scope:

- **Vault escape.** The server should only ever read markdown inside the configured vault. Any way a crafted note, `[[wikilink]]`, or tool argument can make it read or write files *outside* the vault (path traversal, symlink tricks, absolute-path links) is a vulnerability.
- **Content-driven code execution.** Notes are data. Any way vault content can lead to code execution on the host (in the Python package, the MCP server, or the npm launcher) is a vulnerability.
- **Launcher spawn safety.** The `wikimoth-mcp` npm launcher resolves and spawns a local Python/`wikimoth` executable. Any way an unprivileged actor can redirect that resolution to a binary of their choosing (beyond the documented `WIKIMOTH_PYTHON` override, which trusts the user's own environment) is a vulnerability.

Prompt-injection *content* risks (a malicious note steering the agent that reads it) are a property of any retrieval system feeding an LLM and are out of scope unless they cross into one of the categories above.
