#!/usr/bin/env node
'use strict';
/*
 * wikimoth-mcp - zero-dependency npx launcher for the WikiMoth MCP server.
 *
 * `npx -y wikimoth-mcp` spawns the Python `wikimoth mcp` stdio server so any MCP
 * client (Claude Desktop/Code, Cursor, Windsurf, ...) can add WikiMoth with the
 * npx pattern it already uses, without Python-toolchain awareness.
 *
 * Behaviour:
 *  - Finds a Python that can `import wikimoth` (WIKIMOTH_PYTHON, then python3,
 *    python, and on Windows the `py -3` launcher), resolving names PATHEXT-aware
 *    so .cmd/.bat shims and the py launcher are found, and probing with a timeout
 *    so a hung interpreter can never wedge the launcher.
 *  - If none, self-heals via `uvx --from 'wikimoth>=0.2,<0.3' wikimoth mcp`
 *    (ephemeral install pinned to this launcher's expected server contract).
 *  - Injects a resolved vault path (WIKIMOTH_VAULT / --vault / <cwd>/.wikimoth/vault),
 *    handling both `--vault PATH` and `--vault=PATH`, so the server never silently
 *    reads an empty vault from the client's working directory.
 *  - Passes stdio through untouched (the MCP JSON-RPC channel; all launcher
 *    diagnostics go to stderr) and forwards the signals that exist on the platform.
 */

const { spawnSync, spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

const IS_WIN = process.platform === 'win32';
const PROBE_TIMEOUT_MS = 8000;

function log(msg) {
  process.stderr.write(`[wikimoth-mcp] ${msg}\n`);
}

// Resolve a command to an absolute executable, honoring PATHEXT on Windows so
// .cmd/.bat shims (a scoop/pipx uvx, the py launcher) are found: Node's
// shell-less spawn only appends .exe. An explicit path is used as-is if it exists.
function resolveExecutable(cmd) {
  if (cmd.includes('/') || cmd.includes('\\')) {
    if (fs.existsSync(cmd)) return cmd;
    return IS_WIN && fs.existsSync(cmd + '.exe') ? cmd + '.exe' : null;
  }
  const exts = IS_WIN ? (process.env.PATHEXT || '.EXE;.CMD;.BAT;.COM').split(';') : [''];
  for (const dir of (process.env.PATH || '').split(path.delimiter)) {
    if (!dir) continue;
    for (const ext of exts) {
      const candidate = path.join(dir, cmd + ext);
      try {
        if (fs.statSync(candidate).isFile()) return candidate;
      } catch (_) { /* keep looking */ }
    }
  }
  return null;
}

// Run a resolved executable with a bounded timeout; true only on a clean exit 0.
function runsOk(exe, args) {
  try {
    const r = spawnSync(exe, args, {
      stdio: 'ignore', timeout: PROBE_TIMEOUT_MS, killSignal: 'SIGKILL',
    });
    return !r.error && r.status === 0;
  } catch (_) {
    return false;
  }
}

// Explicit vault from forwarded args, supporting `--vault PATH` and `--vault=PATH`.
// Returns the value, or null when absent or dangling (flag with no value).
function explicitVault(a) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] === '--vault') {
      const next = a[i + 1];
      return next !== undefined && !next.startsWith('-') ? next : null;
    }
    if (a[i].startsWith('--vault=')) return a[i].slice('--vault='.length);
  }
  return null;
}

function main() {
  const forwarded = process.argv.slice(2);

  const given = explicitVault(forwarded);
  const vault = given
    ? path.resolve(given)
    : process.env.WIKIMOTH_VAULT
      ? path.resolve(process.env.WIKIMOTH_VAULT)
      : path.join(process.cwd(), '.wikimoth', 'vault');

  // Interpreter candidates. `pre` are leading args (e.g. the Windows py launcher).
  const candidates = [];
  if (process.env.WIKIMOTH_PYTHON) {
    candidates.push({ name: process.env.WIKIMOTH_PYTHON, pre: [], pinned: true });
  }
  candidates.push({ name: 'python3', pre: [] }, { name: 'python', pre: [] });
  if (IS_WIN) candidates.push({ name: 'py', pre: ['-3'] });

  let cmd = null;
  let baseArgs = null;
  for (const c of candidates) {
    const exe = resolveExecutable(c.name);
    if (exe && runsOk(exe, c.pre.concat(['-c', 'import wikimoth']))) {
      cmd = exe;
      baseArgs = c.pre.concat(['-m', 'wikimoth', 'mcp']);
      break;
    }
    if (c.pinned) {
      log(`WIKIMOTH_PYTHON=${c.name} cannot import wikimoth; ignoring the pin and auto-detecting.`);
    }
  }

  if (!cmd) {
    const uvx = resolveExecutable('uvx');
    if (uvx && runsOk(uvx, ['--version'])) {
      log('WikiMoth not importable from any Python on PATH.');
      log('First run may be slow: uvx is fetching wikimoth into an ephemeral env...');
      cmd = uvx;
      baseArgs = ['--from', 'wikimoth>=0.2,<0.3', 'wikimoth', 'mcp'];
    } else {
      log('No Python with WikiMoth installed, and uvx is not available.');
      log('Fix: `pip install wikimoth` (Python 3.10+), or install uv: https://docs.astral.sh/uv/');
      log('Or set WIKIMOTH_PYTHON to a Python that has wikimoth installed.');
      process.exit(1);
    }
  }

  // If the user gave an explicit --vault (either form) forward args as-is;
  // otherwise inject our resolved vault, dropping any dangling bare --vault so
  // the Python argparse does not error. Keep WIKIMOTH_VAULT consistent either way.
  let args;
  if (given) {
    args = baseArgs.concat(forwarded);
  } else {
    const cleaned = forwarded.filter((t) => t !== '--vault');
    args = baseArgs.concat(['--vault', vault], cleaned);
  }
  const env = Object.assign({}, process.env, { WIKIMOTH_VAULT: vault });

  log(`server: ${cmd} ${baseArgs.join(' ')}`);
  log(`vault:  ${vault}`);

  const child = spawn(cmd, args, { stdio: 'inherit', env });

  const signals = IS_WIN ? ['SIGINT', 'SIGBREAK'] : ['SIGINT', 'SIGTERM', 'SIGHUP'];
  for (const sig of signals) {
    process.on(sig, () => {
      try { child.kill(sig); } catch (_) { /* child already gone */ }
    });
  }
  child.on('error', (err) => {
    log(`failed to start the server: ${err.message}`);
    process.exit(1);
  });
  child.on('exit', (code, signal) => {
    if (signal) {
      try { process.kill(process.pid, signal); } catch (_) { process.exit(1); }
      return;
    }
    process.exit(code == null ? 0 : code);
  });
}

main();
