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
 *  - WIKIMOTH_PYTHON, when set, is trusted verbatim (no probing; a wrong pin
 *    fails loudly at server start). Otherwise the launcher finds a Python that
 *    can `import wikimoth` (python3, python, and `py -3` on Windows), with
 *    bounded-timeout probes so a hung interpreter can never wedge the launcher.
 *  - On Windows, real binaries (.exe/.com) are preferred across the whole PATH;
 *    .cmd/.bat shims (scoop/pipx uvx, pyenv-win) are spawned through a shell,
 *    because Node >= 18.20 refuses to spawn them directly (CVE-2024-27980).
 *    On POSIX the executable bit is honored, matching execvp.
 *  - If no Python has WikiMoth, self-heals via `uvx --from 'wikimoth>=0.2,<0.3'`
 *    (ephemeral install pinned to this launcher's expected server contract).
 *  - Injects a resolved vault path (last `--vault`/`--vault=` wins, matching
 *    argparse; then WIKIMOTH_VAULT; then <cwd>/.wikimoth/vault) so the server
 *    never silently reads an empty vault from the client's working directory.
 *    A valueless --vault fails fast, exactly like the Python CLI.
 *  - Passes stdio through untouched (the MCP JSON-RPC channel; all launcher
 *    diagnostics go to stderr) and dies by the child's signal when it is killed.
 */

const { spawnSync, spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');

const IS_WIN = process.platform === 'win32';
const PROBE_TIMEOUT_MS = 8000;
const MIN_HYGIENE_VERSION = [0, 2];

function log(msg) {
  process.stderr.write(`[wikimoth-mcp] ${msg}\n`);
}

function isExecutable(p) {
  try {
    if (!fs.statSync(p).isFile()) return false;
    if (!IS_WIN) fs.accessSync(p, fs.constants.X_OK); // execvp parity
    return true;
  } catch (_) {
    return false;
  }
}

// Resolve a command to an absolute executable. On Windows, prefer real
// binaries (.exe/.com) across the WHOLE PATH before .cmd/.bat shims, which
// Node can only run through a shell.
function resolveExecutable(cmd) {
  if (cmd.includes('/') || cmd.includes('\\')) {
    if (isExecutable(cmd)) return cmd;
    if (IS_WIN) {
      for (const ext of ['.exe', '.com', '.cmd', '.bat']) {
        if (isExecutable(cmd + ext)) return cmd + ext;
      }
    }
    return null;
  }
  const dirs = (process.env.PATH || '').split(path.delimiter).filter(Boolean);
  const extGroups = IS_WIN ? [['.exe', '.com'], ['.cmd', '.bat']] : [['']];
  for (const exts of extGroups) {
    for (const dir of dirs) {
      for (const ext of exts) {
        const candidate = path.join(dir, cmd + ext);
        if (isExecutable(candidate)) return candidate;
      }
    }
  }
  return null;
}

const needsShell = (exe) => IS_WIN && /\.(cmd|bat)$/i.test(exe);

// cmd.exe-safe quoting for the shell path (Node does not quote args itself).
function winQuote(s) {
  return /[\s"&()^!<>|;,]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function spawnSmart(exe, args, opts) {
  if (needsShell(exe)) {
    const cmdline = [winQuote(exe), ...args.map(winQuote)].join(' ');
    return spawn(cmdline, Object.assign({ shell: true }, opts));
  }
  return spawn(exe, args, opts);
}

function spawnSyncSmart(exe, args, opts) {
  if (needsShell(exe)) {
    const cmdline = [winQuote(exe), ...args.map(winQuote)].join(' ');
    return spawnSync(cmdline, Object.assign({ shell: true }, opts));
  }
  return spawnSync(exe, args, opts);
}

// One bounded probe: does this interpreter have wikimoth, and which version?
function wikimothVersion(exe, pre) {
  try {
    const r = spawnSyncSmart(
      exe,
      pre.concat(['-c', 'import wikimoth,sys;sys.stdout.write(getattr(wikimoth,"__version__","0"))']),
      { stdio: ['ignore', 'pipe', 'ignore'], timeout: PROBE_TIMEOUT_MS, killSignal: 'SIGKILL', encoding: 'utf8' },
    );
    if (r.error || r.status !== 0) return null;
    return (r.stdout || '').trim() || '0';
  } catch (_) {
    return null;
  }
}

function versionBelow(v, [maj, min]) {
  const parts = String(v).split('.').map((n) => parseInt(n, 10) || 0);
  return parts[0] < maj || (parts[0] === maj && (parts[1] || 0) < min);
}

// Explicit vault from forwarded args, supporting `--vault PATH` and
// `--vault=PATH`. argparse is last-wins, so the LAST occurrence counts.
// Returns {value} for a usable vault, {dangling:true} for a valueless flag,
// or null when absent.
function explicitVault(a) {
  let found = null;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === '--vault') {
      const next = a[i + 1];
      found = next !== undefined && !next.startsWith('-') ? { value: next } : { dangling: true };
    } else if (a[i].startsWith('--vault=')) {
      const v = a[i].slice('--vault='.length);
      found = v ? { value: v } : { dangling: true };
    }
  }
  return found;
}

function main() {
  const forwarded = process.argv.slice(2);

  const explicit = explicitVault(forwarded);
  if (explicit && explicit.dangling) {
    // Match the Python CLI, which fails loudly on a valueless --vault;
    // starting anyway on a silently-substituted default misreads vaults.
    log('error: --vault needs a value (e.g. --vault /path/to/vault).');
    process.exit(2);
  }
  const vault = explicit
    ? path.resolve(explicit.value)
    : process.env.WIKIMOTH_VAULT
      ? path.resolve(process.env.WIKIMOTH_VAULT)
      : path.join(process.cwd(), '.wikimoth', 'vault');

  let cmd = null;
  let baseArgs = null;
  let version = null;

  if (process.env.WIKIMOTH_PYTHON) {
    // An explicit pin is trusted, not probed: it skips auto-detection, and a
    // wrong pin fails loudly at server start (visible in client logs).
    const pinned = resolveExecutable(process.env.WIKIMOTH_PYTHON);
    if (!pinned) {
      log(`error: WIKIMOTH_PYTHON=${process.env.WIKIMOTH_PYTHON} not found or not executable.`);
      process.exit(1);
    }
    cmd = pinned;
    baseArgs = ['-m', 'wikimoth', 'mcp'];
  } else {
    const seen = new Set();
    const candidates = [{ name: 'python3', pre: [] }, { name: 'python', pre: [] }];
    if (IS_WIN) candidates.push({ name: 'py', pre: ['-3'] });
    for (const c of candidates) {
      const exe = resolveExecutable(c.name);
      if (!exe) continue;
      let real = exe;
      try { real = fs.realpathSync(exe); } catch (_) { /* keep exe */ }
      if (seen.has(real)) continue; // python3/python often alias the same binary
      seen.add(real);
      const v = wikimothVersion(exe, c.pre);
      if (v !== null) {
        cmd = exe;
        baseArgs = c.pre.concat(['-m', 'wikimoth', 'mcp']);
        version = v;
        break;
      }
    }
  }

  if (!cmd) {
    const uvx = resolveExecutable('uvx');
    if (uvx) {
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

  if (version && versionBelow(version, MIN_HYGIENE_VERSION)) {
    log(`warning: wikimoth ${version} found; the memory-hygiene tools need >= 0.2 (pip install -U wikimoth).`);
  }

  const args = explicit
    ? baseArgs.concat(forwarded)
    : baseArgs.concat(['--vault', vault], forwarded);
  const env = Object.assign({}, process.env, { WIKIMOTH_VAULT: vault });

  log(`server: ${cmd}${version ? ` (wikimoth ${version})` : ''}`);
  log(`vault:  ${vault}`);

  const child = spawnSmart(cmd, args, { stdio: 'inherit', env });

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
      // Die by the child's signal. Our forwarding handlers must be removed
      // first, or the re-raise is swallowed and the launcher lingers exit-0.
      for (const sig of signals) process.removeAllListeners(sig);
      try { process.kill(process.pid, signal); } catch (_) { process.exit(1); }
      setTimeout(() => process.exit(1), 100); // fallback if the signal is non-fatal here
      return;
    }
    process.exit(code == null ? 0 : code);
  });
}

main();
