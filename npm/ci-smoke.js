'use strict';
/*
 * CI smoke for the launcher (not shipped in the npm tarball; see "files").
 * Spawns bin.js against a Python that has wikimoth installed and asserts the
 * MCP initialize + tools/list handshake completes. Exits nonzero on failure.
 *
 * Usage: node npm/ci-smoke.js [python-exe]
 */
const { spawn } = require('node:child_process');
const path = require('node:path');
const os = require('node:os');
const fs = require('node:fs');

const bin = path.join(__dirname, 'bin.js');
const py = process.argv[2] || '';
const vault = fs.mkdtempSync(path.join(os.tmpdir(), 'wm-ci-'));

const env = Object.assign({}, process.env, { WIKIMOTH_VAULT: vault });
if (py) env.WIKIMOTH_PYTHON = py;

const child = spawn(process.execPath, [bin], { env, stdio: ['pipe', 'pipe', 'inherit'] });

let buf = '';
const responses = [];
child.stdout.on('data', (d) => {
  buf += d.toString();
  let idx;
  while ((idx = buf.indexOf('\n')) !== -1) {
    const line = buf.slice(0, idx).trim();
    buf = buf.slice(idx + 1);
    if (line) {
      try { responses.push(JSON.parse(line)); } catch (_) { /* non-JSON noise is a failure via timeout */ }
      check();
    }
  }
});

function send(obj) { child.stdin.write(JSON.stringify(obj) + '\n'); }
send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'ci-smoke', version: '0' } } });
send({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });

function finish(code) {
  try { child.stdin.end(); } catch (_) { /* already closed */ }
  child.kill();
  fs.rmSync(vault, { recursive: true, force: true });
  process.exit(code);
}

function check() {
  const init = responses.find((r) => r.id === 1);
  const tools = responses.find((r) => r.id === 2);
  if (init && tools) {
    const si = (init.result || {}).serverInfo || {};
    const names = ((tools.result || {}).tools || []).map((t) => t.name);
    console.log(`ci-smoke OK: serverInfo=${JSON.stringify(si)} tools=${names.length}`);
    finish(si.name === 'wikimoth' && names.length > 0 ? 0 : 1);
  }
}

setTimeout(() => { console.error('ci-smoke TIMEOUT: no handshake in 60s'); finish(2); }, 60000);
