// dev/verify-gates.mjs: local verifier for the free-model-aggregator entry.
//
// Checks, in order:
//   1. `node main.ts --list` with no trial env prints {"providers":[]} and exits 0.
//   2. `node main.ts --list` inside trial-0's env prints a catalog whose shape
//      matches the practice reference (every provider, base_url/key_env/wire,
//      and each provider's free model set), byte-stable across two runs.
//   3. Practice trials 0, 1 and 2, run manually with node (no Docker): start
//      program/env/server.mjs on the trial's input/env, run main.ts against
//      it, SIGTERM the env, score the export with program/scorer/score.mjs.
//   4. Determinism: trial 0 run twice leaves a byte-identical state export.
//
// Prints PASS/FAIL per row and exits 1 if any gate fails.
// Usage: node dev/verify-gates.mjs   (from the repo root)

import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, openSync, readFileSync, rmSync, writeSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const KIT = join(REPO, '..', 'kit', 'free-model-aggregator-01-kit');
const ENV_SERVER = join(KIT, 'program', 'env', 'server.mjs');
const SCORER = join(KIT, 'program', 'scorer', 'score.mjs');
const WORK = join(REPO, 'dev', '.verify-work');

const results = [];
function row(name, pass, detail) {
  results.push({ name, pass, detail });
  process.stdout.write(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail !== '' ? ` — ${detail}` : ''}\n`);
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], ...opts });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('error', (err) => resolve({ code: 1, stdout, stderr: String(err) }));
    child.on('close', (code) => resolve({ code: code ?? 1, stdout, stderr }));
  });
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function startEnv(n, dir) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(join(dir, 'state'), { recursive: true });
  const envFd = openSync(join(dir, 'env.log'), 'w');
  const envProc = spawn('node', [ENV_SERVER], {
    cwd: REPO,
    env: {
      ...process.env,
      MAKEFUN_STATE_DIR: join(dir, 'state'),
      MAKEFUN_ENV_DATA_DIR: join(KIT, 'practice', `trial-${n}`, 'input', 'env'),
      MAKEFUN_ENV_PORT: '8080',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  envProc.stdout.on('data', (d) => writeSync(envFd, d));
  envProc.stderr.on('data', (d) => writeSync(envFd, d));
  let ready = false;
  for (let i = 0; i < 100 && !ready; i++) {
    await sleep(100);
    ready = existsSync(join(dir, 'state', 'ready'));
  }
  if (!ready) {
    envProc.kill('SIGKILL');
    return null;
  }
  return envProc;
}

async function stopEnv(envProc) {
  envProc.kill('SIGTERM');
  await new Promise((resolve) => {
    const timer = setTimeout(() => { envProc.kill('SIGKILL'); resolve(); }, 5000);
    envProc.on('close', () => { clearTimeout(timer); resolve(); });
  });
}

async function listGate() {
  const bare = await run('node', ['main.ts', '--list'], { cwd: REPO, env: { ...process.env, MAKEFUN_PROMPT_FILE: '', MAKEFUN_ENV_URL: '' } });
  row('--list outside a trial', bare.code === 0 && bare.stdout.trim() === '{"providers":[]}', bare.code === 0 ? bare.stdout.trim() : `exit ${bare.code}: ${bare.stderr.slice(0, 200)}`);

  // --list probes the providers, so it needs the env up, exactly as a trial does.
  const envProc = await startEnv(0, join(WORK, 'list-env'));
  if (envProc === null) {
    row('--list in trial-0 env', false, 'env did not become ready');
    return null;
  }
  const env = {
    ...process.env,
    MAKEFUN_PROMPT_FILE: join(KIT, 'practice', 'trial-0', 'input', 'prompt.txt'),
    MAKEFUN_ENV_URL: 'http://127.0.0.1:8080',
  };
  const a = await run('node', ['main.ts', '--list'], { cwd: REPO, env });
  const b = await run('node', ['main.ts', '--list'], { cwd: REPO, env });
  await stopEnv(envProc);
  if (a.code !== 0) {
    row('--list in trial-0 env', false, `exit ${a.code}: ${a.stderr.slice(0, 300)}`);
    return null;
  }
  row('--list byte-stable across runs', a.stdout === b.stdout, '');
  let catalog = null;
  try {
    catalog = JSON.parse(a.stdout);
  } catch {
    row('--list catalog parses', false, 'not JSON');
    return null;
  }
  row('--list catalog parses', true, `${catalog.providers.length} providers`);
  const truth = JSON.parse(readFileSync(join(KIT, 'practice', 'trial-0', 'reference', 'catalog.json'), 'utf8'));
  let shapeOk = true;
  let shapeDetail = '';
  if (catalog.providers.length !== truth.providers.length) {
    shapeOk = false;
    shapeDetail = `holds ${catalog.providers.length} providers, reference has ${truth.providers.length}`;
  } else {
    for (const t of truth.providers) {
      const entry = catalog.providers.find((p) => p.id === t.id);
      if (entry === undefined) { shapeOk = false; shapeDetail = `missing provider ${t.id}`; break; }
      if (entry.base_url !== t.base_url || entry.key_env !== t.key_env || entry.wire !== t.wire) {
        shapeOk = false; shapeDetail = `provider ${t.id} base_url/key_env/wire mismatch`; break;
      }
      const free = (entry.models ?? []).filter((m) => m.free === true).map((m) => m.id).sort();
      const want = [...t.free_models].sort();
      if (free.join(',') !== want.join(',')) { shapeOk = false; shapeDetail = `provider ${t.id} free set mismatch`; break; }
      const paidMarkedFree = (entry.models ?? []).some((m) => m.free === true && t.paid_models.includes(m.id));
      if (paidMarkedFree) { shapeOk = false; shapeDetail = `provider ${t.id} marks a paid model free`; break; }
    }
  }
  row('--list catalog matches reference', shapeOk, shapeDetail);
  return a.stdout;
}

async function runTrial(n, dir) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(join(dir, 'state'), { recursive: true });
  mkdirSync(join(dir, 'scores'), { recursive: true });
  const envLog = join(dir, 'env.log');
  const agentLog = join(dir, 'agent.log');
  const envProc = spawn('node', [ENV_SERVER], {
    cwd: REPO,
    env: {
      ...process.env,
      MAKEFUN_STATE_DIR: join(dir, 'state'),
      MAKEFUN_ENV_DATA_DIR: join(KIT, 'practice', `trial-${n}`, 'input', 'env'),
      MAKEFUN_ENV_PORT: '8080',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const envFd = openSync(envLog, 'w');
  envProc.stdout.on('data', (d) => writeSync(envFd, d));
  envProc.stderr.on('data', (d) => writeSync(envFd, d));
  let ready = false;
  for (let i = 0; i < 100 && !ready; i++) {
    await sleep(100);
    ready = existsSync(join(dir, 'state', 'ready'));
  }
  if (!ready) {
    envProc.kill('SIGKILL');
    return { ok: false, detail: 'env did not become ready' };
  }
  const agentFd = openSync(agentLog, 'w');
  const agentProc = spawn('node', ['main.ts'], {
    cwd: REPO,
    env: {
      ...process.env,
      MAKEFUN_PROMPT_FILE: join(KIT, 'practice', `trial-${n}`, 'input', 'prompt.txt'),
      MAKEFUN_ENV_URL: 'http://127.0.0.1:8080',
      MAKEFUN_TRIAL_SEED: `verify-trial-${n}`,
      MAKEFUN_TRIAL_NOW: '2026-11-15T00:00:00Z',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  agentProc.stdout.on('data', (d) => writeSync(agentFd, d));
  agentProc.stderr.on('data', (d) => writeSync(agentFd, d));
  const agentExit = await new Promise((resolve) => {
    const timer = setTimeout(() => { agentProc.kill('SIGKILL'); resolve(-1); }, 120000);
    agentProc.on('close', (code) => { clearTimeout(timer); resolve(code ?? 1); });
  });
  envProc.kill('SIGTERM');
  await new Promise((resolve) => {
    const timer = setTimeout(() => { envProc.kill('SIGKILL'); resolve(); }, 5000);
    envProc.on('close', () => { clearTimeout(timer); resolve(); });
  });
  if (agentExit !== 0) return { ok: false, detail: `agent exited ${agentExit}` };
  const stateFile = join(dir, 'state', 'export', 'state.json');
  if (!existsSync(stateFile)) return { ok: false, detail: 'no state export' };
  const scored = await run('node', [SCORER], {
    env: {
      ...process.env,
      MAKEFUN_OUTPUT_DIR: join(dir, 'state', 'export'),
      MAKEFUN_REFERENCE_DIR: join(KIT, 'practice', `trial-${n}`, 'reference'),
      MAKEFUN_SCORES_DIR: join(dir, 'scores'),
    },
  });
  if (scored.code !== 0) return { ok: false, detail: `scorer failed: ${scored.stderr.slice(0, 200)}` };
  const scores = JSON.parse(readFileSync(join(dir, 'scores', 'scores.json'), 'utf8'));
  return { ok: true, scores, stateFile, detail: scored.stdout.trim() };
}

async function main() {
  mkdirSync(WORK, { recursive: true });
  const listOut = await listGate();
  const scoresByTrial = {};
  for (const n of [0, 1, 2]) {
    const r = await runTrial(n, join(WORK, `trial-${n}`));
    if (!r.ok) {
      row(`trial-${n}`, false, r.detail);
      continue;
    }
    scoresByTrial[n] = r.scores;
    const s = r.scores;
    row(
      `trial-${n}`,
      s.score >= 90 && s.served === 100 && s.wire === 100 && s.catalog === 100,
      `score ${s.score} (quality ${s.quality}, served ${s.served}, wire ${s.wire}, failover ${s.failover}, catalog ${s.catalog}), calls ${s.upstream_calls}, working ${s.working}`,
    );
  }
  // The --list document must be the register body's catalog, byte for byte:
  // the trial-0 state export carries the registered catalog.
  if (listOut !== null && scoresByTrial[0] !== undefined) {
    const registered = JSON.parse(readFileSync(join(WORK, 'trial-0', 'state', 'export', 'state.json'), 'utf8')).catalog;
    const listed = JSON.parse(listOut);
    row('--list equals registered catalog', JSON.stringify(listed) === JSON.stringify(registered), '');
  } else {
    row('--list equals registered catalog', false, 'no --list output or trial-0 result to compare');
  }
  const again = await runTrial(0, join(WORK, 'trial-0-repeat'));
  if (!again.ok) {
    row('determinism (trial-0 re-run)', false, again.detail);
  } else {
    const first = readFileSync(join(WORK, 'trial-0', 'state', 'export', 'state.json'));
    const second = readFileSync(join(WORK, 'trial-0-repeat', 'state', 'export', 'state.json'));
    row('determinism (trial-0 re-run)', first.equals(second), first.equals(second) ? `identical export, score ${again.scores.score}, calls ${again.scores.upstream_calls}` : 'state exports differ');
  }
  const failed = results.filter((r) => !r.pass).length;
  process.stdout.write(`\n${results.length - failed}/${results.length} gates passed\n`);
  process.exit(failed === 0 ? 0 : 1);
}

await main();
