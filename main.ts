// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE, probes each provider's
// catalog route, builds one catalog, registers with the env, and serves the
// twenty requests the env's client sends, one at a time, then exits 0.
//
// ROUTING: the manifest lists all twenty requests up front, so this server
// plans globally (lib/plan.ts) instead of routing greedily per request: the
// heaviest and most tightly-constrained requests get the scarce top models
// first, every assignment fits inside the bottom of each provider's budget
// range, and a provider that fails twice in a row is retired. On a 402, 429
// or 5xx (or any failed call) the same request is retried on the next-best
// qualifying free model, in rating order, until one serves or the candidates
// run out. A paid model is never a legal pick and is never chosen.
//
// WIRES: every client request is translated into the chosen provider's wire
// (openai, anthropic or gemini) through one neutral shape (lib/wire.ts), so
// the system prompt, user text, images, tools, stop sequences, max_tokens
// and JSON mode travel untouched; the provider's answer is normalized and
// re-emitted in the client's wire, JSON or SSE, with the provider's usage
// and the stop reason mapped.
//
// DETERMINISM: no clock and no randomness anywhere in the request path; the
// port is the only thing the OS picks, and it never reaches a response. The
// catalog printed by `node main.ts --list` is the same object the register
// body carries, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  answerToAnthropic,
  answerToAnthropicStream,
  answerToOpenAI,
  answerToOpenAIStream,
  anthropicError,
  anthropicModelList,
  anthropicToNeutral,
  neutralToAnthropic,
  neutralToGemini,
  neutralToOpenAI,
  openAIError,
  openAIModelList,
  openAIToNeutral,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type NeutralRequest,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { candidatesFor, Planner, type Candidate } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
const planner = new Planner();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(_clientApi: 'openai' | 'anthropic', neutral: NeutralRequest): Requirement {
  let vision = false;
  for (const msg of neutral.messages) {
    if (msg.role !== 'user') continue;
    for (const part of msg.parts) if (part.imageData !== undefined || part.imageUrl !== undefined) vision = true;
  }
  return {
    tools: neutral.tools.length > 0,
    vision,
    json: neutral.jsonMode,
  };
}

interface PendingRequest {
  id: string;
  clientApi: 'openai' | 'anthropic';
  neutral: NeutralRequest;
  requires: Requirement;
  jsonOnWire: boolean;
}

/** The manifest row for the request now arriving, with its requirements. */
function pendingFor(clientApi: 'openai' | 'anthropic', body: Json, neutral: NeutralRequest): PendingRequest {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const id = entry !== undefined ? entry.id : `extra-${manifestIndex}`;
  const requires = entry !== undefined ? entry.requires : requirementsFromBody(clientApi, neutral);
  // JSON mode has to travel on the wire only when the client actually sent
  // response_format: an OpenAI client does, an Anthropic client never does.
  const jsonOnWire = clientApi === 'openai' && neutral.jsonMode;
  return { id, clientApi, neutral, requires, jsonOnWire };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  answer: NormalizedAnswer | null;
}

/** The provider's answer for one candidate, normalized, or a failure. */
async function callCandidate(candidate: Candidate, pending: PendingRequest): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = pending.neutral.stream;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = neutralToOpenAI(pending.neutral, candidate.model.id);
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    payload = neutralToAnthropic(pending.neutral, candidate.model.id);
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    payload = neutralToGemini(pending.neutral);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, answer: null };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, status: res.status, answer: null };
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  if (answer === null) return { ok: false, status: 502, answer: null };
  return { ok: true, status: res.status, answer };
}

/**
 * Serve one request: re-plan the requests still to come with the calls made
 * so far, try the plan's pick first, then every other qualifying free model
 * in rating order until one serves. A failed call spends budget and counts
 * toward retiring its provider, exactly as the env counts it.
 */
async function serve(pending: PendingRequest): Promise<{ answer: NormalizedAnswer; model: string } | null> {
  const manifest = brief.workload?.manifest ?? [];
  // The requests still to come are the manifest rows from the arrival index
  // onward (manifestIndex was already advanced for the current request).
  const future = manifest.slice(manifestIndex).map((entry) => ({
    id: entry.id,
    api: entry.api,
    stream: entry.stream,
    weight: entry.weight,
    requires: entry.requires,
  }));
  const current = {
    id: pending.id,
    api: pending.clientApi,
    stream: pending.neutral.stream,
    weight: (manifest[manifestIndex - 1]?.weight ?? 1) as 1 | 2,
    requires: pending.requires,
  };
  const all = [current, ...future];
  const clientApiOf = (req: { api: 'openai' | 'anthropic' }): 'openai' | 'anthropic' => req.api;
  const jsonOnWireOf = (req: { id: string; api: 'openai' | 'anthropic'; requires: Requirement }): boolean => {
    if (req.id === pending.id) return pending.jsonOnWire;
    // A future request's JSON mode travels on its wire only for an OpenAI
    // client with a json requirement; the manifest's json requirement is the
    // proxy the planner can see before the body arrives.
    return req.api === 'openai' && req.requires.json === true;
  };
  const plan = planner.plan(all, probes, brief.ratings, clientApiOf, jsonOnWireOf);
  const allCandidates = candidatesFor(probes, brief.ratings, pending.requires, pending.clientApi, pending.jsonOnWire);
  const order = planner.tryOrder(plan.get(pending.id) ?? null, allCandidates);
  for (const candidate of order) {
    const result = await callCandidate(candidate, pending);
    planner.recordCall(candidate.probe.provider.id, result.ok);
    if (result.ok && result.answer !== null) return { answer: result.answer, model: candidate.model.id };
    log(`failover: ${pending.id} via ${candidate.probe.provider.id}/${candidate.model.id} failed`);
  }
  return null;
}

function emit(res: ServerResponse, pending: PendingRequest, answer: NormalizedAnswer, model: string): void {
  // The model named back is the one that served the request.
  if (pending.clientApi === 'openai') {
    if (pending.neutral.stream) sendSse(res, answerToOpenAIStream(answer, model));
    else sendJson(res, 200, answerToOpenAI(answer, model));
  } else {
    if (pending.neutral.stream) sendSse(res, answerToAnthropicStream(answer, model));
    else sendJson(res, 200, answerToAnthropic(answer, model));
  }
}

/** POST /v1/chat/completions and POST /v1/messages share this path. */
async function completion(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const neutral = clientApi === 'openai' ? openAIToNeutral(body) : anthropicToNeutral(body);
  const pending = pendingFor(clientApi, body, neutral);
  const served = await serve(pending);
  if (served === null) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  emit(res, pending, served.answer, served.model);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await completion(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await completion(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
