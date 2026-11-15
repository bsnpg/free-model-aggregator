// free-model-aggregator-01: one server over ten simulated free-tier model
// providers, speaking the OpenAI Chat Completions API and the Anthropic
// Messages API, in JSON and in server-sent events.
//
// The trial gives this server a brief (the provider directory, the sealed
// quality table and the workload manifest) at $MAKEFUN_PROMPT_FILE and an
// env at $MAKEFUN_ENV_URL. It reads the brief, probes each provider's
// catalog route, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//      there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port",
//      "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the
//      auth header it got: x-api-key means Anthropic, Authorization means
//      OpenAI), POST /v1/chat/completions and POST /v1/messages, each in
//      JSON or in server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the
//      env says {"done": true}. A process still running when the wall ends
//      scores zero, so it never waits for anything else.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before
// any server starts. Run outside a trial, with no brief and no env, it
// prints {"providers":[]}: it has nothing to probe.
//
// Routing: the plan in lib/plan.ts assigns each request the best free
// model meeting its requirements under the providers' per-trial budgets
// (planned against the bottom of each budget range). A refused call goes
// straight to failover on the request's candidates best-first, preferring
// providers not yet called. Nothing here reads the clock or rolls dice, so
// a replay leaves the same state export as the live pass. Exits 0 when the
// env says the workload is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { isRecord, readBrief, type JsonObject } from './lib/brief.ts';
import {
  anthropicModelList,
  buildCatalog,
  openAIModelList,
  probeProvider,
  servedModels,
  type Catalog,
  type Probe,
} from './lib/catalog.ts';
import { Router, candidatesFor } from './lib/plan.ts';
import type { Candidate } from './lib/plan.ts';
import { callOnce } from './lib/upstream.ts';
import type { Attempt } from './lib/upstream.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicStreamAnswer,
  openAIError,
  openAIJsonAnswer,
  openAIStreamAnswer,
  wantsJsonMode,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most failover attempts after the primary and its one retry. */
const MAX_FAILOVER_ATTEMPTS = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival
 *  order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document
 *  and the register body) and the routing plan. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (ENV_URL !== '') {
    probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
    for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  }
  catalog = buildCatalog(probes, brief.ratings);
  router = new Router(brief.workload, probes, brief.ratings);
}

/** The requirements of the request now arriving: the manifest's row by
 *  arrival order, else what the body itself declares. */
function requirementsFor(body: JsonObject): { id: string; requires: import('./lib/brief.ts').Requirement; jsonMode: boolean } {
  const row = brief.workload[manifestIndex];
  manifestIndex += 1;
  const jsonMode = wantsJsonMode(body, row !== undefined && row.api === 'anthropic' ? 'anthropic' : 'openai');
  if (row !== undefined) return { id: row.id, requires: row.requires, jsonMode };
  // No manifest row for this request: read the requirements off the body.
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    id: `request-${manifestIndex}`,
    requires: {
      tools: Array.isArray(body.tools) && body.tools.length > 0,
      vision,
      json: jsonMode,
    },
    jsonMode,
  };
}

/** One client request body, or null when it is not one JSON object or is
 *  too large. */
async function readJson(req: IncomingMessage): Promise<JsonObject | null> {
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

function sendError(res: ServerResponse, api: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(probes, brief.ratings);
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Pipe a same-wire provider event stream straight to the client. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Answer the client from a normalized provider answer, in its own wire. */
function answerClient(
  res: ServerResponse,
  api: 'openai' | 'anthropic',
  stream: boolean,
  attempt: Attempt,
  model: string,
): void {
  const answer = attempt.answer;
  if (answer === null) {
    sendError(res, api, 502, 'upstream_error', 'the provider answered in a shape this server could not read');
    return;
  }
  if (!stream) {
    sendJson(res, 200, api === 'anthropic' ? anthropicJsonAnswer(answer, model) : openAIJsonAnswer(answer, model));
    return;
  }
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  res.end(api === 'anthropic' ? anthropicStreamAnswer(answer, model) : openAIStreamAnswer(answer, model));
}

/**
 * One client request: route it, call the provider, fail over on refusal.
 * The attempt order is the plan's primary, the same provider once more,
 * then the request's candidates best-first (skipping spent providers).
 * Every attempt is counted against its provider's budget, success or
 * refusal, exactly as the provider counts it.
 */
async function serve(api: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (router === null) return sendError(res, api, 503, 'overloaded_error', 'the router is not ready');
  const stream = body.stream === true;
  const { id, requires, jsonMode } = requirementsFor(body);
  const primary = router.primaryFor(id);

  const tried = new Set<string>();
  let attempts = 0;
  const tryCandidate = async (candidate: Candidate): Promise<Attempt | null> => {
    if (router === null) return null;
    if (router.remainingOf(candidate.providerId) <= 0) return null;
    router.noteCall(candidate.providerId);
    attempts += 1;
    const attempt = await callOnce(ENV_URL, candidate, api, body, stream);
    if (attempt.ok) {
      log(`${id}: ${candidate.providerId}/${candidate.modelId} after ${attempts} call(s)`);
      if (attempt.passthrough !== null) {
        res.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        });
        await pipeStream(attempt.passthrough, res);
      } else {
        answerClient(res, api, stream, attempt, candidate.modelId);
      }
      return attempt;
    }
    log(`${id}: ${candidate.providerId}/${candidate.modelId} refused HTTP ${attempt.status}`);
    return null;
  };

  // The plan's primary, when it still has budget. A refusal goes straight
  // to failover: retrying the same provider costs a call most of the time
  // the failure schedule has two adjacent numbers (the draws cluster), and
  // the failover order already prefers a provider not yet called, whose
  // first call always answers.
  if (primary !== undefined) {
    tried.add(primary.providerId);
    const served = await tryCandidate(primary);
    if (served !== null) return;
  }
  // Failover: the request's candidates best-first, spent providers out.
  const order = router.failoverOrder(requires, jsonMode, probes, brief.ratings, primary?.providerId ?? '');
  for (const candidate of order) {
    if (attempts >= 2 + MAX_FAILOVER_ATTEMPTS) break;
    if (tried.has(candidate.providerId)) continue;
    tried.add(candidate.providerId);
    const served = await tryCandidate(candidate);
    if (served !== null) return;
  }
  // No candidate served it. The full candidate list (ignoring the budget
  // caps) is the last resort: a call past the planned cap can still land
  // inside the provider's real budget.
  const lastResort = candidatesFor(requires, jsonMode, probes, brief.ratings).filter(
    (c) => !tried.has(c.providerId),
  );
  for (const candidate of lastResort.slice(0, 2)) {
    tried.add(candidate.providerId);
    router.noteCall(candidate.providerId);
    attempts += 1;
    const attempt = await callOnce(ENV_URL, candidate, api, body, stream);
    if (attempt.ok) {
      log(`${id}: ${candidate.providerId}/${candidate.modelId} (last resort) after ${attempts} call(s)`);
      if (attempt.passthrough !== null) {
        res.writeHead(200, {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        });
        await pipeStream(attempt.passthrough, res);
      } else {
        answerClient(res, api, stream, attempt, candidate.modelId);
      }
      return;
    }
    log(`${id}: ${candidate.providerId}/${candidate.modelId} (last resort) refused HTTP ${attempt.status}`);
  }
  sendError(res, api, 503, 'overloaded_error', 'no free model could serve this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
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

/** POST /register with the port and the one catalog; the env answers with
 *  the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
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
    // The same document the register body carries, byte for byte: print
    // the one built object and stop before any server starts. A process
    // whose stdout is not flushed loses the tail, so wait for it.
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
  // 8080 is the env's on one layout; the OS's ephemeral range never holds
  // it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels(probes, brief.ratings).length} free models`);
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
