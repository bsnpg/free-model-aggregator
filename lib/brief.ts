// The trial brief: the provider directory, the quality table and the workload
// manifest, all carried as JSON documents inside the prose of the file at
// $MAKEFUN_PROMPT_FILE. This module finds each document by its `schema`
// member and turns it into typed rows the rest of the server works from.
//
// Nothing here reads the clock or rolls dice: the same brief always parses
// to the same rows.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const DIRECTORY_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type JsonObject = Record<string, unknown>;
export const isRecord = (v: unknown): v is JsonObject =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider row of the brief's directory. `path` is the base path the
 *  entry joins to $MAKEFUN_ENV_URL (for example `/p/groq`); `key` is the
 *  stand-in key the simulated service accepts. */
export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  path: string;
  keyEnv: string;
  key: string;
}

/** One row of the sealed quality table. */
export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one workload request needs from the model that serves it. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One row of the workload manifest: the requests arrive in this order. */
export interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  best: { id: string; ratingMilli: number }[];
  workload: WorkloadRow[];
  error: string | null;
}

const EMPTY: Brief = { providers: [], ratings: [], best: [], workload: [], error: null };

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBool(v: unknown): boolean {
  return v === true;
}

/** The index of the `}` that closes the object opening at `start`, or -1.
 *  String contents (and their escapes) are skipped so braces inside a
 *  quoted value never count. */
function closingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function schemaDocuments(text: string): JsonObject[] {
  const found: JsonObject[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closingBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // A `{` that opens prose rather than JSON: look on.
    }
  }
  return found;
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme {
  if (v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none') return v;
  return 'bearer';
}

function providerOf(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = wireOf(v.wire);
  if (id === '' || wire === null) return null;
  // The directory spells the base as `path` (/p/<id>); accept the other
  // spellings a directory might use, `path` first.
  const path =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth: authOf(v.auth),
    path,
    keyEnv: asString(v.key_env),
    key: asString(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: asNumberOrNull(v.rating_milli) ?? 0,
    contextLength: asNumberOrNull(v.context_length),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: asBool(v.free),
  };
}

function requirementOf(v: unknown): Requirement {
  if (!isRecord(v)) return {};
  const out: Requirement = {};
  if (v.tools !== undefined) out.tools = asBool(v.tools);
  if (v.vision !== undefined) out.vision = asBool(v.vision);
  if (v.json !== undefined) out.json = asBool(v.json);
  const minContext = asNumberOrNull(v.min_context);
  if (minContext !== null) out.minContext = minContext;
  return out;
}

function workloadOf(v: unknown): WorkloadRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: requirementOf(v.requires),
  };
}

/** The brief at `path`. A missing or unreadable brief yields an empty one
 *  with `error` set, so `--list` can still print `{"providers":[]}`. */
export function readBrief(path: string): Brief {
  if (path === '') return { ...EMPTY, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...EMPTY, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const directory = docs.find((d) => d.schema === DIRECTORY_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined || !Array.isArray(directory.providers)) {
    return { ...EMPTY, error: `the brief at ${path} holds no ${DIRECTORY_SCHEMA} document` };
  }
  const providers = directory.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined || !Array.isArray(ratingsDoc.ratings)
      ? []
      : ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best =
    ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [{ id: asString(v.id), ratingMilli: asNumberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const workload =
    workloadDoc === undefined || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, error: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingRowOf(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** A base path joined to the env's URL; an absolute base is used as it is. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (b === '') return root;
  return `${root}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key, in its own scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}
