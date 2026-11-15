// The catalog: probing each provider's own catalog route, and the one
// document `--list` prints and the register body carries.
//
// The probe speaks each provider's wire:
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails leaves its provider out of the catalog; the trial
// goes on with the providers that answered. The catalog's base_url is the
// directory's own path (`/p/<id>`), never the path joined to the env URL:
// the env's origin differs between a live pass and its replay, and the
// catalog is part of the state export a replay must reproduce byte for byte.
//
// The quality table is the sealed truth where it has a row: a model's free
// flag and context length come from the table first, the listing second.

import { authHeaders, isRecord, joinBase, ratingRowOf } from './brief.ts';
import type { DirectoryProvider, JsonObject, RatingRow, Wire } from './brief.ts';

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's probe result. */
export interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

function textOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** Whether a model is free: the quality table's row where it has one (the
 *  sealed truth), the listing's own word where it does not, paid when
 *  neither says. A paid model is never routed. */
export function isFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the
 *  listing's own fields where it does not. */
export function capabilitiesOf(
  model: ListedModel,
  row: RatingRow | null,
): { contextLength: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? textOf(v.name).replace(/^models\//, '') : textOf(v.id);
  if (id === '') return null;
  const context =
    numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    displayName:
      textOf(v.display_name) !== ''
        ? textOf(v.display_name)
        : textOf(v.displayName) !== ''
          ? textOf(v.displayName)
          : id,
    free: boolOrNull(v.free),
    contextLength: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, tolerating every failure. */
export async function probeProvider(
  envUrl: string,
  provider: DirectoryProvider,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.path);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** The catalog the probes and the quality table make: every provider that
 *  answered, every model it listed, models sorted by id within a provider. */
export function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFree(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.path,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free models across the providers that answered, in the brief's
 *  provider order: what GET /v1/models lists. */
export function servedModels(probes: readonly Probe[], ratings: RatingRow[]): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}

/** A fixed epoch second for every listed model's `created`: nothing in the
 *  reply reads the clock, so a replay prints the same document. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`, for the same reason. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the OpenAI list shape. */
export function openAIModelList(models: readonly ServedModel[]): JsonObject {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: the Anthropic list shape. */
export function anthropicModelList(models: readonly ServedModel[]): JsonObject {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}
