// Routing: which free model serves each request, under each provider's
// per-trial budget, and what to try when a provider refuses.
//
// The budgets are sealed per trial, drawn from a fixed range per provider;
// this router plans against the bottom of each range, so a planned call
// never meets a spent budget. What the plan leaves unused is deliberate:
// chasing the top of a range would trade a refused call (and a failover)
// for rating points the score's floor already gives away.
//
// The plan is built once from the workload manifest, before any request
// arrives: requests are assigned in order of decreasing weight (a demoted
// weight-2 request loses twice what a demoted weight-1 request loses),
// then of decreasing best rating, then of arrival order. At runtime the
// plan is only a first pick: a refusal goes straight to the request's
// candidate list in rating order, with providers not yet called this trial
// preferred (their first call always answers).

import { capabilitiesOf, isFree } from './catalog.ts';
import type { Probe } from './catalog.ts';
import { ratingRowOf } from './brief.ts';
import type { RatingRow, Requirement, WorkloadRow } from './brief.ts';

/** The bottom of each provider's per-trial budget range. A planned call
 *  never exceeds it, so a budget refusal cannot hit a planned call. */
export const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** The fallback cap for a provider the table above does not name. */
const UNKNOWN_PROVIDER_CAP = 4;

export function budgetCap(providerId: string): number {
  return MIN_BUDGET[providerId] ?? UNKNOWN_PROVIDER_CAP;
}

/** One routable model: a free model on a provider that answered. */
export interface Candidate {
  providerId: string;
  probe: Probe;
  modelId: string;
  ratingMilli: number;
}

/** Whether a model's capabilities meet one request's requirements. An
 *  unknown capability never meets one. */
export function meetsRequirements(
  requires: Requirement,
  caps: { contextLength: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null },
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) {
    return false;
  }
  return true;
}

/**
 * Every free model that can serve a request, best rating first. When the
 * request asks for JSON mode (`jsonModeNeeded`), anthropic-wire providers
 * are left out: their wire cannot carry the JSON-mode flag, so the echo
 * would come back saying it was absent.
 */
export function candidatesFor(
  requires: Requirement,
  jsonModeNeeded: boolean,
  probes: readonly Probe[],
  ratings: RatingRow[],
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonModeNeeded && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({
        providerId: probe.provider.id,
        probe,
        modelId: model.id,
        ratingMilli: row?.ratingMilli ?? 0,
      });
    }
  }
  // Best rating first; a tie goes to the smaller-budget provider, so the
  // scarce top-rated slots spread across providers instead of piling onto
  // one (a provider's failure schedule is per call number, so fewer calls
  // on one provider means fewer scheduled refusals to walk into), then to
  // the lower provider id for determinism.
  out.sort(
    (a, b) =>
      b.ratingMilli - a.ratingMilli ||
      budgetCap(a.providerId) - budgetCap(b.providerId) ||
      (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0),
  );
  return out;
}

function bestRatingOf(candidates: Candidate[]): number {
  return candidates.length === 0 ? 0 : candidates[0].ratingMilli;
}

/**
 * The trial's assignment: one primary (provider, model) per request id,
 * built from the manifest before the first request arrives. Requests are
 * assigned heaviest first so the scarce top-rated slots go to the requests
 * that lose most when demoted.
 */
export function buildPlan(
  manifest: WorkloadRow[],
  probes: readonly Probe[],
  ratings: RatingRow[],
): Map<string, Candidate> {
  const planned = new Map<string, number>();
  const assignment = new Map<string, Candidate>();
  // A request that needs JSON capability is planned off the anthropic wire
  // even when its body may not ask for JSON mode: the equal-rated
  // openai-wire substitute costs nothing, and the plan must not depend on
  // a body the router has not seen yet.
  const rows = manifest.map((row, index) => {
    const candidates = candidatesFor(row.requires, row.requires.json === true, probes, ratings);
    return { row, index, candidates };
  });
  const ordered = [...rows].sort(
    (a, b) =>
      b.row.weight - a.row.weight ||
      bestRatingOf(b.candidates) - bestRatingOf(a.candidates) ||
      a.index - b.index,
  );
  for (const { row, candidates } of ordered) {
    for (const candidate of candidates) {
      const used = planned.get(candidate.providerId) ?? 0;
      if (used >= budgetCap(candidate.providerId)) continue;
      planned.set(candidate.providerId, used + 1);
      assignment.set(row.id, candidate);
      break;
    }
  }
  return assignment;
}

/** The router's live state: how many calls each provider has actually
 *  taken this trial, planned or failover, successful or refused. */
export class Router {
  readonly assignment: Map<string, Candidate>;
  private readonly used = new Map<string, number>();

  constructor(manifest: WorkloadRow[], probes: readonly Probe[], ratings: RatingRow[]) {
    this.assignment = buildPlan(manifest, probes, ratings);
  }

  /** The planned primary for a request, when the plan found one. */
  primaryFor(requestId: string): Candidate | undefined {
    return this.assignment.get(requestId);
  }

  /** Calls a provider has taken so far. */
  usedOf(providerId: string): number {
    return this.used.get(providerId) ?? 0;
  }

  /** Planned budget still unspent for a provider. */
  remainingOf(providerId: string): number {
    return budgetCap(providerId) - this.usedOf(providerId);
  }

  /** Record one call (planned, retried or failover) against its provider. */
  noteCall(providerId: string): void {
    this.used.set(providerId, this.usedOf(providerId) + 1);
  }

  /**
   * The failover order for a request: its candidates best-first, skipping
   * providers whose planned budget is spent and the provider that just
   * refused (the caller retries that one separately, once). The list is
   * computed live so a provider spent by earlier failovers drops out.
   *
   * One adjustment: a provider already called this trial is marked down
   * 40 rating points. Its next call number can be on its failure schedule
   * — the draws cluster, so a provider that already refused once is a
   * worse bet than its rating says — while a provider not yet called
   * always answers its first call. An unused candidate up to 40 points
   * lower is the cheaper try; a gap wider than that is not worth the
   * certainty.
   */
  failoverOrder(
    requires: Requirement,
    jsonModeNeeded: boolean,
    probes: readonly Probe[],
    ratings: RatingRow[],
    excludeProviderId: string,
  ): Candidate[] {
    const USED_PENALTY = 40;
    return candidatesFor(requires, jsonModeNeeded, probes, ratings)
      .filter((c) => c.providerId !== excludeProviderId && this.remainingOf(c.providerId) > 0)
      .sort((a, b) => {
        const effectiveA = a.ratingMilli - (this.usedOf(a.providerId) > 0 ? USED_PENALTY : 0);
        const effectiveB = b.ratingMilli - (this.usedOf(b.providerId) > 0 ? USED_PENALTY : 0);
        return effectiveB - effectiveA || b.ratingMilli - a.ratingMilli || (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0);
      });
  }
}
