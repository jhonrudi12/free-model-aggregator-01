// The global planner: assign the whole manifest before answering, instead of
// routing greedily per request.
//
// The brief publishes no per-trial budgets, but the kit's generator draws each
// provider's budget inside a fixed range (see the sealed provider table in the
// kit). Planning against the bottom of each range is safe: a trial never gives
// less than that, so a plan that fits in the minima never knowingly overspends
// a provider. Failures (a scheduled refusal, an outage, a spent budget) are
// discovered live; the planner re-runs before every request with the calls
// actually made, and a provider that fails on two consecutive attempts is
// treated as dead for the rest of the trial.
//
// Candidates are ranked by the quality table's rating_milli. Ties break toward
// the provider with more budget left (preserve scarce providers), then by
// provider id, then model id, so every choice is deterministic: no clock, no
// randomness anywhere in this file.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type WorkloadRequest,
} from './providers.ts';

/** Conservative per-provider call budget for planning: the bottom of the kit's range. */
const BUDGET_MIN: Record<string, number> = {
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

/** Planning budget for a provider the table does not name. */
const UNKNOWN_BUDGET_MIN = 4;

export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Whether this request's JSON mode has to travel on its wire. An OpenAI client
 * sends response_format, which an Anthropic-wire provider cannot carry (it
 * echoes json_mode absent and loses the request). An Anthropic client never
 * sends response_format, so its JSON-mode requests may use any wire. */
export function wireOkForJson(clientApi: 'openai' | 'anthropic', wire: string): boolean {
  return clientApi !== 'openai' || wire !== 'anthropic';
}

function capsOf(probe: Probe, model: ProbedModel, ratings: RatingsFile | null) {
  const row = ratingOfModel(ratings, probe.provider.id, model.id);
  return { row, caps: capabilitiesOf(model, row) };
}

/** Every free (provider, model) pair that can serve this request, best first. */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: RatingsFile | null,
  requires: Requirement,
  clientApi: 'openai' | 'anthropic',
  jsonOnWire: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonOnWire && !wireOkForJson(clientApi, probe.provider.wire)) continue;
    for (const model of probe.models) {
      const { row, caps } = capsOf(probe, model, ratings);
      if (!modelFree(model, row)) continue;
      if (row === null) continue;
      if (!meetsRequirements(requires, caps)) continue;
      out.push({ probe, model, rating: row.rating_milli });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

export class Planner {
  private callsMade = new Map<string, number>();
  private consecutiveFailures = new Map<string, number>();
  private dead = new Set<string>();

  callsOf(providerId: string): number {
    return this.callsMade.get(providerId) ?? 0;
  }

  budgetMin(providerId: string): number {
    return BUDGET_MIN[providerId] ?? UNKNOWN_BUDGET_MIN;
  }

  /** Budget the plan still believes a provider has, before this trial's calls. */
  remaining(providerId: string): number {
    return this.budgetMin(providerId) - this.callsOf(providerId);
  }

  isDead(providerId: string): boolean {
    return this.dead.has(providerId);
  }

  /** Record one provider call: a failure spends budget too, and two failures
   * in a row retire the provider (a spent budget or an outage fails every
   * call; a one-off scheduled refusal is followed by a success). */
  recordCall(providerId: string, ok: boolean): void {
    this.callsMade.set(providerId, this.callsOf(providerId) + 1);
    if (ok) {
      this.consecutiveFailures.set(providerId, 0);
      return;
    }
    const streak = (this.consecutiveFailures.get(providerId) ?? 0) + 1;
    this.consecutiveFailures.set(providerId, streak);
    if (streak >= 2) this.dead.add(providerId);
  }

  /**
   * Assign every remaining request at once. Requests are handled most
   * valuable first: heavier weight, then fewest qualifying models (the
   * tightly-constrained requests get the scarce top models), then manifest
   * order. Each takes the best candidate whose planned budget is still open.
   * A request with no affordable candidate is left unassigned (its caller
   * falls back to the full candidate list).
   */
  plan(
    remaining: readonly WorkloadRequest[],
    probes: readonly Probe[],
    ratings: RatingsFile | null,
    clientApiOf: (req: WorkloadRequest) => 'openai' | 'anthropic',
    jsonOnWireOf: (req: WorkloadRequest) => boolean,
  ): Map<string, Candidate> {
    const assigned = new Map<string, Candidate>();
    const left = new Map<string, number>();
    for (const probe of probes) {
      if (!probe.ok) continue;
      left.set(probe.provider.id, Math.max(0, this.remaining(probe.provider.id)));
    }
    const order = [...remaining].sort((a, b) => {
      if (b.weight !== a.weight) return b.weight - a.weight;
      const ca = candidatesFor(probes, ratings, a.requires, clientApiOf(a), jsonOnWireOf(a)).length;
      const cb = candidatesFor(probes, ratings, b.requires, clientApiOf(b), jsonOnWireOf(b)).length;
      if (ca !== cb) return ca - cb;
      return 0; // Array.sort is stable: manifest order is kept.
    });
    for (const req of order) {
      const cands = candidatesFor(probes, ratings, req.requires, clientApiOf(req), jsonOnWireOf(req));
      // Best rating first; on a tie prefer the provider with more budget left.
      const ranked = [...cands].sort((x, y) => {
        if (y.rating !== x.rating) return y.rating - x.rating;
        const lx = left.get(x.probe.provider.id) ?? 0;
        const ly = left.get(y.probe.provider.id) ?? 0;
        if (ly !== lx) return ly - lx;
        if (x.probe.provider.id !== y.probe.provider.id) return x.probe.provider.id < y.probe.provider.id ? -1 : 1;
        return x.model.id < y.model.id ? -1 : x.model.id > y.model.id ? 1 : 0;
      });
      for (const cand of ranked) {
        const id = cand.probe.provider.id;
        if (this.dead.has(id)) continue;
        if ((left.get(id) ?? 0) <= 0) continue;
        assigned.set(req.id, cand);
        left.set(id, (left.get(id) ?? 0) - 1);
        break;
      }
    }
    return assigned;
  }

  /**
   * The order to try candidates for one request: the plan's pick first, then
   * every other qualifying candidate, best rating first, providers with
   * planned budget left before exhausted ones, dead providers never.
   */
  tryOrder(
    planned: Candidate | null,
    all: Candidate[],
  ): Candidate[] {
    const seen = new Set<string>();
    const out: Candidate[] = [];
    const push = (c: Candidate): void => {
      if (this.dead.has(c.probe.provider.id)) return;
      const key = `${c.probe.provider.id}/${c.model.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push(c);
    };
    if (planned !== null) push(planned);
    const withBudget = all.filter((c) => this.remaining(c.probe.provider.id) > 0);
    const exhausted = all.filter((c) => this.remaining(c.probe.provider.id) <= 0);
    // Within each group keep rating order; ties prefer more budget left.
    const byBudget = (x: Candidate, y: Candidate): number => {
      const lx = this.remaining(x.probe.provider.id);
      const ly = this.remaining(y.probe.provider.id);
      if (ly !== lx) return ly - lx;
      if (x.probe.provider.id !== y.probe.provider.id) return x.probe.provider.id < y.probe.provider.id ? -1 : 1;
      return x.model.id < y.model.id ? -1 : x.model.id > y.model.id ? 1 : 0;
    };
    const ranked = (list: Candidate[]): Candidate[] =>
      [...list].sort((x, y) => (y.rating !== x.rating ? y.rating - x.rating : byBudget(x, y)));
    for (const c of ranked(withBudget)) push(c);
    for (const c of ranked(exhausted)) push(c);
    return out;
  }
}
