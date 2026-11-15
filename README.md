# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/jhonrudi12/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/jhonrudi12/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten free-tier providers, in JSON and in server-sent events,
for the make.fun bounty **free-model-aggregator-01** ("The best free AI model
aggregator").

Node 22, TypeScript, standard library only — nothing is installed at run time.

## Layout

- `main.ts` — entrypoint: reads the brief, probes the providers, builds the
  one catalog, registers with the env, serves the client, polls `/work` and
  exits 0 when the workload is done. `node main.ts --list` prints the catalog
  and exits 0.
- `lib/providers.ts` — the brief's three documents (provider directory,
  quality table, workload manifest) and the per-wire catalog probe.
- `lib/plan.ts` — the global planner (see below).
- `lib/wire.ts` — the client wires and the three upstream wires, through one
  neutral request shape, plus the normalized answer and the client emitters.
- `dev/verify-gates.mjs` — local verifier: `--list` checks, catalog-shape
  checks against the practice references, all three practice trials, and a
  determinism re-run. Run it with `node dev/verify-gates.mjs`.
- `LICENSE` — MIT.

## How it works

**Catalog.** At startup the server probes every provider's catalog route in
that provider's own wire and auth scheme, and builds one catalog document
from what the listings say plus the quality table's free flag and context
length. The same object is printed by `--list` and posted in the register
body, byte for byte. `GET /v1/models` lists exactly the free model ids in
that catalog, in OpenAI shape for `Authorization: Bearer` and Anthropic shape
for `x-api-key`.

**Global planner (quality).** The manifest lists all twenty requests up
front, so the server plans globally instead of routing greedily per request.
Before each request it re-plans the requests still to come with the calls
actually made so far: heaviest weight first, then fewest qualifying models
(the tightly-constrained requests get the scarce top models), then manifest
order. Each request takes the best free model meeting its requirements whose
planned budget is still open. Budgets are planned against the bottom of each
provider's range from the kit, so a plan never knowingly overspends a
provider; a provider that fails on two consecutive attempts is retired for
the rest of the trial. A paid model is never a legal pick.

**Failover.** On any failed provider call the same request is retried on the
next-best qualifying free model, in rating order, until one serves or the
candidates run out. A failed call spends budget, exactly as the env counts
it, and counts toward retiring its provider.

**Wires.** Every request is translated into the chosen provider's wire
(openai, anthropic, gemini) through one neutral shape, so the system prompt,
user text, images, tools, stop sequences, `max_tokens` and JSON mode travel
untouched. A JSON-mode request from an OpenAI client never goes to an
Anthropic-wire provider (that wire cannot carry `response_format`). Every
upstream answer, JSON or SSE, is parsed into one normalized answer and
re-emitted in the client's wire, in that wire's canonical event order, with
the provider's usage counts and the stop reason mapped (`end_turn`/`STOP`
→ `stop`, `max_tokens` → `length`, `tool_use` → `tool_calls`,
`refusal` → `content_filter`).

**Determinism.** No clock and no unseeded randomness anywhere in the request
path: no `Date.now()`, no `Math.random()`. The port is the only thing the OS
picks, and it never reaches a response. Replaying a trial leaves the same
state export byte for byte.

## Running it

```sh
node main.ts            # trial mode: needs MAKEFUN_PROMPT_FILE + MAKEFUN_ENV_URL
node main.ts --list     # print the catalog and exit 0
node dev/verify-gates.mjs   # run the local gates (3 practice trials + determinism)
```

## Local results (practice trials, kit scorer)

| trial | score | quality | served | wire | failover | catalog | calls | working |
|-------|-------|---------|--------|------|----------|---------|-------|---------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 25    | 1       |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 28    | 1       |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 30    | 1       |

Quality sits at 99 rather than 100 because the env schedules a refusal on
specific call numbers of each provider; any first-choice call that lands on
one fails over to the next-best model (a few milli lower). The determinism
re-run of trial 0 leaves a byte-identical state export.
