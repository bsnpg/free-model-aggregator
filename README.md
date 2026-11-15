# Free Model Aggregator

One server over ten free-tier model providers, speaking the OpenAI Chat
Completions API and the Anthropic Messages API, in JSON and in server-sent
events. Built for the make.fun competition
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01).

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/bsnpg/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/bsnpg/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
Referral code: `26zqsqqz`

## What it does

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory,
  the sealed quality table and the workload manifest.
- Probes each provider's catalog route in its own wire (OpenAI, Anthropic
  or Gemini) and builds one catalog — the document `node main.ts --list`
  prints and the register body carries, byte for byte.
- Plans each request's model before the trial starts: the best free model
  meeting its requirements (tools, vision, JSON mode, context length),
  under each provider's per-trial budget (planned against the bottom of
  each budget range, so a planned call never meets a spent budget).
- Serves `GET /v1/models` (OpenAI shape for `Authorization: Bearer`,
  Anthropic shape for `x-api-key`), `POST /v1/chat/completions` and
  `POST /v1/messages`, in JSON and in SSE, translating between all three
  upstream wires while preserving the request's system prompt, tools,
  images, `max_tokens`, stop sequences and JSON mode exactly — the
  provider's answer echo must match the client's request.
- Fails over on a 402/429/5xx: one retry on the same provider, then the
  request's candidates best-first. Exits 0 when the env says the workload
  is done.

## Run

```sh
node main.ts --list        # print the catalog and exit 0
node main.ts               # serve (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
```

Node 22+, standard library only — no dependencies, no wall clock, no
unseeded randomness, so a replay leaves the same state export as the
live pass.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, `/work` poll.
- `lib/brief.ts` — the brief's three JSON documents.
- `lib/catalog.ts` — catalog probes and the one catalog document.
- `lib/plan.ts` — quality-ranked, budget-aware routing and failover order.
- `lib/translate.ts` — the three wires: request/response translation.
- `lib/upstream.ts` — one upstream attempt against one provider.
