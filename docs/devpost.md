# Devpost write-up — prove-or-abstain

## Inspiration

Every "AI agent for analytics" demo we'd seen had the same flaw: point it at
a metric drop and it will *always* produce a confident-sounding explanation,
whether or not the data actually supports one. That's a liability the moment
you let an agent act autonomously — a wrong but confident diagnosis is worse
than no diagnosis. We wanted to build the opposite: an agent whose autonomy
is gated by proof, and whose only alternative to acting is a clean,
explained refusal. Not "I don't know" — a *specific*, math-backed reason it
can't localize a cause.

## What it does

`prove-or-abstain` investigates a metric anomaly (e.g. a conversion or churn
rate move) the way a careful analyst would: it forms a hypothesis about which
dimension explains the shift, tests it with an exact rate/mix/interaction
decomposition, and checks the result against four deterministic gates
(materiality, concentration, a two-proportion z-test on the leading segment,
mechanism cleanliness). If a dimension clears all four, it **ASSERTs** the
cause, drills one level down to refine it (e.g. `device=mobile`, narrowed to
`segment=paid` within mobile), and proposes a scoped action — which it can
*execute* autonomously if confidence is high and autopilot is enabled. If no
dimension clears the bar, it **ABSTAINs**, escalates to a human, and names
exactly which gate failed and why.

It works on rate metrics and sum metrics (revenue decomposes into volume ×
average basket), takes arbitrary CSV panels or a multi-period time series
with a rolling pooled baseline, and — after a proven verdict only — asks the
LLM for business hypotheses about the *why*, clearly labelled as unverified
speculation, never mixed with the proven numbers.

It's built as a bounded LangGraph loop (never an unbounded cycle — the agent
tries each candidate dimension once, in an LLM-suggested order, and stops),
wrapped in a thin FastAPI service, containerized, and deployable to Alibaba
Cloud Function Compute as-is.

## How we built it

- **The math first.** Before any agent logic existed, we built the exact
  rate/mix/interaction decomposition and validated it against a hand-derived
  oracle (`scripts/attribution_reference.py` / `scripts/gate_check.py`) on three calibrated
  scenarios: a clean localized cause, a diffuse uniform shift, and a mix
  shift where composition and rate move together. Zero residual, verified.
- **Gates before agent.** The ASSERT/ABSTAIN decision (`prove_or_abstain/gates.py`) was tuned
  and validated (`scripts/gate_check_gates.py`) against those same three scenarios
  *before* wiring the LangGraph loop around it — so the safety property was
  proven independently of the orchestration.
- **LangGraph for the bounded loop.** `detector → hypothesizer → investigator
  → verifier` with a conditional edge back to `hypothesizer` when a dimension
  doesn't localize, bounded by the number of candidate dimensions. This
  conditional cycle is the actual reason LangGraph earns its place here
  rather than a linear chain.
- **A hard LLM boundary.** The LLM (Qwen via DashScope) does exactly two
  things: order which dimension to try first, and write the final sentence.
  It never computes a number and never decides a verdict — every number in
  the output comes from pandas/numpy. A mock mode (`QWEN_MOCK=1`) makes the
  whole pipeline deterministic and runnable offline for the demo.
- **Ship as one container.** A `python:3.12-slim` Dockerfile, `.env` never
  baked into the image, secrets injected at runtime — verified end-to-end
  with a live container run (`/health`, both demo panels) before writing a
  single line of deployment docs.

## Challenges we ran into

- **Making "diffuse" actually hard.** A naive uniform shift is *too* easy to
  reject — every gate fails cleanly. We added a third scenario, `mixshift`,
  where composition and rate move simultaneously so the interaction term is
  non-trivial. That's what actually stress-tests the "clean mechanism" gate
  rather than just the "localized" one.
- **Keeping the LLM boundary honest.** It's tempting to let the model
  "helpfully" adjust a number or nudge a verdict in its prose. We drew the
  line at ordering + rephrasing only, and the mock-mode fallback means the
  verdict is provably identical whether the LLM is in the loop or not.
- **Bounding the loop without hardcoding a scenario.** The loop bound is
  `len(dims)`, not a magic constant — so it generalizes to any number of
  candidate dimensions without needing a rewrite.

## Accomplishments we're proud of

- A decomposition with **zero residual**, verified against an independent
  oracle — the numbers are not approximate.
- An abstention that is a **safety property enforced in code**: `ABSTAIN`
  cannot produce `EXECUTE`, full stop, not a convention that's easy to
  violate.
- A **principled significance gate**: the same perfectly-concentrated move
  asserts on 6,000 users (p < 1e-5) and abstains on 60 (p = 0.55) — a real
  two-proportion z-test, not a magic sample-size threshold.
- **Drill-down**: after proving `device=mobile`, the agent re-decomposes
  within mobile and narrows the cause to `segment=paid` — or states
  explicitly that the whole segment is affected.
- Calibrated scenarios that fail for *different*, nameable reasons (no
  anomaly / diffuse / entangled mix / not significant), not just "on" and
  "off".
- A container that actually runs — build, `/health`, both verdict paths,
  all verified live, not just claimed.

## What's next

Out of MVP scope for the hackathon, but the direction is clear: real action
targets behind the actuator (Slack, Jira, Stripe, ad platforms), a **closed
control loop** that observes the effect of an executed action and
re-investigates, live data connectors (warehouse, Stripe, GA) instead of CSV
uploads, seasonality-aware baselines, real-time monitoring with alerting,
and memory of past investigations so recurring causes are recognized faster.

## Built with

`python` · `fastapi` · `langgraph` · `pandas` · `numpy` · `pydantic` ·
`uvicorn` · `docker` · Qwen (`qwen-plus`) via Alibaba Cloud DashScope ·
Alibaba Cloud Container Registry + Function Compute
