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
average basket), and accepts data four different ways: an arbitrary CSV
panel, a live SQL query, a Google Sheet, or a multi-period time series with a
rolling pooled baseline. A fifth mode, **"Watch a source"**, needs no
baseline file at all — send one observation at a time, tagged with a source
id, and it compares each new one against everything already persisted for
that source (SQLite-backed, survives a restart). If the columns you send
don't already match our schema, Qwen maps them — the *one* place in the
whole system where its judgment can actually change the outcome, everywhere
else it only orders, routes, or phrases.

A continuous autopilot (`monitor.py`) watches a set of sources on a timer,
investigates on movement, and fires a webhook (Slack/Discord/Teams
auto-detected) on a confident `ASSERT` — with every execution logged to an
audit trail a human can resolve. A free-text endpoint lets Qwen route plain
English ("why did conversion drop?") to the right investigation. And an MCP
server exposes the whole pipeline as tools a Qwen Cloud agent can call
directly, so Qwen can be the outer orchestrator instead of just a component.

It's built as a bounded LangGraph loop (never an unbounded cycle — the agent
tries each candidate dimension once, in an LLM-suggested order, and stops),
wrapped in a thin FastAPI service, containerized, and deployable to Alibaba
Cloud Function Compute as-is.

## How we built it

- **The math first.** Before any agent logic existed, we built the exact
  rate/mix/interaction decomposition and validated it against a hand-derived
  oracle (`scripts/attribution_reference.py` / `scripts/gate_check.py`) on
  three calibrated scenarios: a clean localized cause, a diffuse uniform
  shift, and a mix shift where composition and rate move together. Zero
  residual, verified.
- **Gates before agent.** The ASSERT/ABSTAIN decision (`prove_or_abstain/gates.py`)
  was tuned and validated (`scripts/gate_check_gates.py`) against those same
  three scenarios *before* wiring the LangGraph loop around it — so the
  safety property was proven independently of the orchestration.
- **LangGraph for the bounded loop.** `detector → hypothesizer → investigator
  → verifier` with a conditional edge back to `hypothesizer` when a dimension
  doesn't localize, bounded by the number of candidate dimensions. This
  conditional cycle is the actual reason LangGraph earns its place here
  rather than a linear chain. A second orchestration mode lets Qwen itself
  drive the loop via tool calls instead of the fixed graph — a determinism
  guard still checks every untested dimension before concluding, so the two
  modes are provably identical on the ASSERT/ABSTAIN verdict. One nuance,
  found by actually running it live rather than trusting the mock: when a
  single narrow cell collapses, it concentrates 100% on both of its
  defining dimensions at once, so Qwen's test order can decide which one is
  the headline cause vs. the drill-down refinement — the full diagnosis is
  recovered either way, never lost or invented.
- **A benchmark that can't lie to itself.** 30 synthetic scenarios with
  ground truth written from how each panel is *generated*, never from
  running the pipeline — otherwise accuracy would be circular. 100%
  accuracy, 0% false-ASSERT, 0% false-ABSTAIN, identical across graph/agent
  mode. We're upfront in the README about what that number does and doesn't
  prove: it's a strong regression test against cases built clearly on one
  side of each gate's threshold, not evidence on genuinely ambiguous data —
  which is why three *real* public datasets (historical airline passenger
  counts, the Titanic manifest, US college-major employment stats) also go
  through the pipeline, reshaped but not invented, kept deliberately
  separate from the official benchmark number to avoid smuggling
  circularity back in. The college-majors one also has a genuinely raw,
  unrenamed-columns variant sent through "Watch a source" — the deterministic
  mock heuristic can't map it (proven by a test asserting it 400s), which is
  finally a fair, non-softball test of `map_schema()`'s judgment rather than
  the math.
- **Calibration, not just accuracy.** `calibrate_confidence()` in `benchmark.py`
  buckets ASSERT confidence and reports Expected Calibration Error — does a 0.7 confidence
  actually mean "right ~70% of the time"? (ECE ≈ 0.19, conservative:
  under-confident rather than over-confident, the safe direction for an
  agent that can act.)
- **A hard LLM boundary, audited not just asserted.** Qwen orders which
  dimension to try, phrases the report, classifies an unfamiliar metric as
  rate/sum, routes free-text questions, and — the one deliberate exception —
  maps a raw source's columns onto our schema, with a self-verification pass
  where it re-checks its own first answer before it's acted on. Everywhere
  else, `QWEN_MOCK=1` reproduces the identical verdict offline, and
  `audit.py` freezes any investigation into a SHA256-hashed, replayable
  trail so that claim is checkable, not just stated.
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
- **Keeping the LLM boundary honest while giving it real jobs.** It's
  tempting to let the model "helpfully" adjust a number or nudge a verdict.
  We drew the line at ordering, routing, and phrasing — until "Watch a
  source" needed to interpret genuinely unfamiliar raw columns, where there
  *is* no single mechanically-correct answer. Rather than fake determinism
  there, we made it the one explicit, documented exception, with Qwen
  self-verifying its own mapping before it's used and a deterministic
  validation pass as the backstop.
- **Not letting the persisted baseline get away from us.** An earlier
  version of continuous monitoring kept its "last known" baseline in an
  in-process dict — lost on restart, and a duplicate of the same
  materiality check the graph already does. Moving it to a real SQLite
  table with a pooled reference window (reusing the same summed-counts
  algebra as the time-series baseline) removed both problems at once.
- **Bounding the loop without hardcoding a scenario.** The loop bound is
  `len(dims)`, not a magic constant — so it generalizes to any number of
  candidate dimensions without needing a rewrite.
- **A single-cell anomaly is inherently ambiguous between its two axes, and
  only a live run against a real key surfaced it.** Mock mode never
  reorders dimensions, so it always tested `device` before `segment` and
  always reported the same headline cause — masking that a single-cell
  collapse mathematically concentrates 100% on *both* its defining
  dimensions, not just the one mock happened to try first. A live Qwen run
  ordered them differently and got marked "wrong" by our own benchmark,
  which only checked the top-level cause. We verified the drill-down
  recovers the other dimension either way (the diagnosis was never
  actually wrong, just labelled differently), fixed the benchmark to credit
  either field, and documented the nuance rather than quietly special-
  casing it away.

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
- **100% on a 30-scenario benchmark, 0% false-ASSERT and false-ABSTAIN**,
  plus three real external datasets that agree with well-documented history
  or reveal a genuine, un-planted pattern (the 1960 air-travel growth spike
  was systemic, not seasonal; Titanic survival localizes to sex, not the
  popular class explanation; the STEM employment gap concentrates in
  majority-women majors) — and we say plainly, in the README, what the
  synthetic 100% does and doesn't prove.
- A **continuous autopilot** that persists its own baseline, survives a
  restart, and never lets one broken data source corrupt another's history.
- A **fully offline, reproducible pipeline** (`QWEN_MOCK=1`) with a
  SHA256-hashed, replayable audit trail — and one honestly-documented
  exception to that guarantee, not a silently broken promise.
- A container that actually runs — build, `/health`, both verdict paths,
  all verified live, not just claimed.

## What's next

- OAuth-native connectors (Stripe, GA4, Amplitude) beyond the current
  DSN/shared-link model
- Real downstream actions wired behind the actuator (Slack alerts, feature
  flags, campaign pausing) and a **closed control loop** that observes the
  effect of an executed action and re-investigates
- Seasonality- and trend-aware baselines, and an adaptive (not just
  fixed-window) pooling strategy for long-running watched sources
- Deeper drill-down (currently one level: winning segment × one other
  dimension)
- `evidence.py`'s embedded operational-event table replaced by a real
  calendar/deploy-log/ticketing integration
- A genuinely multi-turn conversational front-end, beyond today's single
  filtered follow-up

## Built with

`python` · `fastapi` · `langgraph` · `pandas` · `numpy` · `pydantic` ·
`uvicorn` · `sqlite` · `sqlalchemy` · `docker` · Qwen (`qwen-plus`) via
Alibaba Cloud DashScope · MCP (Model Context Protocol) ·
Alibaba Cloud Container Registry + Function Compute
