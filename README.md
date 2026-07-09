# prove-or-abstain

An auto-verifying causal investigation agent for SaaS metrics. It detects an
anomaly in a metric (e.g. a conversion drop), generates falsifiable hypotheses
about the cause, **tests each one with contribution-attribution math**, gates
its own conclusion through a verifier, and ends on a **decision**: when the
evidence supports a cause it recommends a scoped action (and can execute it
under autopilot); **when the data cannot support any cause, it abstains and
hands the wheel back to a human** — instead of acting on an unproven diagnosis.

> An autopilot that can act must be able to refuse to act. This one **acts on
> proof and disengages otherwise** — the abstention is a safety property, not
> a gimmick.

![architecture](docs/architecture.svg)

## Why this is credible

The numbers come from `pandas`/`numpy`, **never** from the LLM. The model
orders which dimensions to investigate first and writes the final verdict in
plain language — but every figure is computed and auditable. Contributions
decompose exactly: `rate + mix + interaction` sums to the observed change in
the metric, with zero residual (`gate_check.py` proves this against a
hand-derived oracle).

## Architecture

```
   detector ──route_after_detect──┐
      │ "investigate"             │ "report" (no anomaly)
      ▼                           │
 hypothesizer ◄──────┐            │
      │              │ "loop"     │
      ▼              │            │
 investigator        │            │
      │              │            │
      ▼              │            │
   verifier ─route_after_verify───┤
      │ "actuate"                 │
      ▼                           │
   actuator                       │
      │                           │
      ▼                           ▼
   reporter ───────────────────► END
```

A LangGraph state machine with a **bounded** conditional loop: if a dimension
doesn't clear the gates, the agent tries the next candidate dimension, up to
`len(dims)` iterations — never an unbounded cycle. `graph.py` is the *only*
file that imports `langgraph`; every node in `nodes.py` is a plain function
`(state) -> dict` and can be chained by hand for testing.

| Node | Role | Numbers from |
|------|------|--------------|
| detector | flag anomalies whose relative move clears materiality | pandas |
| hypothesizer | pick the next dimension to test (LLM orders, never invents) | deterministic queue + LLM ordering |
| investigator | decompose the metric's move on that dimension | pandas/numpy (`attribution.py`) |
| verifier | run 4 gates -> ASSERT this dimension, or move on | deterministic thresholds (`gates.py`) |
| actuator | map verdict + confidence -> a typed `Action` | deterministic mapping |
| reporter | sourced, plain-language conclusion + audit trail | LLM rephrases pre-computed numbers |

**Autonomy.** By default the actuator only *proposes* an action
(`RECOMMEND`/`ESCALATE`). It auto-executes (`EXECUTE`) only when
`autopilot_enabled=True` **and** confidence `>= 0.70`. `ABSTAIN` never
produces `EXECUTE` — this is enforced in code, not by convention.

## The verifier gates

`ASSERT` requires **all four** to pass; if any fails, the reason is named
(no silent guess):

| Gate | Threshold | What it catches |
|------|-----------|------------------|
| material | \|ΔR\|/R0 ≥ 2% | the anomaly is real, not noise |
| localized | concentration ≥ 0.55 | one segment dominates the contribution |
| sample | leading segment n ≥ 1000 | the "cause" isn't a statistical fluke |
| clean | interaction share ≤ 0.50 | mix and rate effects aren't hopelessly entangled |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# offline / no API key needed — LLM calls fall back to a deterministic mock
export QWEN_MOCK=1

# sanity-check the math and the gates against known-good oracles
python gate_check.py           # decomposition matches a hand-derived oracle
python gate_check_gates.py     # CLEAN->ASSERT, DIFFUSE/MIXSHIFT->ABSTAIN
pytest -q

# run the API
uvicorn api.app:app --reload
curl -X POST localhost:8000/investigate \
  -H 'content-type: application/json' -d '{"panel":"clean"}'
curl -X POST localhost:8000/investigate \
  -H 'content-type: application/json' -d '{"panel":"diffuse"}'
```

With a real key (`DASHSCOPE_API_KEY`, Qwen via DashScope), unset `QWEN_MOCK`
and the hypothesizer/reporter call the model instead of the deterministic
fallback — the verdict never changes either way, only the prose and the
dimension exploration order.

## API

```
POST /investigate
  body:   { "panel": "clean" | "diffuse" | "mixshift", "autopilot": bool = false }
  return: { panel, verdict, confidence, root_cause, gates, action, report, trace }

GET /health
  return: { "status": "ok" }
```

The endpoint is a thin wrapper: it builds the initial `AgentState`, calls
`graph.invoke(state)`, and serializes the result. No business logic lives in
FastAPI — everything is in the nodes.

## The three demo panels

In-memory, deterministic, calibrated to exercise the gates differently
(see `panels.py`):

- **`clean`** — the `paid` segment's conversion rate alone collapses
  (7.0% → 5.0%), everything else unchanged. Looks diffuse along `device`
  (paid splits ~50/50 mobile/desktop, so that dimension fails to localize) but
  localizes cleanly along `segment` → the agent **loops once, then ASSERTs**
  `segment=paid`.
- **`diffuse`** — every segment's rate drops by the same 0.6pp. Same
  aggregate ΔR as `clean`, but no dimension concentrates the contribution →
  the agent exhausts both dimensions and **abstains**.
- **`mixshift`** — composition *and* rates move at once (`organic` grows and
  its rate drops, `paid` shrinks), so `rate`, `mix`, and `interaction` are all
  non-zero. The interaction gate catches the entangled mechanism → **abstains**
  for a different, harder reason than `diffuse`.

## Attribution math (rate metrics)

For an aggregate rate `R = Σ wₛ·rₛ` (segment weight × segment rate):

```
ΔR = Σₛ (w₁ₛ·r₁ₛ − w₀ₛ·r₀ₛ)
  rate        = w₀ₛ·(r₁ₛ − r₀ₛ)     # the segment's rate moved
  mix         = r₀ₛ·(w₁ₛ − w₀ₛ)     # the composition moved
  interaction = (w₁ₛ − w₀ₛ)(r₁ₛ − r₀ₛ)
contribution_s = w₁ₛ·r₁ₛ − w₀ₛ·r₀ₛ    # sums to ΔR exactly, zero residual
```

See `attribution.py` (production) and `attribution_reference.py` (the oracle
`gate_check.py` diffs against).

## Docker

```bash
docker build -t prove-or-abstain .
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=... prove-or-abstain
# no key -> falls back to mock mode automatically
curl localhost:8000/health
```

`.env` is never copied into the image (`.dockerignore`); secrets are injected
at runtime — `docker run -e ...` locally, or the Alibaba Cloud environment
variable panel in production.

## Deployment (Alibaba Cloud)

Container Registry (ACR) + Function Compute, custom-container runtime — the
image runs as-is, no handler rewrite needed:

```bash
docker buildx build --platform linux/amd64 \
  -t registry.<region>.aliyuncs.com/<namespace>/prove-or-abstain:v1 --push .
```

Then in the FC console: new function → Custom Container runtime → point at
the ACR image → port `8000` → set `DASHSCOPE_API_KEY` → attach an HTTP
trigger. `/health` backs both the container `HEALTHCHECK` and FC's own probe.

## Status

Phases 0–8 complete: attribution core validated against an oracle (Phase 0),
gates tuned against `clean`/`diffuse`/`mixshift` (Phase 1), LangGraph loop
wired end-to-end, Qwen client with a deterministic mock fallback, FastAPI
surface, Dockerfile verified with a live container run. Phase 9 (this
material) covers the writeup.

## Roadmap (post-hackathon SaaS direction)

Out of MVP scope, signalled here as product vision, not built for the demo:
real action targets behind the actuator interface (Slack, Jira, Stripe, ad
platforms), the **full closed control loop** (act → observe the effect →
re-investigate), live connectors (Stripe, GA, warehouse), CSV/file upload for
arbitrary panels, real-time monitoring + alerting, multi-tenant, a polished
frontend, and memory of past investigations (pgvector).

## License

MIT.
