# prove-or-abstain

A causal investigation agent for product metrics. Given a baseline period and
a current period, it determines whether a metric moved materially, tests which
segment explains the move, and returns one of two verdicts:

- **ASSERT** — the cause is localized and quantified, and the conclusion
  passes every verification gate. The agent recommends a scoped action, or
  executes it directly when autopilot is enabled and confidence is high.
- **ABSTAIN** — the evidence does not single out a cause. The agent escalates
  to a human and states which gate failed and why.

The second verdict is the point of the design. An agent that is allowed to
act on data needs a principled way to refuse to act when the data does not
support a conclusion — otherwise it will always produce a plausible-sounding
diagnosis, right or wrong.

![architecture](docs/architecture.svg)

## How it works

The agent is a LangGraph state machine with six nodes and one conditional
loop. When a dimension fails to localize the cause, the verifier routes back
to the hypothesizer to try the next candidate dimension; the loop is bounded
by the number of dimensions, so it always terminates.

| Node | Role |
|------|------|
| detector | compares each metric to its baseline, flags material moves |
| hypothesizer | selects the next dimension to test |
| investigator | decomposes the metric change along that dimension (`attribution.py`) |
| verifier | checks the decomposition against the gates (`gates.py`) |
| actuator | maps the verdict to a typed action: recommend, execute, or escalate |
| reporter | writes the conclusion and keeps the full audit trail |

Responsibilities are split strictly. All numbers come from pandas/numpy. The
LLM (Qwen via DashScope) does two things only: suggest the order in which
dimensions are tested, and phrase the final report from figures that are
already computed. It never produces a number and never decides a verdict, so
the outcome is identical with or without it — a deterministic mock
(`QWEN_MOCK=1`) runs the same pipeline offline.

## Verification gates

`ASSERT` requires all four gates to pass. A failed gate produces an
`ABSTAIN` with the failing condition named in the response.

| Gate | Condition | Purpose |
|------|-----------|---------|
| material | \|ΔR\|/R₀ ≥ 2% | the move is large enough to matter |
| localized | top contribution share ≥ 0.55 | one segment actually dominates |
| sample | leading segment n ≥ 1000 | the leader is not a small-sample artifact |
| clean | interaction share ≤ 0.50 | rate and mix effects are separable |

## Quickstart

Requires Python 3.11+ (the pinned pandas/numpy versions do not install on
older interpreters; developed and tested on 3.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export QWEN_MOCK=1        # run offline; omit if DASHSCOPE_API_KEY is set
pytest -q                 # math vs. oracle, gate verdicts, API behaviour

uvicorn api.app:app --reload
curl -X POST localhost:8000/investigate \
  -H 'content-type: application/json' -d '{"panel":"clean"}'
```

## API

```
POST /investigate
  body:   { "panel": "clean" | "diffuse" | "mixshift", "autopilot": false }
  return: { verdict, confidence, root_cause, gates, action, report, trace }

GET /health
```

The endpoint builds the initial state, runs the graph, and serializes the
result; the service layer contains no analysis logic.

Three built-in panels cover the three interesting outcomes:

- `clean` — one segment's conversion rate collapses while everything else is
  stable. The first dimension tried (`device`) fails to localize, the second
  (`segment`) succeeds: **ASSERT**, cause `segment=paid`.
- `diffuse` — every segment drops by the same amount. Same aggregate change
  as `clean`, but no dimension concentrates it: **ABSTAIN**.
- `mixshift` — population mix and rates shift at the same time, so rate and
  mix effects are entangled: **ABSTAIN**, for a different named reason than
  `diffuse`.

With autopilot on (`"autopilot": true`), an ASSERT with confidence ≥ 0.70
returns an `EXECUTE` action instead of a recommendation. An ABSTAIN never
executes, regardless of flags; the test suite enforces this.

## Attribution math

For a rate metric `R = Σ wₛ·rₛ` (segment weight × segment rate), the change
between periods decomposes exactly, per segment:

```
rate        = w₀·(r₁ − r₀)        the segment's rate moved
mix         = r₀·(w₁ − w₀)        the population composition moved
interaction = (w₁ − w₀)·(r₁ − r₀)
contribution = rate + mix + interaction
```

Segment contributions sum to the total ΔR with zero residual.
`attribution.py` is validated against an independently written oracle
(`attribution_reference.py`) in the test suite and in `gate_check.py`.

## Docker

```bash
docker build -t prove-or-abstain .
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=... prove-or-abstain
```

Secrets are injected at runtime; `.env` is excluded from the image. Without
a key the service falls back to mock mode. For Alibaba Cloud, push an
amd64 image to Container Registry and run it on Function Compute
(custom-container runtime, port 8000, HTTP trigger) — `/health` serves as
the probe endpoint.

```bash
docker buildx build --platform linux/amd64 \
  -t registry.<region>.aliyuncs.com/<namespace>/prove-or-abstain:v1 --push .
```

## Limitations

- Demo data is in-memory (`panels.py`); there is no ingestion layer yet.
- Two-period comparison only (baseline vs. current), rate metrics only.
- Actions are typed objects returned by the API; nothing is wired to real
  downstream systems.

## License

MIT.
