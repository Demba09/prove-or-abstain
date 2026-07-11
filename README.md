# prove-or-abstain

![CI](https://github.com/Demba09/prove-or-abstain/actions/workflows/ci.yml/badge.svg)

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

The agent is a LangGraph state machine with seven nodes and one conditional
loop. When a dimension fails to localize the cause, the verifier routes back
to the hypothesizer to try the next candidate dimension; the loop is bounded
by the number of dimensions, so it always terminates.

| Node | Role |
|------|------|
| detector | compares each metric to its baseline, flags material moves |
| hypothesizer | selects the next dimension to test |
| investigator | decomposes the metric change along that dimension (`attribution.py`) |
| verifier | checks the decomposition against the gates (`gates.py`) |
| driller | after an ASSERT, re-decomposes within the winning segment to refine the cause |
| actuator | maps the verdict to a typed action: recommend, execute, or escalate |
| reporter | writes the conclusion and keeps the full audit trail |

Responsibilities are split strictly. All numbers come from pandas/numpy. The
LLM (Qwen via DashScope) does three things only: suggest the order in which
dimensions are tested, phrase the final report from figures that are already
computed, and — on ASSERT only — offer business hypotheses about the *why*,
explicitly labelled as unverified speculation. It never produces a number and
never decides a verdict, so the verdict is identical with or without it — a
deterministic mock (`QWEN_MOCK=1`) runs the same pipeline offline.

## Verification gates

`ASSERT` requires all four gates to pass. A failed gate produces an
`ABSTAIN` with the failing condition named in the response.

| Gate | Condition | Purpose |
|------|-----------|---------|
| material | \|ΔR\|/R₀ ≥ 2% | the move is large enough to matter |
| localized | top contribution share ≥ 0.55 | one segment actually dominates |
| significant | two-proportion z-test on the leading segment, p ≤ 0.01 | the leader's move is not sampling noise |
| clean | interaction share ≤ 0.50 | rate and mix effects are separable |

The significance gate is a real hypothesis test, not a magic sample-size
number: a perfectly concentrated move on a segment of 60 users abstains with
`p=0.55`, the same move on 6000 users asserts with `p<1e-5`. For sum metrics
(no per-unit variance to test against) it falls back to a minimum-sample
floor of 1000.

## Quickstart

Requires Python 3.11+ (the pinned pandas/numpy versions do not install on
older interpreters; developed and tested on 3.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export QWEN_MOCK=1        # run offline; omit if DASHSCOPE_API_KEY is set
pytest -q                 # math vs. oracle, gate verdicts, API behaviour

uvicorn api.app:app --reload
```

Open http://localhost:8000 for the demo page — run the built-in scenarios or
upload your own CSVs — or use the API directly:

```bash
curl -X POST localhost:8000/investigate \
  -H 'content-type: application/json' -d '{"panel":"clean"}'
```

## API

```
GET  /                     demo page
POST /investigate
  body:   { "panel": "clean" | "diffuse" | "mixshift" | "deep", "autopilot": false }
  return: { verdict, confidence, root_cause, gates, drilldown,
            action, report, speculations, trace }
POST /investigate/upload
  multipart: baseline=<csv>, current=<csv>, autopilot=<bool>, sum_metrics=<csv names>
  same return shape
POST /investigate/series
  multipart: series=<csv with a 'period' column>, window=<int, optional>
  last period vs. a rolling baseline pooled over the prior `window` periods
GET  /health
```

The endpoint builds the initial state, runs the graph, and serializes the
result; the service layer contains no analysis logic.

Four built-in panels cover the interesting outcomes:

- `clean` — one segment's conversion rate collapses while everything else is
  stable. The first dimension tried (`device`) fails to localize, the second
  (`segment`) succeeds: **ASSERT**, cause `segment=paid`.
- `diffuse` — every segment drops by the same amount. Same aggregate change
  as `clean`, but no dimension concentrates it: **ABSTAIN**.
- `mixshift` — population mix and rates shift at the same time, so rate and
  mix effects are entangled: **ABSTAIN**, for a different named reason than
  `diffuse`.
- `deep` — a single cell (`paid × mobile`) collapses: **ASSERT**
  `device=mobile`, then the driller narrows it to `segment=paid` within
  mobile.

After an ASSERT, the response also carries `speculations`: short business
hypotheses about the *why* (a campaign change? a payment incident?), written
by the LLM and explicitly labelled as unverified speculation — kept strictly
apart from the proven verdict.

With autopilot on (`"autopilot": true`), an ASSERT with confidence ≥ 0.70
returns an `EXECUTE` action instead of a recommendation. An ABSTAIN never
executes, regardless of flags; the test suite enforces this.

### Bring your own data

`POST /investigate/upload` takes two CSVs in long panel format — one row per
(metric, segment...) cell, with raw counts:

```
metric, <dim1>, [<dim2>, ...], n, c
```

`n` is the cell's population, `c` the numerator (conversions, churned users,
…). Dimension columns are inferred: everything except `metric`, `n`, `c` and
`period`. Sample files live in `examples/`:

```bash
curl -X POST localhost:8000/investigate/upload \
  -F baseline=@examples/baseline.csv \
  -F current=@examples/current_clean.csv

# sum metric (revenue): n = customers, c = total amount
curl -X POST localhost:8000/investigate/upload \
  -F baseline=@examples/revenue_baseline.csv \
  -F current=@examples/revenue_current.csv \
  -F sum_metrics=revenue

# time series: one CSV with a 'period' column, rolling pooled baseline
curl -X POST localhost:8000/investigate/series \
  -F series=@examples/series_clean.csv -F window=4
```

Metrics named in `sum_metrics` are decomposed as sums (volume/rate split,
e.g. revenue = customers × average basket) instead of rates.

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

Sum metrics (`decompose_sum`) use the same algebra with raw counts instead
of shares: ΔV splits per segment into a volume effect (`r₀·Δn`), a rate
effect (`n₀·Δr`) and their interaction, summing to `c₁ − c₀` exactly.

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

- No live data connectors yet — data comes in as CSV uploads or the built-in
  panels.
- The rolling baseline pools prior periods; there is no seasonality or trend
  modelling.
- Drill-down goes one level deep (winning segment × one other dimension).
- Actions are typed objects returned by the API; nothing is wired to real
  downstream systems.

## License

MIT.
