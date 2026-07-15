<!-- PROVE-OR-ABSTAIN -->
<p align="center">
  <img src="https://img.alicdn.com/imgextra/i3/O1CN01k1oD3R1Glhjj4hE2r_!!6000000000663-55-tps-158-28.svg" width="80" alt="logo" />
</p>

# prove-or-abstain

[![CI](https://github.com/Demba09/prove-or-abstain/actions/workflows/ci.yml/badge.svg)](https://github.com/Demba09/prove-or-abstain/actions)
[![Stars](https://img.shields.io/github/stars/Demba09/prove-or-abstain?style=social)](https://github.com/Demba09/prove-or-abstain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

<br>

> **Monday, 9am. Conversion is down 3.2%. The PM drops a message in Slack:**
> *"What happened? Campaign? Bug? New users behaving differently?"*
>
> The data team spends the next 4 hours slicing dashboards. They check segment by segment,
> device by device. Eventually — maybe — they find a culprit. Or maybe the drop is systemic
> and they're chasing noise.
>
> **Prove-or-Abstain answers in 2 seconds, with statistical proof — or refuses to guess.**

<br>

## What it does

Prove-or-Abstain is an AI agent that **investigates metric changes** across
your business segments and returns one of two verdicts:

| Verdict | Meaning | Example |
|---------|---------|---------|
| **ASSERT** | Cause found and statistically proven | *"'Paid' segment collapsed (p < 0.001). Recommend pausing the campaign targeting paid users."* |
| **ABSTAIN** | No single cause isolates — systemic or diffuse | *"Drop is real but equally distributed across all segments. Escalate to human — this is not a targeting issue."* |

**The ABSTAIN verdict is the innovation.** An agent allowed to act on data must
have a principled way to refuse to act when evidence is insufficient. Without it,
the agent will always fabricate a plausible-sounding diagnosis — right or wrong.

---

## Real-world walkthrough

A SaaS company with 200K monthly users tracks conversion across 4 segments
(organic, paid, referral, email) and 2 devices (mobile, desktop).

### Scenario 1 — A broken campaign

The "clean" panel simulates this: paid traffic's conversion dropped from 7%
to 5% overnight. Nothing else changed.

1. **Detector** flags conversion (|ΔR/R₀| = 3.1% ≥ 2%) — material
2. **Hypothesizer** (Qwen) suggests testing `device` first
3. **Investigator** decomposes the change along `device` → no single device dominates (concentration = 0.52 < 0.55) → **device: ABSTAIN**
4. Loop back to try the next dimension: `segment`
5. **Investigator** decomposes along `segment` → paid drives 87% of the drop (concentration = 0.87 ≥ 0.55)
6. **Verifier** runs the z-test: p < 0.001 — significant
7. **Driller** refines: within `segment=paid`, split is 50/50 mobile/desktop — nothing to narrow
8. **Actuator**: confidence 0.79 ≥ 0.70 → **ASSERT, EXECUTE** → "Pause the paid campaign"

### Scenario 2 — A systemic market drop

The "diffuse" panel: same aggregate drop, but every segment dropped equally (0.6 pp each).

1. **Detector** flags the anomaly
2. After testing both `device` and `segment`: **concentration never exceeds 0.55**
3. **Verifier** names the reason: "diffuse cause"
4. **Actuator**: **ABSTAIN, ESCALATE** → "Drop is real but not localized. This is likely a market-wide or seasonal effect — a human needs to investigate."

The verdict is **deterministic**: the LLM (Qwen) suggests which dimension to test first,
but the math decides everything. Run the same scenario with `QWEN_MOCK=1` and you get the same result.

---

## Try it yourself

```bash
git clone https://github.com/Demba09/prove-or-abstain
pip install -r requirements.txt
QWEN_MOCK=1 uvicorn api.app:app --reload
# Open http://localhost:8000
```

The demo page lets you run the 4 built-in scenarios (clean, diffuse, mixshift, deep),
ask questions in plain English, or plug in your own data — CSV upload, SQL database query,
Google Sheets.

![architecture](docs/architecture.svg)

## How it works

The agent is a LangGraph state machine with seven nodes and one conditional loop. When a dimension fails to localize the cause, the verifier routes back to the hypothesizer to try the next candidate dimension; the loop is bounded by the number of dimensions, so it always terminates.

| Node | Role |
|------|------|
| detector | compares each metric to its baseline, flags material moves |
| hypothesizer | selects the next dimension to test |
| investigator | decomposes the metric change along that dimension (`attribution.py`) |
| verifier | checks the decomposition against the gates (`gates.py`) |
| driller | after an ASSERT, re-decomposes within the winning segment to refine the cause |
| actuator | maps the verdict to a typed action: recommend, execute, or escalate |
| reporter | writes the conclusion and keeps the full audit trail |

**Division of labour:** All numbers come from pandas/numpy. The LLM (Qwen via DashScope)
does three things only: suggests the order of dimensions, writes the report from computed
figures, and — on ASSERT — offers business hypotheses explicitly labelled as speculation.
It never produces a number and never decides a verdict.

## Verification gates

`ASSERT` requires all four gates to pass. A failed gate produces an `ABSTAIN` with the
failing condition named in the response.

| Gate | Condition | Purpose |
|------|-----------|---------|
| material | \|ΔR\|/R₀ ≥ 2% | the move is large enough to matter |
| localized | top contribution share ≥ 0.55 | one segment actually dominates |
| significant | two-proportion z-test on the leading segment, p ≤ 0.01 | the leader's move is not sampling noise |
| clean | interaction share ≤ 0.50 | rate and mix effects are not entangled |

The significance gate is a real hypothesis test: a perfectly concentrated move on 60 users
abstains with p=0.55; the same move on 6000 users asserts with p<1e-5.

## Quickstart

Requires Python 3.12+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
QWEN_MOCK=1 pytest -q
QWEN_MOCK=1 uvicorn api.app:app --reload
```

## API

```
GET  /                     demo page
POST /investigate          built-in scenario: { "panel": "clean" | "diffuse" | "mixshift" | "deep", "autopilot": false }
POST /investigate/query    natural language: { "query": "why did conversion drop?" }
POST /investigate/upload   CSV upload (multipart: baseline + current)
POST /investigate/sql      live database: { "dsn": "...", "baseline_query": "...", "current_query": "..." }
POST /investigate/sheets   live Google Sheets: { "baseline_url": "...", "current_url": "..." }
POST /investigate/series   time series (multipart: series.csv + window)
GET  /panels/{name}        schema reference for SQL/Sheets/CSV
GET  /health               healthcheck
GET  /docs                 Swagger / ReDoc
```

## Bring your own data

### CSV upload

```
metric, <dim1>, [<dim2>, ...], n, c
```

```bash
curl -X POST localhost:8000/investigate/upload \
  -F baseline=@examples/baseline.csv \
  -F current=@examples/current_clean.csv
```

### SQL database (Postgres, MySQL, SQLite)

```bash
curl -X POST localhost:8000/investigate/sql -H 'content-type: application/json' -d '{
  "dsn": "postgresql://user:pass@host/db",
  "baseline_query": "SELECT metric, segment, device, n, c FROM conversions WHERE period = '\''last_month'\''",
  "current_query":  "SELECT metric, segment, device, n, c FROM conversions WHERE period = '\''this_month'\''"
}'
```

### Google Sheets

```bash
curl -X POST localhost:8000/investigate/sheets -H 'content-type: application/json' -d '{
  "baseline_url": "https://docs.google.com/spreadsheets/d/<id>/edit#gid=0",
  "current_url":  "https://docs.google.com/spreadsheets/d/<id>/edit#gid=1"
}'
```

## Attribution math

For a rate metric `R = Σ wₛ·rₛ`:

```
rate        = w₀·(r₁ − r₀)
mix         = r₀·(w₁ − w₀)
interaction = (w₁ − w₀)·(r₁ − r₀)
contribution = rate + mix + interaction
```

Zero residual. Validated against an independent oracle (`attribution_reference.py`).
Sum metrics (`decompose_sum`) use the same algebra with raw counts instead of shares.

## Docker

```bash
docker build -t prove-or-abstain .
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=... prove-or-abstain
```

For Alibaba Cloud: push to Container Registry, run on Function Compute (port 8000).
`/health` serves as the probe endpoint.

## Built for the Qwen Cloud Hackathon — Track 4: Autopilot Agent

| Requirement | Implementation |
|-------------|----------------|
| **Handle ambiguous inputs** | `/investigate/query` — Qwen routes free-text questions to the right scenario |
| **Invoke external tools** | SQL connector, Google Sheets connector, CSV upload, time series |
| **Human-in-the-loop checkpoints** | ABSTAIN always escalates; autopilot requires confidence ≥ 0.70 to execute |
| **Production-ready, not toy demo** | Docker, CI, 43 tests, full audit trail, Swagger docs |

**Qwen Cloud integration:** `llm.py` calls Qwen via DashScope for dimension ordering,
report phrasing, and query routing only. The math (pandas, numpy) and statistics
(z-test, p ≤ 0.01) run independently. The verdict is **identical** with or without the LLM — 
`QWEN_MOCK=1` proves this.

## What's next

- OAuth-native connectors (Stripe, GA4, Amplitude) beyond the current DSN/shared-link model
- Seasonality and trend modelling for time series
- Deeper drill-down (currently one level: winning segment × one other dimension)
- Downstream actions wired to real systems (Slack alerts, feature flags, campaign pausing)

## License

MIT.
