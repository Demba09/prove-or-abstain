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

> **"Conversion dropped 3% this week. Campaign? Bug? Market?"**
>
> Analytics tools will always give you an answer. Prove-or-Abstain is different.  
> It investigates, tests, and **refuses to guess** when the data doesn't prove a cause.

<br>

## What it does

A product team opens their dashboard on Monday morning. Conversion is down 3%.
They have two choices:

- **Dig through dashboards for hours**, trying to isolate which segment, device, or channel broke.
- **Ask an agent that does it in seconds** — and only acts when the math proves it.

Prove-or-Abstain investigates metric changes across your segments and returns one of two verdicts:

| Verdict | Meaning | Example |
|---------|---------|---------|
| **ASSERT** | Cause found and proven | *"'Paid' segment's rate collapsed (p < 0.001). Recommend pausing the campaign."* |
| **ABSTAIN** | No single cause isolated | *"Drop is real but diffuse across all segments. Cannot isolate — escalate to human."* |

**The ABSTAIN verdict is the point.** An agent allowed to act on data needs a principled way to refuse — otherwise it will always produce a plausible-sounding diagnosis, right or wrong.

![architecture](docs/architecture.svg)

## Try it now

```bash
git clone https://github.com/Demba09/prove-or-abstain
pip install -r requirements.txt
QWEN_MOCK=1 uvicorn api.app:app --reload
# Open http://localhost:8000
```

Or **bring your own data**: SQL databases, Google Sheets, CSV uploads, or time series.

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

Requires Python 3.12+ (the pinned numpy version does not install on 3.11 or
older; developed and tested on 3.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export QWEN_MOCK=1        # run offline; omit if DASHSCOPE_API_KEY is set
pytest -q                 # math vs. oracle, gate verdicts, API behaviour

uvicorn api.app:app --reload
```

Open http://localhost:8000 for the demo page — run the built-in scenarios,
ask a free-text question, or upload your own CSVs — or use the API directly:

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
            action, report, speculations, llm, trace }
POST /investigate/query
  body:   { "query": "<free-text question>", "autopilot": false }
  Qwen SELECTS one of the four built-in panels/metrics from the question
  (llm.route_query, guarded to the supplied options only — see "Using Qwen
  for real" below); the same deterministic pipeline then runs unchanged.
  return: { panel, routing: {panel, metric, reason}, ...same shape as above }
POST /investigate/upload
  multipart: baseline=<csv>, current=<csv>, autopilot=<bool>, sum_metrics=<csv names>
  return: { panel, dataset: {baseline, current}, ...same shape as /investigate }
POST /investigate/sql
  body: { "dsn": "<sqlalchemy url>", "baseline_query": "<single SELECT>",
          "current_query": "<single SELECT>", "autopilot": false, "sum_metrics": "" }
  pulls both panels straight from a database (connectors/sql.py) instead of a
  CSV round trip — same long-panel contract, see "Live data: the SQL connector" below
POST /investigate/sheets
  body: { "baseline_url": "<docs.google.com spreadsheet URL>",
          "current_url": "<docs.google.com spreadsheet URL>", "autopilot": false, "sum_metrics": "" }
  pulls both panels from a Google Sheet (connectors/gsheets.py) — same
  long-panel contract, see "Live data: Google Sheets" below
POST /investigate/series
  multipart: series=<csv with a 'period' column>, window=<int, optional>
  last period vs. a rolling baseline pooled over the prior `window` periods
GET  /panels/{clean|diffuse|mixshift|deep}
  return: { panel, baseline: [...rows], current: [...rows] }
  the raw long-panel rows behind a built-in scenario — a live reference for
  the shape /investigate/sql, /investigate/sheets and /investigate/upload expect
GET  /health
```

`upload`/`sql`/`sheets` echo the exact rows they fed the pipeline back as
`dataset: {baseline, current}` — the demo page uses it (plus `/panels/{name}`
for the built-in scenarios) to draw a baseline-vs-current chart per segment
under the verdict, so you can see what's actually driving the numbers
instead of just the aggregate gates.

## Using Qwen for real (not just mock mode)

`llm.py` calls Qwen (`qwen-plus` by default) via DashScope's OpenAI-compatible
endpoint for exactly three things: ordering which dimension to try
(`plan_dimensions`), routing a free-text question to a panel
(`route_query`), and phrasing the final report/speculations — never a
number, never a verdict.

- **Get a key**: a `DASHSCOPE_API_KEY` from Alibaba Cloud Model Studio.
  `export DASHSCOPE_API_KEY=...` (or put it in `.env`), then unset
  `QWEN_MOCK` (or leave it unset).
- **Region matters**: international accounts use the default base URL
  (`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`); mainland China
  accounts must set `QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`.
  A key/endpoint mismatch fails the network call — and the client falls back
  to the deterministic template so the pipeline still runs. That fallback is
  a safety property, but it can silently look identical to mock mode.
- **Verify before demoing**: run `python check_qwen.py` for a one-shot
  real round trip, and check the `llm` field in any `/investigate*`
  response (`{"model": "qwen-plus", "mode": "real" | "mock" | "fallback"}`)
  — `mode` is also appended to `report` on the demo page and logged in
  `trace` (e.g. `hypothesizer: dimension plan [qwen:qwen-plus] -> [...]`,
  or `[fallback] (<error>)` if a real call failed and the template kicked
  in). This makes a live Qwen call and a mock/fallback run distinguishable
  in the API output itself, not just in the source code.

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

Malformed data is rejected with a 400 before it can reach the math:
missing or negative `n`/`c` values, and `c > n` on a rate metric (declare
revenue-like metrics via `sum_metrics` — for those, `c` is a total and may
exceed `n`). This applies to every data source: upload, SQL, Sheets, series.

### Live data: the SQL connector

`POST /investigate/sql` skips the CSV round trip: it runs two of your own
queries directly against a database (Postgres, MySQL, SQLite — anything
SQLAlchemy has a driver for) and feeds the result straight into the same
pipeline, as long as each query already projects onto the long-panel shape
(`metric, <dims...>, n, c`):

```bash
curl -X POST localhost:8000/investigate/sql -H 'content-type: application/json' -d '{
  "dsn": "postgresql://user:pass@host/db",
  "baseline_query": "SELECT metric, segment, device, n, c FROM conversions WHERE period = '\''last_month'\''",
  "current_query":  "SELECT metric, segment, device, n, c FROM conversions WHERE period = '\''this_month'\''"
}'
```

Trust model: you supply your own DSN and credentials — the endpoint grants
no access beyond what that connection already has. The one guard it adds
(`connectors/sql.py`) is scoping each call to a single read-only statement
(`SELECT` or `WITH ... SELECT`, no `;`-separated second statement); it is a
safety rail against accidents, not a substitute for connecting with a
SELECT-only role or a reporting replica in a real deployment.

**There is no import step.** The endpoint doesn't copy your data anywhere —
it runs your query live and hands the result straight to the pipeline. That
means the query's result columns *are* the panel: if your source table is
at the raw-event grain (one row per user/transaction) rather than already
aggregated per cell, the aggregation has to happen in the query itself, e.g.:

```sql
-- raw events table: (user_id, segment, device, converted, period)
SELECT 'conversion' AS metric, segment, device,
       COUNT(*)               AS n,
       SUM(converted)         AS c
FROM   events
WHERE  period = 'this_month'
GROUP BY segment, device
```

Not sure your query is shaped right? `GET /panels/clean` (or `diffuse` /
`mixshift` / `deep`) returns the exact long-panel rows behind that built-in
scenario — a live reference for the shape a query or sheet needs to produce.

### Live data: Google Sheets

`POST /investigate/sheets` reads two Google Sheets (or two tabs of the same
sheet, via their `gid`) directly — no export-to-CSV-then-upload step. Each
sheet must already be in the long-panel shape and shared as "anyone with
the link" (or published to the web):

```bash
curl -X POST localhost:8000/investigate/sheets -H 'content-type: application/json' -d '{
  "baseline_url": "https://docs.google.com/spreadsheets/d/<id>/edit#gid=0",
  "current_url":  "https://docs.google.com/spreadsheets/d/<id>/edit#gid=1"
}'
```

`connectors/gsheets.py` accepts a share link, an `edit#gid=` link, or an
already-built CSV export link, and normalizes any of them to the sheet's
CSV export endpoint. It only ever requests `docs.google.com` — any other
host is rejected before a request is made, so this can't be repurposed as
a general URL fetcher.

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

## Built for the Qwen Cloud Hackathon

This project is submitted to **Track 4: Autopilot Agent** — build an agent
that automates real-world business workflows end-to-end.

| Requirement | How we meet it |
|-------------|---------------|
| **Handle ambiguous inputs** | `/investigate/query` accepts free-text questions; Qwen routes them to the right analysis panel |
| **Invoke external tools** | SQL connector (Postgres/MySQL/SQLite), Google Sheets connector |
| **Human-in-the-loop checkpoints** | ABSTAIN never executes; autopilot requires confidence ≥ 0.70 |
| **Production-ready** | Docker image, CI pipeline, 43 tests, FastAPI with Swagger docs |

**Qwen Cloud integration:** The agent calls Qwen models via
[DashScope](https://dashscope-intl.aliyuncs.com/compatible-mode/v1) for three
things only — dimension ordering, report phrasing, and query routing. It
never uses the LLM to compute a number or decide a verdict: the math (pandas,
numpy) and the statistics (z-test, p ≤ 0.01) run independently. The verdict
is **identical** with or without the LLM.

## Limitations

## What's next

- OAuth connectors (Stripe, GA4, Amplitude) beyond the current DSN/shared-link model
- Seasonality and trend modelling for time series
- Deeper drill-down (currently one level: winning segment × one other dimension)
- Downstream actions wired to real systems (Slack alerts, feature flags, campaign pausing)

## License

MIT.
