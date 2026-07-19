<!-- PROVE-OR-ABSTAIN -->
<p align="center">
  <img src="https://img.alicdn.com/imgextra/i3/O1CN01k1oD3R1Glhjj4hE2r_!!6000000000663-55-tps-158-28.svg" width="80" alt="logo" />
</p>

# prove-or-abstain

[![CI](https://github.com/Demba09/prove-or-abstain/actions/workflows/ci.yml/badge.svg)](https://github.com/Demba09/prove-or-abstain/actions)
[![benchmark](https://img.shields.io/badge/benchmark-100%25%20(30%2F30)-brightgreen.svg)](#benchmark)
[![false-ASSERT](https://img.shields.io/badge/false--ASSERT-0%25-brightgreen.svg)](#benchmark)
[![calibration](https://img.shields.io/badge/ECE-0.19-blue.svg)](#calibration)
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
cd prove-or-abstain
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
| investigator | decomposes the metric change along that dimension (`prove_or_abstain/attribution.py`) |
| verifier | checks the decomposition against the gates (`prove_or_abstain/gates.py`) |
| driller | after an ASSERT, re-decomposes within the winning segment to refine the cause |
| actuator | maps the verdict to a typed action: recommend, execute, or escalate |
| reporter | writes the conclusion and keeps the full audit trail |

**Division of labour:** All numbers come from pandas/numpy. The LLM (Qwen via DashScope)
does three things only: suggests the order of dimensions, writes the report from computed
figures, and — on ASSERT — offers business hypotheses explicitly labelled as speculation.
It never produces a number and never decides a verdict.

### Two orchestration modes

`POST /investigate` accepts `"mode": "graph"` (default) or `"mode": "agent"`:

- **graph** — the fixed LangGraph state machine above.
- **agent** — Qwen becomes the lead investigator. Instead of a hardcoded loop,
  it calls tools (`test_dimension`, `drill`, `finalize`) via OpenAI-style
  function calling and decides which dimension to test, in what order, and when
  to stop. The response carries an `agent_trace` of every tool call it made.

Crucially, **both modes return the identical verdict.** The tools run the same
gate math, and a determinism guard guarantees the LLM can never change the
outcome — if Qwen skips a dimension or finalizes early, every untested
dimension is checked deterministically before concluding, so a lazy or
divergent model can never cause a false ABSTAIN. Qwen drives the *path*; the
math decides the *verdict*. Offline (`QWEN_MOCK=1` or no key), the loop is
replayed deterministically and reproduces the graph exactly.

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

On ASSERT, a **confidence score** (product of the concentration, significance and
cleanliness factors, 0..1) gates the autopilot: EXECUTE requires confidence ≥ 0.70,
anything lower downgrades to RECOMMEND. Every EXECUTE is recorded in the audit trail
(`GET /executions`) and, if `WEBHOOK_URL` is set, POSTed to your endpoint
(Slack/Discord/Teams formats auto-detected) so a human sees every autonomous action.

## Benchmark

30 synthetic scenarios with **known ground truth derived from how each panel is
generated** (a paid-only collapse ⇒ `ASSERT segment=paid`; a uniform drop ⇒
`ABSTAIN`) — never from the pipeline's own output, so accuracy is not circular.
Run it yourself, offline, in ~2 seconds:

```bash
python -m prove_or_abstain.benchmark
```

| Category | Scenarios | Expected | Result |
|---|---|---|---|
| clean → localizes on segment | 5 | ASSERT segment | 5/5 ✅ |
| clean → localizes on device | 5 | ASSERT device | 5/5 ✅ |
| clean → no dominant cause | 3 | ABSTAIN | 3/3 ✅ |
| diffuse (systemic) | 5 | ABSTAIN | 5/5 ✅ |
| mixshift (entangled) | 3 | ABSTAIN | 3/3 ✅ |
| deep (ASSERT + drill-down) | 3 | ASSERT device | 3/3 ✅ |
| edge (small-n, sum metric, single dim) | 3 | mixed | 3/3 ✅ |
| noisy (borderline confidence) | 3 | ASSERT | 3/3 ✅ |

```
accuracy = 100% (30/30)   false-ASSERT = 0%   false-ABSTAIN = 0%
```

The result is **identical in `graph` and `agent` mode** — the math decides, so
Qwen's orchestration can't change a verdict. The critical number is
**false-ABSTAIN = 0% paired with false-ASSERT = 0%**: the agent never misses a
real, localizable cause, and never invents one that isn't there.

### vs. a raw LLM

`compare_llm_raw()` gives a bare Qwen only a text summary (no data) and asks for
the cause. Where the truth is "no single cause" (systemic), a raw model tends to
invent a plausible-sounding culprit — exactly the hallucination the gates
prevent. This needs a live key:

```bash
DASHSCOPE_API_KEY=sk-... python -m prove_or_abstain.benchmark
```

> _Live hallucination-rate numbers are populated from a real run; run the command
> above with a key to reproduce them in your own environment._

## Calibration

Does a confidence of 0.7 mean "right ~70% of the time"?
`python -m prove_or_abstain.calibrate` buckets the benchmark's ASSERT
predictions by confidence and reports the Expected Calibration Error:

```
ECE = 0.19   (n = 18 ASSERT predictions)
```

Every asserted cause in the benchmark is correct — even at 0.41 confidence — so
the score is **conservative**: it under-states reliability rather than
over-stating it. For an agent that can *act* on its verdict, erring toward
under-confidence is the safe direction.

## Cost

Token usage is tracked per request (`cost` field in the API response) and per
model. Pricing (DashScope international, $/1M tokens):

| Model | Input | Output |
|---|---|---|
| qwen-turbo | $0.40 | $1.20 |
| qwen-plus | $0.80 | $2.40 |
| qwen-max | $1.40 | $5.60 |

Because the **verdict is model-independent** (the gates decide, the LLM only
orchestrates and phrases), `cross_model_eval()` shows the same accuracy across
qwen-turbo/plus/max — so you can run the **cheapest** model without losing
correctness. The cross-model latency/cost table is generated live:

```bash
DASHSCOPE_API_KEY=sk-... python -m prove_or_abstain.benchmark
```

> _Measured latency, tokens and cost per model are populated from a real run;
> run the command above with a key to reproduce them._

## Repository layout

```
prove_or_abstain/   core package — the deterministic pipeline
  agent_state.py      typed state shared by the graph nodes
  metrics.py          aggregation of the long-panel counts
  attribution.py      rate/mix/interaction decomposition
  gates.py            the 4 verification gates + confidence score
  nodes.py            detector → hypothesizer → investigator → verifier → …
  graph.py            the compiled LangGraph state machine
  agent_loop.py       Qwen-orchestrated alternative (mode="agent")
  llm.py              the Qwen boundary (mock mode, routing, wording, tools)
  panels.py           built-in demo scenarios
  autopilot.py        execution tracker (adapter over memory.py)
  memory.py           SQLite persistence — investigation history + alerts
  monitor.py          continuous autonomous surveillance loop
  webhook.py          outbound notifications on EXECUTE
  cost_tracker.py     token counting + cost estimation
  benchmark.py        30 ground-truth scenarios + cross-model eval
  calibrate.py        confidence calibration + ECE
  audit.py            reproducible, verifiable audit trails
  evidence.py         synthetic operational-event lookup, grounds ASSERT speculation
  connectors/         SQL (Postgres/MySQL/SQLite) and Google Sheets
api/                deployment entry point — FastAPI app + static demo page (SSE stream)
mcp_server.py       MCP entry point for Qwen Cloud agents
scripts/            validation & demo tooling (see below)
tests/              pytest suite (64 tests, runs offline with QWEN_MOCK=1)
examples/           sample CSVs for the upload endpoints
docs/               architecture diagram, demo script, devpost text
```

## Development setup

Requires Python 3.12+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
QWEN_MOCK=1 pytest -q
QWEN_MOCK=1 uvicorn api.app:app --reload
```

### Configuration

Copy `.env.example` to `.env` (never committed, excluded from the Docker image):

| Variable | Effect |
|----------|--------|
| `DASHSCOPE_API_KEY` | Qwen/DashScope key. Absent → the app runs in deterministic mock mode |
| `QWEN_BASE_URL` | DashScope endpoint — intl vs mainland China accounts differ, see `.env.example` |
| `QWEN_MODEL` | model name, defaults to `qwen-plus` |
| `QWEN_MOCK=1` | force mock mode even with a key set |
| `WEBHOOK_URL` | where EXECUTE notifications are POSTed — payload auto-formats for Slack, Discord or Teams from the hostname, generic JSON otherwise. Absent → stdout |
| `PROBATIO_DB` | SQLite path for investigation history + alerts (`memory.py`). Default `:memory:` — set a file path to persist across restarts |

### Validate the pipeline yourself

Every layer has an independent check that runs offline:

```bash
python scripts/gate_check.py        # math layer vs a hand-written oracle
python scripts/gate_check_gates.py  # decision layer on 3 calibrated scenarios
python scripts/simulate.py          # full flow without LangGraph, mock-forced
python scripts/run_phase1.py        # the 2 headline scenarios through the real graph
python scripts/check_qwen.py        # is my DashScope key/endpoint alive?

python -m prove_or_abstain.benchmark   # 30 ground-truth scenarios -> accuracy
python -m prove_or_abstain.calibrate   # confidence calibration + ECE
python -m prove_or_abstain.monitor     # one autonomous surveillance cycle
python -m prove_or_abstain.audit       # audit trail + reproducibility check
```

## API

```
GET  /                     demo page
POST /investigate          built-in scenario: { "panel": "clean" | "diffuse" | "mixshift" | "deep", "autopilot": false, "mode": "graph" | "agent" }
GET  /investigate/stream   Server-Sent Events: stream the investigation step by step (?panel=&autopilot=)
POST /investigate/query    natural language: { "query": "why did conversion drop?", "previous_panel": "clean" }
POST /investigate/suggest  setup helper: upload a sample CSV, get back sum-vs-rate metric classification
POST /investigate/upload   CSV upload (multipart: baseline + current)
POST /investigate/sql      live database: { "dsn": "...", "baseline_query": "...", "current_query": "..." }
POST /investigate/sheets   live Google Sheets: { "baseline_url": "...", "current_url": "..." }
POST /investigate/series   time series (multipart: series.csv + window)
POST /investigate/check    autonomous monitor — runs all panels, auto-executes on high confidence
GET  /panels/{name}        schema reference for SQL/Sheets/CSV
GET  /dashboard            autopilot status, active alerts, uptime
GET  /executions           audit trail of all EXECUTE actions
POST /executions/{id}/resolve  human resolves an active alert
GET  /health               healthcheck
```

### Where Qwen's contribution stops being decorative

Ordering 2 dimensions or rephrasing an already-computed verdict is low-stakes
busywork a template does just as well (`QWEN_MOCK=1` proves it — same
verdicts, same accuracy). Four places narrow that gap to where an LLM
actually beats a fixed rule:

- **`suggest_setup()`** (`POST /investigate/suggest`) — classifying an
  unfamiliar metric NAME as rate or sum is a real text-understanding call;
  unlike dimensions (exactly inferred from the CSV's own columns, no
  ambiguity), a rule-based keyword list is the fallback, not the answer.
- **Wider dimension spaces** — `plan_dimensions()`'s ordering only matters
  when there's more than 2 candidates to order. `examples/plan_baseline.csv`
  / `plan_current.csv` add a 3rd dimension (`plan`) where neither `segment`
  nor `device` localizes (concentration 0.25 and 0.50, both < 0.55) and only
  `plan` does (1.0) — testing it first instead of last finds the cause in 1
  iteration instead of 3. Same verdict either order; real difference in cost
  and latency (see `test_dimension_order_changes_speed_not_verdict`).
- **Conversational follow-up** — `POST /investigate/query` accepts
  `previous_panel` plus a follow-up like *"and on mobile only?"*: Qwen may
  select a `(dim, segment)` filter from the panel's known values (guarded by
  `_guard_filter`, same anti-invention rule as the panel/metric selection)
  and the pipeline re-runs, filtered, unchanged otherwise.
- **Evidence-grounded speculation** (`prove_or_abstain/evidence.py`) — on
  ASSERT, `speculate_causes()` is handed any operational events already
  logged for the winning segment (a campaign change, a deploy...) and grounds
  a hypothesis in the most relevant one instead of guessing blindly. Still
  labelled speculation for a human to confirm — this is a small embedded
  table standing in for a real calendar/deploy-log integration (see
  "What's next").

## Autonomous monitoring, persistence & audit

The Track-4 autopilot is a continuous loop, not just an endpoint:

- **`monitor.py`** — `MetricMonitor` watches a set of sources (SQL / Sheets /
  CSV / inline), and every cycle fetches the current panel, compares it to the
  rolling baseline, and on a material move runs the investigation, persists the
  verdict, and fires the webhook on a confident ASSERT. One broken feed never
  kills the loop.

  ```bash
  python -m prove_or_abstain.monitor          # one demo cycle on a built-in panel
  ```

- **`memory.py`** — SQLite persistence (`PROBATIO_DB`, default `:memory:`) for
  the full investigation history and deduplicated active alerts. `autopilot.py`
  is a thin adapter over it, so `/dashboard`, `/executions` and
  `/executions/{id}/resolve` are backed by a real store.

- **`audit.py`** — freezes any investigation into a verifiable trail (SHA256
  input hash, Qwen's tool calls, the four gate decisions, verdict/confidence,
  cost). `verify_replay()` re-runs the same inputs and confirms the verdict is
  bit-for-bit reproducible — the guarantee an auditor wants.

### Architecture

```
              data sources                      Qwen Cloud (DashScope)
     SQL · Sheets · CSV · inline                  orchestrates + phrases
                 │                                        │ tool calls
                 ▼                                        ▼
   monitor.py ──────────────►  agent_loop / graph  ◄── gates decide the verdict
   (rolling baseline,          detector→investigate      (pure pandas/numpy)
    ≥2% triggers)              →verify→drill→act              │
                 │                     │                      │
                 ▼                     ▼                      ▼
        memory.py (SQLite)     webhook.notify          audit.py (SHA256 trail
     history + active alerts   Slack/Discord/Teams        + verify_replay)
                 │                                     cost_tracker.py ($/tokens)
                 ▼
     /dashboard · /executions · SSE /investigate/stream
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

Zero residual. Validated against an independent oracle (`scripts/attribution_reference.py`).
Sum metrics (`decompose_sum`) use the same algebra with raw counts instead of shares.

## Docker

```bash
docker build -t prove-or-abstain .
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=... prove-or-abstain
```

For Alibaba Cloud: push to Container Registry, run on Function Compute (port 8000).
`/health` serves as the probe endpoint.

## Qwen Cloud MCP Server

Prove-or-Abstain exposes an **MCP (Model Context Protocol)** server so Qwen Cloud
agents can call it directly as a tool — making Qwen the primary orchestrator.

```bash
python mcp_server.py           # stdio transport — connect to Qwen Cloud
python mcp_server.py --port 8080  # SSE transport for testing
```

**Available MCP tools:**

| Tool | Description |
|------|-------------|
| `investigate_scenario` | Run investigation on a built-in scenario |
| `investigate_sql` | Run investigation from a live database query |
| `autonomous_check` | Autonomous monitoring — checks all panels with autopilot ON |
| `get_dashboard` | View active alerts, total checks, uptime |
| `resolve_alert` | Human-in-the-loop — mark an alert as resolved |
| `describe_panels` | List available scenarios so Qwen knows what to call |
| `describe_gates` | Explain the 4 verification gates |

With MCP, a Qwen agent:
1. Receives a user question (e.g., "why did conversion drop?")
2. Calls `describe_panels` to see available scenarios
3. Calls `investigate_scenario("clean")` and `investigate_scenario("diffuse")`
4. Interprets the results: "The drop localizes to paid — but a diffuse scenario shows it could be systemic"
5. Generates a human-readable response with recommendations

**Qwen is now the agent. Prove-or-Abstain is its skill.**

## Built for the Qwen Cloud Hackathon — Track 4: Autopilot Agent

| Requirement | Implementation |
|-------------|----------------|
| **Handle ambiguous inputs** | `/investigate/query` — Qwen routes free-text questions to the right scenario |
| **Qwen orchestrates via tool calls** | `mode="agent"` — Qwen drives the investigation through function calling (`agent_loop.py`), math still decides |
| **Invoke external tools** | SQL connector, Google Sheets connector, CSV upload, time series |
| **Continuous autonomy** | `monitor.py` watches sources, investigates on movement, persists + alerts |
| **Human-in-the-loop checkpoints** | ABSTAIN always escalates; autopilot requires confidence ≥ 0.70 to execute; alerts resolvable |
| **Provable, not just a demo** | 30-scenario benchmark (100%, 0% false-ASSERT), ECE calibration, reproducible audit trails, per-request cost |
| **Production-ready** | Docker, CI, 64 tests, SQLite persistence, SSE streaming, API docs at `/docs` (ReDoc) |

**Qwen Cloud integration:** `prove_or_abstain/llm.py` calls Qwen via DashScope for dimension ordering,
report phrasing, and query routing only. The math (pandas, numpy) and statistics
(z-test, p ≤ 0.01) run independently. The verdict is **identical** with or without the LLM — 
`QWEN_MOCK=1` proves this.

## What's next

- OAuth-native connectors (Stripe, GA4, Amplitude) beyond the current DSN/shared-link model
- Seasonality and trend modelling for time series
- Deeper drill-down (currently one level: winning segment × one other dimension)
- Downstream actions wired to real systems (Slack alerts, feature flags, campaign pausing)
- `evidence.py`'s embedded table replaced by a real calendar/deploy-log/ticketing integration
- Multi-turn `/investigate/query` beyond a single filtered follow-up (a real conversation, not one filter)

## License

MIT.
