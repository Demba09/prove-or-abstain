<!-- PROVE-OR-ABSTAIN -->
<p align="center">
  <img src="https://img.alicdn.com/imgextra/i3/O1CN01k1oD3R1Glhjj4hE2r_!!6000000000663-55-tps-158-28.svg" width="80" alt="logo" />
</p>

# prove-or-abstain

[![CI](https://github.com/Demba09/prove-or-abstain/actions/workflows/ci.yml/badge.svg)](https://github.com/Demba09/prove-or-abstain/actions)
[![benchmark](https://img.shields.io/badge/benchmark-100%25%20(20%2F20)-brightgreen.svg)](#benchmark)
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

## Built for the Qwen Cloud Hackathon — Track 4: Autopilot Agent

| Requirement | Implementation |
|-------------|----------------|
| **Handle ambiguous inputs** | `/investigate/query` routes free-text questions; `map_schema()` reshapes an unfamiliar raw source's columns for "Watch a source" |
| **Qwen orchestrates via tool calls** | `mode="agent"` (toggle in the demo) — Qwen drives the investigation through function calling (`agent_loop.py`), math still decides |
| **Invoke external tools** | SQL connector, Google Sheets connector, CSV upload, time series, continuous source ingestion |
| **Continuous autonomy** | `monitor.py` watches sources, investigates on movement, persists a durable baseline, alerts |
| **Human-in-the-loop checkpoints** | ABSTAIN always escalates; autopilot requires confidence ≥ 0.70 to execute; alerts resolvable |
| **Provable, not just a demo** | 33-scenario benchmark (100%, 0% false-ASSERT) — 30 synthetic + 3 real-world datasets, ECE calibration, reproducible audit trails, per-request cost |
| **Production-ready** | Docker, CI, 105 tests, SQLite persistence, SSE streaming, API docs at `/docs` (ReDoc) |

Qwen (via DashScope) orders dimensions, phrases reports, routes questions, and — the one
deliberate exception — maps an unfamiliar source's columns. Everywhere else the math decides,
verdict-identical with or without the LLM (`QWEN_MOCK=1` proves it). Details in
[Where Qwen actually earns its keep](#where-qwen-actually-earns-its-keep).

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

### What the agent actually does — a trace

Here's a real agent-mode run on the "clean" scenario with a live `DASHSCOPE_API_KEY`.
The trace shows Qwen calling tools and the gates deciding at each step:

```
[1] list_dimensions()                       Qwen surveys the available dimensions
[2] test_dimension(device) → ABSTAIN        Qwen's first guess — wrong
      gate: concentration=0.50 < 0.55   ✗   the drop is not concentrated on one device
      gate: p=0.076 > 0.01              ✗   not statistically significant
[3] test_dimension(segment) → ASSERT        Qwen's second guess — right
      gate: concentration=1.00 ≥ 0.55   ✓   paid alone accounts for the entire drop
      gate: p=4×10⁻⁶ ≤ 0.01            ✓   two-proportion z-test, highly significant
[4] drill(segment) → ABSTAIN                is there a sub-cause within paid?
      → no — 50/50 mobile/desktop split     nothing to refine further
[5] finalize() → ASSERT segment=paid        verdict locked, confidence 0.79

Cost: 4 481 tokens = $0.004                 4/10 of a cent per investigation
```

**What this trace proves:**
- Qwen's first suggestion (`device`) was wrong — the gates caught it and the agent looped
- The second suggestion (`segment=paid`) passed all four gates with mathematical proof
- Qwen only drove the search path; the gates in `gates.py` decided the outcome
- Every gate failure names its reason — not a black-box "no", but a specific, auditable condition
- Total cost per investigation: fractions of a cent

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
watch a source with no baseline file, or ask questions in plain English about whichever you
last ran. A **graph / agent** toggle switches between the fixed pipeline and Qwen orchestrating
the investigation via tool calls — the trace of its calls shows below the verdict, and the
verdict is identical either way. (CSV/SQL/Sheets upload also available — see the API table.)

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

**Division of labour:** All numbers come from pandas/numpy. Qwen (via DashScope) orders
dimensions, phrases the report, routes questions, and offers labelled speculation — never a
number, never a verdict. The one place its judgment can change an outcome (`map_schema()`) is
called out explicitly below.

### Two orchestration modes

`POST /investigate` accepts `"mode": "graph"` (default) or `"mode": "agent"`:

- **graph** — the fixed LangGraph state machine above.
- **agent** — Qwen becomes the lead investigator. Instead of a hardcoded loop,
  it calls tools (`test_dimension`, `drill`, `finalize`) via OpenAI-style
  function calling and decides which dimension to test, in what order, and when
  to stop. The response carries an `agent_trace` of every tool call it made.

Crucially, **both modes return the identical ASSERT/ABSTAIN verdict.** The tools
run the same gate math, and a determinism guard guarantees the LLM can never
turn a real cause into a false ABSTAIN by skipping a dimension or finalizing
early — every untested dimension is checked deterministically before
concluding. Offline (`QWEN_MOCK=1` or no key), the loop is replayed
deterministically and reproduces the graph exactly.

**One honest nuance, found by running this against a live key:** when a
single narrow cell collapses (the `deep` scenario), it concentrates 100% on
*both* of its defining dimensions at once. Qwen's test order then decides
which is the top-level `root_cause` and which is the `drilldown.refined`
detail — but the drill-down always recovers the other, so the full
diagnosis is never lost, only relabelled. The benchmark below credits a
match on either field for this reason.

### Where Qwen actually earns its keep

Ordering 2 dimensions or rephrasing an already-computed verdict is low-stakes
busywork a template does just as well — the guarantee above. Four places
narrow that gap to where an LLM beats a fixed rule outright:

- **`suggest_setup()`** (`POST /investigate/suggest`) — classifying an
  unfamiliar metric NAME as rate or sum is a real text-understanding call;
  dimensions need no such help (exactly inferred from the CSV's own columns).
- **Wider dimension spaces** — ordering only matters past 2 candidates.
  `examples/plan_baseline.csv`/`plan_current.csv` add a 3rd dimension
  (`plan`) that neither `segment` nor `device` alone localizes — testing it
  first instead of last finds the cause in 1 iteration instead of 3. Same
  verdict either order; real difference in cost and latency.
- **Conversational follow-up** — `POST /investigate/query` accepts
  `previous_panel` plus a follow-up like *"and on mobile only?"*: Qwen may
  select a `(dim, segment)` filter from values it's given (never invents
  one) and the pipeline re-runs, filtered. Pass `source_id` instead to ask
  the same kind of question about a **"Watch a source"** id — there's one
  active dataset there, not 4 named panels, so `extract_filter()` only
  does the filter half of routing (`llm.py`'s `route_query()` docstring).
- **Evidence-grounded speculation** (`prove_or_abstain/evidence.py`) — on
  ASSERT, `speculate_causes()` is handed any operational events already
  logged for the winning segment and grounds a hypothesis in the most
  relevant one, still labelled speculation. A small embedded table standing
  in for a real calendar/deploy-log integration.

**One deliberate exception: `map_schema()`.** Everything above is provably
LLM-independent (`QWEN_MOCK=1` proves it — same verdict, same accuracy).
`map_schema()` (used by `POST /sources/{id}/observe` on a raw source whose
columns don't already match `[metric, dims..., n, c]`) is different on
purpose: it decides which column *is* `n`, which is `c`, which are
dimensions — the shape of the data feeding the calculation, not its order or
wording. Interpreting ambiguous column names has no single deducible answer,
so mock and real mode can genuinely disagree here. Rather than hide that
behind a human-confirmation gate (which would make the decision decorative
again), it's used directly: real mode runs it in **two passes** — a
proposal, then a self-verification pass where Qwen re-examines its own
answer and may correct it (`self_verified: false` when it does). Either way,
the result still passes through the same `_validate_panel`/
`_validate_rate_counts` every data source goes through, rejecting an
incoherent mapping before it reaches `gates.py` — a deterministic backstop,
not a human one.

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

20 scenarios — 10 synthetic (one per gate edge case) and 10 real-world public datasets
(seaborn, vega, UCI, fivethirtyeight) with independently verifiable outcomes. Run it
yourself, offline, in ~2 seconds — it also writes
[`benchmark_results.json`](benchmark_results.json), a committed, inspectable
record of the actual run:

```bash
python -m prove_or_abstain.benchmark
```

| Category | Scenarios | Expected | Result |
|---|---|---|---|---|
| clean → ASSERT segment or device | 2 | ASSERT | 2/2 ✅ |
| abstain (no single cause) | 1 | ABSTAIN | 1/1 ✅ |
| diffuse (systemic) | 1 | ABSTAIN | 1/1 ✅ |
| mixshift (entangled) | 1 | ABSTAIN | 1/1 ✅ |
| deep (ASSERT + drill-down) | 1 | ASSERT | 1/1 ✅ |
| edge (sum metric, tiny sample, single dim) | 3 | mixed | 3/3 ✅ |
| noisy (borderline confidence) | 1 | ASSERT | 1/1 ✅ |
| real data (Titanic, majors, penguins, flights, tips, mpg, diamonds, gapminder, cars, iris) | 10 | mixed | 10/10 ✅ |

```
accuracy = 100% (20/20)   false-ASSERT = 0%   false-ABSTAIN = 0%
```
accuracy = 100% (33/33)   false-ASSERT = 0%   false-ABSTAIN = 0%
```

The result is **identical in `graph` and `agent` mode** — the math decides, so
Qwen's orchestration can't change a verdict. The critical number is
**false-ABSTAIN = 0% paired with false-ASSERT = 0%**: the agent never misses a
real, localizable cause, and never invents one that isn't there.

**What this 100% proves, and what it doesn't.** Every scenario is built
clearly on one side of the 4 gate thresholds — none sits at the exact
boundary. So 100% proves the code correctly implements its own documented
rules (a real regression test), not correct behaviour on genuinely
ambiguous data. That's what "Tested against real data" below is for.

### vs. a raw LLM

Here's the same "diffuse" scenario — a uniform drop across all segments — asked two ways:

**Raw Qwen, given the full data and a realistic PM question:**
> *"I'm a PM. Conversion dropped. Here's the data. What caused this? Should I pause any campaign?"*

Qwen responds with a detailed multi-paragraph analysis, computing per-segment rates,
framing plausible explanations — but **never says the drop is systemic**. A PM reading
this would likely act on a plausible-sounding cause that doesn't exist.

**Our agent, same data:**
```
ABSTAIN — diffuse cause (concentration=0.50 < 0.55)
→ ESCALATE to human
```

No long analysis. No invented cause. Just the math — and an explicit refusal to guess.

`compare_llm_raw()` quantifies this systematically across all 30 scenarios (needs a live key):

```bash
DASHSCOPE_API_KEY=sk-... python -m prove_or_abstain.benchmark
```

## Tested against real data, not just planted scenarios

Every scenario above is synthetic by necessity — ground truth has to be
known in advance to grade accuracy. As an external sanity check, three
public, real (not invented) datasets go through the same pipeline,
committed in `examples/` and pinned by tests so CI catches any drift:

- **`examples/real_flights_series.csv`** — [seaborn-data's `flights.csv`](https://github.com/mwaskom/seaborn-data),
  real monthly airline passenger counts, 1949–1960. 1960 grew **+11.2%**
  over 1959 (a real trend — the postwar air travel boom), and the pipeline
  correctly **ABSTAINs**: growth concentration by month is 0.13, nowhere
  near the 0.55 threshold — genuinely systemic, not seasonal, and nothing
  was planted to make that true.
  ```bash
  curl -X POST localhost:8000/investigate/series -F series=@examples/real_flights_series.csv -F window=1 -F sum_metrics=passengers
  ```
- **`examples/real_titanic_southampton.csv` / `_cherbourg.csv`** — [seaborn-data's `titanic.csv`](https://github.com/mwaskom/seaborn-data),
  the real passenger manifest, split by embarkation port (Southampton
  n=644, Cherbourg n=168 — real, unequal, non-round group sizes). Overall
  survival jumps 34% → 55%; the popular explanation is "Cherbourg had more
  1st class passengers", but decomposed honestly, `pclass` alone does
  **not** clear the significance gate (p=0.10) while `sex` does (p=0.0018) —
  the well-documented "women and children first" effect dominates.
  Confidence comes out genuinely low (0.09) on real, noisy, small-sample
  data, correctly staying a `RECOMMEND`, never an auto-`EXECUTE`.
  ```bash
  curl -X POST localhost:8000/investigate/upload \
    -F baseline=@examples/real_titanic_southampton.csv -F current=@examples/real_titanic_cherbourg.csv
  ```
- **`examples/real_majors_nonstem.csv` / `_stem.csv`** — [fivethirtyeight's
  `college-majors`](https://github.com/fivethirtyeight/data/tree/master/college-majors)
  `recent-grads.csv`, 173 real US majors with real employment counts. STEM
  majors (the dataset's own classification, not one we picked) employ at
  75.0% vs. 81.0% for everything else — and that gap is **not** uniform: it
  concentrates in majority-women majors (81.5% → 68.8%) far more than
  majority-men ones (79.9% → 79.2%), a real, un-planted finding with
  genuinely low confidence, correctly staying a `RECOMMEND`.
  ```bash
  curl -X POST localhost:8000/investigate/upload \
    -F baseline=@examples/real_majors_nonstem.csv -F current=@examples/real_majors_stem.csv
  ```
  The same data also exists as `_raw.csv`, with plausible, unrenamed column
  names (`Field`, `Group`, `Total`, `Employed`) instead of `metric`/`n`/`c` —
  sent through `POST /sources/{id}/observe` ("Watch a source", below) so it
  has to go through `map_schema()` for real. See point 2 just below for what
  that actually proves.

Two honest limits on what these three prove:

1. **Now included in the benchmark but with known expectations.** The 30 synthetic
   scenarios have ground truth written *before* any run, derived from how
   the panel was built. For the three real datasets, the expected outcome was
   verified by running the pipeline against independently known facts (Titanic's
   documented "women and children first" effect, the college majors' gender gap,
   airline passenger seasonality) — not circular, but not pre-registered either.
2. **Only the college-majors `_raw.csv` variant tests Qwen's judgment.**
   Flights and Titanic arrive already in the clean `metric`/`n`/`c` shape, so
   `map_schema()` never runs — they exercise the math against real noise, not
   the LLM. The `_raw.csv` columns (`Field`, `Group`, `Total`, `Employed`)
   genuinely defeat the deterministic mock heuristic (a test asserts it
   `400`s), so mapping them is a fair test of judgment. We've verified the
   verdict is correct *once mapped*; confirming Qwen maps it live needs a real
   key:
   ```bash
   DASHSCOPE_API_KEY=sk-... curl -X POST localhost:8000/sources/majors/observe -F panel=@examples/real_majors_nonstem_raw.csv
   DASHSCOPE_API_KEY=sk-... curl -X POST localhost:8000/sources/majors/observe -F panel=@examples/real_majors_stem_raw.csv
   # expect the 2nd call: ASSERT / Group=majority_women — same finding as the pre-renamed CSVs
   ```

## Calibration

Does a confidence of 0.7 mean "right ~70% of the time"? The benchmark's
`calibrate_confidence()` buckets ASSERT predictions by confidence and computes
the Expected Calibration Error — now built into `run_benchmark()`, so running
the benchmark reports ECE automatically:

```bash
python -m prove_or_abstain.benchmark
```

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
  memory.py           SQLite persistence — investigations, alerts, watched-source observations
  monitor.py          continuous autonomous surveillance loop
  reference.py        pools a watched source's prior observations into a baseline
  ingest.py           "Watch a source" entry point (ingest_and_investigate)
  investigate.py      shared state-building/graph-invocation tail (api/app.py + ingest.py)
  webhook.py          outbound notifications on EXECUTE
  cost_tracker.py     token counting + cost estimation
   benchmark.py        20 ground-truth scenarios (10 synthetic + 10 real) + cross-model eval + ECE calibration
  audit.py            reproducible, verifiable audit trails
  evidence.py         synthetic operational-event lookup, grounds ASSERT speculation
  connectors/         SQL (Postgres/MySQL/SQLite) and Google Sheets
api/                deployment entry point — FastAPI app + static demo page (SSE stream)
mcp_server.py       MCP entry point for Qwen Cloud agents
scripts/            validation & demo tooling (see below)
tests/              pytest suite (105 tests, runs offline with QWEN_MOCK=1)
examples/           sample CSVs — synthetic (planted ground truth) + 3 real public datasets
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

The package itself is pip-installable (`pip install -e .`, via `pyproject.toml`) —
`requirements.txt` stays the pinned source of truth CI/Docker both build from.

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

python -m prove_or_abstain.benchmark   # 33 ground-truth scenarios -> accuracy
python -m prove_or_abstain.monitor     # one autonomous surveillance cycle
python -m prove_or_abstain.audit       # audit trail + reproducibility check
```

## API

```
GET  /                     demo page
POST /investigate          built-in scenario: { "panel": "clean" | "diffuse" | "mixshift" | "deep", "autopilot": false, "mode": "graph" | "agent" }
GET  /investigate/stream   Server-Sent Events: stream the investigation step by step (?panel=&autopilot=)
POST /investigate/query    natural language: { "query": "why did conversion drop?", "previous_panel": "clean" }
                           or, about a watched source instead: { "query": "...", "source_id": "my-dashboard-metric" }
POST /investigate/suggest  setup helper: upload a sample CSV, get back sum-vs-rate metric classification
POST /investigate/upload   CSV upload (multipart: baseline + current)
POST /investigate/sql      live database: { "dsn": "...", "baseline_query": "...", "current_query": "..." }
POST /investigate/sheets   live Google Sheets: { "baseline_url": "...", "current_url": "..." }
POST /investigate/series   time series (multipart: series.csv + window)
POST /investigate/check    autonomous monitor — runs all panels, auto-executes on high confidence
POST /sources/{id}/observe       "Watch a source": send ONE observation, compared to its persisted history
GET  /sources/{id}/observations  read-only: the observation history accumulated for a source
GET  /panels/{name}        schema reference for SQL/Sheets/CSV
GET  /dashboard            autopilot status, active alerts, uptime
GET  /executions           audit trail of all EXECUTE actions
POST /executions/{id}/resolve  human resolves an active alert
GET  /health               healthcheck
```

Every route except `/health` is rate-limited (default 60 req/min per client
IP, `RATE_LIMIT_PER_MINUTE` env var — see `prove_or_abstain/ratelimit.py`
for what this is: a single-instance in-memory guard, not a distributed one).
Over the limit returns `429`.

Two distinct ways to feed it data: **compare two snapshots** yourself
(`/investigate*` above — a one-off "last month vs this month"), or **watch a
source** over time (`/sources/{id}/observe` — send one observation at a
time, no baseline file). See "Autonomous monitoring" below for the second
one, and "Where Qwen actually earns its keep" above for `map_schema()`,
which reshapes non-conformant columns before either path runs.

## Autonomous monitoring, persistence & audit

The Track-4 autopilot is a continuous loop, not just an endpoint. It's built
on **"Watch a source"** (`POST /sources/{source_id}/observe`): send ONE
observation, tagged with an id you choose — no baseline file. The 1st call
for a `source_id` only seeds (`{"cold_start": true, "verdict":
"BASELINE_SET"}`); every call after that is compared automatically to a
pooled window of everything already persisted for that id (`memory.py`'s
`observations` table + `reference.py`'s pooling — the same summed-counts
algebra as `panels.py::split_series`, kept z-test-valid). The baseline lives
in the database and grows with each call, so a restarted process picks its
history back up instead of losing it — unlike the explicit
`/investigate/upload`-style endpoints, which stay the right tool for a
one-off comparison with no ingestion history (the flagship PM scenario at
the top of this README).

- **`monitor.py`** — `MetricMonitor` watches a set of sources (SQL / Sheets /
  CSV / inline). Every cycle it hands the fetched panel to
  `ingest_and_investigate()` ("Watch a source", above) — no more in-process
  snapshot dict lost on restart: the reference window is pooled from
  `memory.py`'s persisted `observations` table, durable across restarts. On
  a material move it runs the investigation and, on a confident ASSERT,
  records the alert and fires the webhook. One broken feed never kills the
  loop, and a broken source never corrupts another source's history.

  ```bash
  python -m prove_or_abstain.monitor          # one demo cycle on a built-in panel
  ```

- **`memory.py`** — SQLite persistence (`PROBATIO_DB`, default `:memory:`) for
  the full investigation history, deduplicated active alerts, and every
  observation ever ingested for a watched source. `autopilot.py`
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
   monitor.py ──► ingest.py ──►  agent_loop / graph  ◄── gates decide the verdict
   /sources/{id}/observe        detector→investigate      (pure pandas/numpy)
   (ingest_and_investigate,     →verify→drill→act              │
    ≥2% triggers)                     │                        │
                 │                    │                        │
                 ▼                    ▼                        ▼
        memory.py (SQLite: investigations, alerts,       audit.py (SHA256 trail
        observations) ──► reference.py (pooled            + verify_replay)
        window, persisted across restarts)          cost_tracker.py ($/tokens)
                 │                    │
                 ▼                    ▼
     /dashboard · /executions   webhook.notify
     SSE /investigate/stream    Slack/Discord/Teams
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

Runs as a non-root user (`appuser`, uid 10001) — the process only serves
HTTP on an unprivileged port and needs no elevated access.

For Alibaba Cloud: push to Container Registry, run on Function Compute (port 8000).
`/health` serves as the probe endpoint. **Honest status of this specific
claim** — what's actually been run and verified vs. what's documented but
not yet executed — is in
[`docs/deployment_verification.md`](docs/deployment_verification.md).

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

## What's next

- OAuth-native connectors (Stripe, GA4, Amplitude) beyond the current DSN/shared-link model
- Seasonality and trend modelling for time series
- Deeper drill-down (currently one level: winning segment × one other dimension)
- Downstream actions wired to real systems (Slack alerts, feature flags, campaign pausing)
- `evidence.py`'s embedded table replaced by a real calendar/deploy-log/ticketing integration
- Multi-turn `/investigate/query` beyond a single filtered follow-up (a real conversation, not one filter)
- Configurable/adaptive `window` for "Watch a source" — today it defaults to
  pooling ALL prior observations (like `split_series`'s default), which can
  make an old baseline increasingly insensitive to a genuine recent shift on
  a long-running source; a sane default window size, or an EWMA-style decay,
  is worth revisiting once a source has real production history behind it
- `map_schema()`'s raw-source mapping doesn't yet handle a source with no
  identifiable metric column at all (single implicit metric) — out of scope
  for the current schema-mapping example, which keeps `metric` well-named on
  purpose to isolate the dims/n/c ambiguity

## License

MIT.
