# Project Context — prove-or-abstain

Technical context for AI agents working on this codebase. Complements AGENTS.md
(rules/invariants) with architecture rationale, design decisions, and module
responsibilities.

## Architecture overview

The project has two investigation modes that produce identical verdicts:

### Graph mode (`graph.py` + `nodes.py`)

A fixed LangGraph StateGraph with 7 nodes and conditional edges:

```
detector → hypothesizer → investigator → verifier → driller → actuator → reporter
                ↑                                        │
                └──────── loop (not asserted) ───────────┘
```

- The detector flags material anomalies (|ΔR|/R₀ ≥ 2%)
- The hypothesizer asks Qwen to ORDER candidate dimensions by likelihood
- The investigator decomposes each dimension in order
- The verifier checks the 4 gates; if all pass → ASSERT, else loop back
- The driller re-decomposes within the winning segment to refine the cause
- The actuator decides EXECUTE or ESCALATE based on confidence and autopilot
- The reporter assembles the final report (Qwen-written prose + proven numbers)

### Agent mode (`agent_loop.py`)

Qwen orchestrates via tool calls (`list_dimensions`, `test_dimension`, `drill`,
`finalize`). The mock driver tests dimensions in order; the real driver lets
Qwen decide. A determinism guard (`_finalize_verdict`) prevents false ABSTAIN
by running what the LLM skipped through the gates before finalizing.

Both modes call the same math: `detector()`, `_decompose_on()`, `evaluate_gates()`,
`driller()`, `actuator()`, `reporter()` — all in `nodes.py`.

### Shared layer (`investigate.py`)

The `_run_investigation()` function builds the state dict, dispatches to graph
or agent mode, records executions/alerts, snaps costs, and serializes. Every
API endpoint and MCP tool funnels through this function.

## The 4 gates (`gates.py`)

The ASSET/ABSTAIN decision requires ALL four gates to pass. Constants:

| Gate | Threshold | Meaning |
|------|-----------|---------|
| Material | `|ΔR|/R₀ ≥ 0.02` | The move is large enough to investigate |
| Localized | concentration ≥ 0.55 | One segment dominates the contribution |
| Significant | z-test p ≤ 0.01 (rate) or n ≥ 1000 (sum) | Not sampling noise |
| Clean | interaction ≤ 0.50 | Rate and mix effects are separable |

**For sum metrics** (revenue, etc.): the z-test is inapplicable (per-unit
variance unobservable), so the significance gate falls back to a sample-floor
check (leading segment's baseline n ≥ 1000).

Confidence is the product of three clamped factors: concentration, significance,
and mechanism cleanliness — bounded to [0,1].

## Data sources

The project supports 6 data source types, all producing a "long panel" DataFrame
with columns `[metric, dims..., n, c]`:

1. **Built-in panels** — 4 calibrated scenarios (`clean`, `diffuse`, `mixshift`, `deep`) in `panels.py`
2. **CSV upload** — user-supplied baseline/current pair via multipart form
3. **SQL** — live Postgres/MySQL/SQLite queries via SQLAlchemy (with query guards)
4. **Google Sheets** — live CSV export from public sheets (SSRF-guarded: hostname check)
5. **Time series** — multi-period CSV with rolling pooled baseline window
6. **Watch a source** — continuous ingestion with persistent observation history (SQLite)

## LLM client (`llm.py`)

Qwen client via Alibaba Cloud DashScope API. Key design points:

- **Mock mode** (`QWEN_MOCK=1`): deterministic templates replace API calls. Every
  template MUST keep the verdict path identical to a real Qwen run.
- **Singleton pattern**: `get_client()` with double-checked locking (thread-safe).
  The `_CLIENT` module global can be overridden by benchmarks/tests.
- **7 usage modes**: `plan_dimensions`, `speculate_causes`, `write_report`,
  `route_query`, `suggest_setup`, `map_schema`, `chat_with_tools`
- **JSON recovery**: `_robust_parse_json()` handles malformed LLM output.
- **Guard filters**: `_guard_filter()` and `_guarded_schema_mapping()` validate
  that Qwen's suggestions match the actual data — anti-hallucination, not a
  post-hoc check.
- **Cost tracking**: per-request token/cost snapshots via thread-local tracker.

## Persistence (`memory.py` + `autopilot.py`)

SQLite (`:memory:` for tests, file-backed for production via `PROBATIO_DB` env var):

- `memory.py` — low-level SQLite access: investigations, alerts, observations.
  All access is lock-guarded (`threading.Lock`) because `check_same_thread=False`.
- `autopilot.py` — high-level adapter: `record_execution()`, `resolve_execution()`,
  `get_dashboard()`, `record_check()`. Wraps `memory.py` for the API dashboard.

Check counters (`_TOTAL_CHECKS`, `_LAST_CHECK`, `_LAST_VERDICT`) are process-local
with a `threading.Lock`. They reset on restart — by design.

## API layer (`api/app.py`)

FastAPI with 16 endpoints. Key groupings:

- **Demo**: `GET /` (serves static HTML), `GET /panels/{name}`
- **Investigation**: `POST /investigate`, `GET /investigate/stream` (SSE, agent mode only)
- **BYOD**: `POST /investigate/upload`, `/sql`, `/sheets`, `/series`
- **Natural language**: `POST /investigate/query` (Qwen routes free-text to a panel)
- **Suggest**: `POST /investigate/suggest` (classifies metric columns as rate/sum)
- **Watch a source**: `POST /sources/{id}/observe`, `GET /sources/{id}/observations`
- **Autopilot**: `POST /investigate/check`, `GET /dashboard`, `GET /executions`,
  `POST /executions/{id}/resolve`
- **Health**: `GET /health` (exempt from rate limiting)

Middleware: rate limiting (`ratelimit.py`) — 60 req/min/IP, sliding window,
thread-safe. `/health` is exempt.

## MCP server (`mcp_server.py`)

Exposes 6 tools (`investigate_scenario`, `investigate_sql`, `autonomous_check`,
`get_dashboard`, `resolve_alert`, `describe_panels`, `describe_gates`) for Qwen
Cloud agent integration. Supports stdio (default) and SSE (`--port`) transport.

Imports `_run_investigation` from `api.app` (which re-exports from `investigate.py`).

## Thread safety (cross-module)

Modules with potential concurrent access under FastAPI's thread pool:

| Module | Global state | Guard |
|--------|-------------|-------|
| `llm.py` | `_CLIENT` singleton | `threading.Lock` (double-checked) |
| `autopilot.py` | `_TOTAL_CHECKS`, `_LAST_*` | `threading.Lock` |
| `memory.py` | `_CONN` (SQLite) | `threading.Lock` |
| `ratelimit.py` | `_HITS` dict | `threading.Lock` |

## Deployment

- **Docker**: `python:3.12-slim`, non-root user (uid 10001), HEALTHCHECK, port 8000
- **Alibaba Cloud Function Compute**: same image, `$PORT` env var support
- **Local**: `uvicorn api.app:app` on port 8000
- **CI**: GitHub Actions, Python 3.12, `pytest -q` with `QWEN_MOCK=1`

## Glossary

- **Panel**: a DataFrame in long format `[metric, dims..., n, c]`
- **Gate**: one of the 4 boolean checks that determine ASSET vs ABSTAIN
- **Concentration**: the top contributor's share of the total absolute contribution
- **Interaction**: the non-additive part of the decomposition (mix and rate changed simultaneously)
- **Drill-down**: after proving a dimension caused the anomaly, re-decompose within
  that dimension's leading segment to narrow the cause
- **Confidence**: product of clamped concentration, significance, and cleanliness factors
- **ABSTAIN**: the agent refuses to assert a cause — names the specific failing gate(s)
- **EXECUTE**: high-confidence ASSERT → autonomous action via webhook or act() target
- **ESCALATE**: ABSTAIN or low-confidence ASSERT → notify a human
