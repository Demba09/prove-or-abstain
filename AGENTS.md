# AGENTS.md — prove-or-abstain

Instructions for AI coding agents (Claude Code, OpenCode, Cursor, Copilot, etc.)
working on this project. Follow these rules to avoid breaking safety guarantees.

## Non-negotiable invariants

1. **The LLM never computes a number.** Every numeric value in the output
   comes from `pandas`/`numpy`. Qwen orders dimensions, writes prose, routes
   queries, and maps schemas — nothing else. If a change moves a computation
   into `llm.py` from `attribution.py`/`gates.py`/`nodes.py`, it's wrong.

2. **ABSTAIN → no EXECUTE.** The safety property is enforced at the type
   level: `ABSTAIN` verdicts cannot produce an `EXECUTE` action. Never add
   a code path that bypasses this. `agent_loop.py`'s `_finalize_verdict()` is
   the guard that prevents a lazy/erratic Qwen from producing a false
   ABSTAIN — do not weaken or remove it.

3. **Mock mode produces identical verdicts.** Running with `QWEN_MOCK=1`
   must yield the same ASSERT/ABSTAIN outcome (same root cause, same
   confidence) as a real Qwen run. The mock templates in `llm.py` exist to
   guarantee this. If you add a new LLM call, add a mock fallback.

4. **The decomposition is exact (zero residual).** `attribution.py`'s
   rate/mix/interaction decomposition is validated against
   `scripts/attribution_reference.py`. Any change to the math MUST keep
   `test_decompose_matches_oracle` and `test_decompose_zero_residual` green.

5. **Qwen drives the path, gates decide the outcome.** The agent loop lets
   Qwen choose which dimension to test first via tool calls — but the gates
   in `gates.py` are pure computation with no LLM input. The benchmark
   proves identical verdicts across graph and agent modes.

## Commands

```bash
# Tests (always offline — no API key needed)
QWEN_MOCK=1 pytest -q

# Lint
pip install -r requirements-dev.txt && ruff check .

# Benchmark (20 scenarios, both modes — 10 synthetic + 10 real)
python -m prove_or_abstain.benchmark

# Run the API locally
QWEN_MOCK=1 uvicorn api.app:app --reload

# Docker
docker build -t prove-or-abstain .
docker run --rm -p 8000:8000 -e QWEN_MOCK=1 prove-or-abstain
```

## Coding conventions

- Python 3.12+ (f-strings, `str | None` unions, PEP 695 type parameters)
- Use `from __future__ import annotations` in every file
- Use `os.environ.get("KEY")` for secrets — never hardcode keys
- Thread-shared globals MUST use `threading.Lock` (see `llm.py`, `autopilot.py`, `ratelimit.py`)
- SQLite access goes through `memory.py`'s lock (do not bypass)
- New API endpoints go in `api/app.py`; new math stays in `prove_or_abstain/`
- `requirements.txt` is the pinned source of truth; `pyproject.toml` reads from it
- Don't add Markdown/docs files unless asked

## Before each commit

1. `QWEN_MOCK=1 pytest -q` must pass (112 tests)
2. `ruff check .` must pass (0 errors)
3. Benchmark must stay at 100% accuracy, 0% false-ASSERT
4. Commit message in English, one coherent change per commit

## What NOT to do

- Do NOT move math into the LLM (e.g., letting Qwen compute z-scores)
- Do NOT remove the ABSTAIN/EXECUTE guard in `agent_loop.py`
- Do NOT add unbounded loops — the max iteration is `len(dims) + headroom`
- Do NOT expose DASHSCOPE_API_KEY, WEBHOOK_URL, or DSN strings in logs/output
- Do NOT skip the mock fallback for any new LLM call

## Directory structure

```
prove_or_abstain/       # core library — math, gates, LLM client, connectors
  gates.py              #   ASSERT/ABSTAIN decision with 4 gates
  attribution.py        #   exact rate/mix/interaction decomposition
  nodes.py              #   7 LangGraph nodes (graph mode)
  graph.py              #   StateGraph assembly (graph mode)
  agent_loop.py         #   Qwen-orchestrated tool-calling loop (agent mode)
  investigate.py        #   shared serialization layer for both modes
  llm.py                #   Qwen/DashScope client with mock mode
  monitor.py            #   continuous autonomous surveillance
  autopilot.py          #   execution/alert tracker (SQLite-backed)
  memory.py             #   SQLite persistence with thread-locked access
  ratelimit.py          #   in-memory rate limiter (60 req/min/IP)
  webhook.py            #   Slack/Discord/Teams notifications
  connectors/           #   live data sources (SQL, Google Sheets)
  panels.py             #   4 built-in demo scenarios + pooling utilities

api/                    # FastAPI REST layer
  app.py                #   16 endpoints, demo page serve, middleware

tests/                  # 112 tests (single file, 28 marked sections)
scripts/                # validation scripts, oracle, manual checks
docs/                   # devpost, demo script, architecture diagram

mcp_server.py           # MCP server for Qwen Cloud agent integration
Dockerfile              # python:3.12-slim, non-root user, HEALTHCHECK
pyproject.toml          # PEP 621 metadata (deps from requirements.txt)
requirements.txt        # pinned dependencies, source of truth for CI/Docker
```
