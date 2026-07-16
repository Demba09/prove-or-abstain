"""FastAPI surface: a thin wrapper around the LangGraph graph.

GET  /                    -> static demo page (api/static/index.html)
POST /investigate         -> { panel: "clean"|"diffuse"|"mixshift"|"deep", autopilot?: bool }
POST /investigate/upload  -> multipart: baseline.csv + current.csv (+ autopilot, sum_metrics)
POST /investigate/series  -> multipart: one multi-period CSV (+ window, autopilot, sum_metrics)
GET  /health              -> healthcheck for Docker / Alibaba Cloud

Both investigation endpoints build the initial state, call
graph.invoke(state) and serialize the final AgentState (ASSERT/ABSTAIN
verdict, per-dimension gates, root cause, drill-down, report, trace).
No business logic lives here — everything is in the nodes.

Expected CSV format (long panel, same columns as prove_or_abstain/panels.py):
    metric, <dim1>, [<dim2>, ...], n, c
Dimensions are inferred: every column except {metric, n, c, period}.

Run: uvicorn api.app:app --reload
"""

from __future__ import annotations

import asyncio
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from prove_or_abstain.autopilot import get_dashboard, record_check, resolve_execution, record_execution, get_executions
from prove_or_abstain.webhook import notify
from prove_or_abstain.connectors.gsheets import SheetError
from prove_or_abstain.connectors.gsheets import fetch_panel as fetch_sheet_panel
from prove_or_abstain.connectors.sql import SqlQueryError, fetch_panel as fetch_sql_panel
from prove_or_abstain.agent_loop import investigate_agentic
from prove_or_abstain.graph import APP as INVESTIGATION_GRAPH
from prove_or_abstain.llm import get_client
from prove_or_abstain.panels import BASELINE, CLEAN, DEEP, DIFFUSE, MIXSHIFT, split_series

# docs_url=None frees /docs from the built-in Swagger route so the
# redirect below can point it at ReDoc instead.
app = FastAPI(title="prove-or-abstain", version="0.4.0", docs_url=None)

# The four demo panels: CLEAN/DEEP -> ASSERT, DIFFUSE/MIXSHIFT -> ABSTAIN.
_PANELS = {"clean": CLEAN, "diffuse": DIFFUSE, "mixshift": MIXSHIFT, "deep": DEEP}
# Short descriptions handed to the LLM router (/investigate/query) — it
# SELECTS one of these, it cannot invent a panel outside this dict.
_PANEL_DESCRIPTIONS = {
    "clean": "one segment's rate collapses while everything else is stable; "
             "localizes to a single cause",
    "diffuse": "every segment drops by the same amount; no segment stands "
               "out, cause does not localize",
    "mixshift": "population mix and rates shift at the same time, entangled "
                "effects, cause does not localize cleanly",
    "deep": "a single narrow cell collapses; localizes, then a drill-down "
            "narrows it further",
}
_METRICS = ["conversion", "activation"]
_STATIC = Path(__file__).parent / "static"
_REQUIRED = {"metric", "n", "c"}             # mandatory long-panel columns
_RESERVED = _REQUIRED | {"period"}           # non-dimension columns


class InvestigateRequest(BaseModel):
    panel: Literal["clean", "diffuse", "mixshift", "deep"] = "clean"
    autopilot: bool = False   # only takes effect on ASSERT + confidence >= 0.70
    # "graph": the fixed LangGraph pipeline. "agent": Qwen orchestrates the
    # investigation via tool calls (the verdict is identical either way).
    mode: Literal["graph", "agent"] = "graph"


class QueryRequest(BaseModel):
    query: str                # free-text question, e.g. "why did conversion drop?"
    autopilot: bool = False


class SqlRequest(BaseModel):
    dsn: str                  # e.g. "postgresql://user:pass@host/db", "sqlite:///demo.db"
    baseline_query: str       # a single SELECT/WITH, long-panel shape
    current_query: str
    autopilot: bool = False
    sum_metrics: str = ""     # comma-separated SUM-kind metric names


class SheetsRequest(BaseModel):
    baseline_url: str         # a docs.google.com spreadsheet URL (or tab, via gid)
    current_url: str
    autopilot: bool = False
    sum_metrics: str = ""


def _jsonable(v):
    """Flatten numpy scalars / NaN in a final state down to strict JSON —
    necessary as soon as the panels come from a user CSV."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _run_investigation(baseline: pd.DataFrame, current: pd.DataFrame,
                       metrics: list[str], dims: list[str],
                       autopilot: bool,
                       metric_kinds: dict | None = None,
                       mode: str = "graph") -> dict:
    state = {
        "baseline": baseline,
        "current": current,
        "metrics": metrics,
        "metric_kinds": metric_kinds or {},
        "dims": dims,
        "autopilot_enabled": autopilot,
        "trace": [],
    }
    # Both paths produce the same verdict; "agent" adds Qwen's tool-call trace.
    final = investigate_agentic(state) if mode == "agent" \
        else INVESTIGATION_GRAPH.invoke(state)

    win = final.get("winning_report")
    drill = final.get("drilldown")

    actions = final.get("actions")
    if actions and actions[0].kind == "EXECUTE":
        a = actions[0]
        cause = f"{a.dim}={a.segment}" if a.dim else None
        record_execution(
            a.metric, a.dim, a.segment, final.get("confidence", 0.0),
            a.kind, a.detail,
            final.get("report", ""), final.get("trace", []),
        )
        notify(a.metric, final.get("verdict", "ASSERT"),
               final.get("confidence", 0.0),
               cause, a.kind, a.detail)

    return _jsonable({
        "verdict": final.get("verdict"),
        "confidence": final.get("confidence"),
        "root_cause": (
            {"dimension": final.get("winning_dim"), "segment": win.leading_segment}
            if win is not None
            else None
        ),
        "gates": {dim: asdict(rep) for dim, rep in final.get("reports_by_dim", {}).items()},
        "drilldown": (
            {"parent": drill["parent"],
             "refined": drill["refined"],
             "gates": {d: asdict(r) for d, r in drill["reports_by_dim"].items()}}
            if drill
            else None
        ),
        "action": asdict(final["actions"][0]) if final.get("actions") else None,
        "report": final.get("report"),
        "speculations": final.get("speculations", []),
        "llm": final.get("llm"),
        "trace": final.get("trace", []),
        "agent_trace": final.get("agent_trace", []),
    })


@app.get("/")
def home() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/docs", include_in_schema=False)
def docs() -> RedirectResponse:
    return RedirectResponse(url="/redoc")


# The built-in panels never change: serialize them once at import instead
# of a DataFrame -> records -> _jsonable pass on every request.
_PANEL_PAYLOADS = {
    name: _jsonable({
        "panel": name,
        "baseline": BASELINE.to_dict(orient="records"),
        "current": df.to_dict(orient="records"),
    })
    for name, df in _PANELS.items()
}


@app.get("/panels/{name}")
def panel_data(name: Literal["clean", "diffuse", "mixshift", "deep"]) -> dict:
    """The raw long-panel rows behind a built-in scenario — read-only,
    for inspection. This is exactly the shape a SQL query or Google Sheet
    must produce to be usable by /investigate/sql or /investigate/sheets:
    one row per (metric, <dims...>) cell, with raw counts n/c, no rates."""
    return _PANEL_PAYLOADS[name]


@app.post("/investigate")
def investigate(req: InvestigateRequest) -> dict:
    result = _run_investigation(
        BASELINE, _PANELS[req.panel],
        metrics=_METRICS,
        dims=["device", "segment"],
        autopilot=req.autopilot,
        mode=req.mode,
    )
    return {"panel": req.panel, **result}


@app.get("/investigate/stream")
async def investigate_stream(panel: Literal["clean", "diffuse", "mixshift", "deep"] = "clean",
                            autopilot: bool = False) -> StreamingResponse:
    """Server-Sent Events: stream the investigation step by step (detector →
    testing → gate_result → verdict → drill → action → done). Runs the agent
    loop in a threadpool and bridges its on_event callback to the async
    response through an asyncio.Queue."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event_type: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))

    def run() -> None:
        state = {
            "baseline": BASELINE, "current": _PANELS[panel], "metrics": _METRICS,
            "metric_kinds": {}, "dims": ["device", "segment"],
            "autopilot_enabled": autopilot, "trace": [],
        }
        try:
            investigate_agentic(state, on_event=on_event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # end sentinel

    async def gen():
        task = loop.run_in_executor(None, run)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event_type, data = item
                yield f"event: {event_type}\ndata: {json.dumps(_jsonable(data))}\n\n"
        finally:
            await task

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/investigate/query")
def investigate_query(req: QueryRequest) -> dict:
    """Free-text front-end over the built-in panels: Qwen SELECTS one of
    the four demo panels/metrics from `req.query` (never invents one —
    guarded in llm.route_query), then the same deterministic pipeline runs
    unchanged. Demonstrates the LLM boundary on a natural-language intent
    rather than a hardcoded panel name."""
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    routed = get_client().route_query(req.query, _PANEL_DESCRIPTIONS, _METRICS)
    result = _run_investigation(
        BASELINE, _PANELS[routed["panel"]],
        metrics=_METRICS,
        dims=["device", "segment"],
        autopilot=req.autopilot,
    )
    return {"panel": routed["panel"], "routing": routed, **result}


def _dataset_payload(base: pd.DataFrame, curr: pd.DataFrame) -> dict:
    """Echo the exact rows fed into the pipeline — lets the UI draw a
    baseline-vs-current chart for a data source it can't otherwise inspect
    (a SQL/Sheets result, or a just-uploaded CSV file object)."""
    return _jsonable({"baseline": base.to_dict(orient="records"),
                      "current": curr.to_dict(orient="records")})


def _validate_panel(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Long-panel contract shared by every data source (CSV, SQL, ...):
    [metric, <dims...>, n, c], n/c numeric, complete and non-negative,
    at least one dimension column."""
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise HTTPException(400, f"{name}: missing required column(s) {sorted(missing)}")
    for col in ("n", "c"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(400, f"{name}: column '{col}' must be numeric")
        if df[col].isna().any():
            raise HTTPException(400, f"{name}: column '{col}' has missing values")
        if (df[col] < 0).any():
            raise HTTPException(400, f"{name}: column '{col}' has negative values")
    if not [c for c in df.columns if c not in _RESERVED]:
        raise HTTPException(400, f"{name}: needs at least one dimension column "
                                 f"besides {sorted(_REQUIRED)}")
    return df


def _read_panel(upload: UploadFile, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(upload.file.read()))
    except Exception as exc:
        raise HTTPException(400, f"{name}: not a readable CSV ({exc})")
    return _validate_panel(df, name)


def _parse_kinds(sum_metrics: str, metrics: list[str]) -> dict:
    """Form field 'sum_metrics': comma-separated names of the SUM-kind
    metrics (revenue...). Everything else stays a rate."""
    kinds = {}
    for name in (s.strip() for s in sum_metrics.split(",") if s.strip()):
        if name not in metrics:
            raise HTTPException(400, f"sum_metrics: unknown metric '{name}'")
        kinds[name] = "sum"
    return kinds


def _validate_rate_counts(df: pd.DataFrame, name: str, kinds: dict) -> None:
    """For rate metrics, c is a count of successes out of n — c > n is a
    data error the z-test would silently turn into nonsense. Sum metrics
    (revenue: c = total amount) are exempt."""
    rate_rows = df[df["metric"].map(lambda m: kinds.get(m, "rate") == "rate")]
    bad = int((rate_rows["c"] > rate_rows["n"]).sum())
    if bad:
        raise HTTPException(400, f"{name}: {bad} row(s) have c > n on a rate metric "
                                 f"— declare sum-kind metrics via sum_metrics")


def _investigate_pair(base: pd.DataFrame, curr: pd.DataFrame,
                      base_name: str, curr_name: str,
                      autopilot: bool, sum_metrics: str) -> dict:
    """Shared tail of every bring-your-own-data endpoint (upload, SQL,
    Sheets, series): cross-validate the two panels, infer dimensions and
    metrics, and run the investigation."""
    if set(base.columns) != set(curr.columns):
        raise HTTPException(400, f"{base_name} and {curr_name} must have the same columns")
    if set(base["metric"].unique()) != set(curr["metric"].unique()):
        raise HTTPException(400, f"{base_name} and {curr_name} must cover the same metrics")

    dims = [c for c in base.columns if c not in _RESERVED]
    metrics = sorted(base["metric"].unique())
    kinds = _parse_kinds(sum_metrics, metrics)
    _validate_rate_counts(base, base_name, kinds)
    _validate_rate_counts(curr, curr_name, kinds)
    return _run_investigation(base, curr, metrics=metrics, dims=dims,
                              autopilot=autopilot, metric_kinds=kinds)


@app.post("/investigate/upload")
def investigate_upload(baseline: UploadFile = File(...),
                       current: UploadFile = File(...),
                       autopilot: bool = Form(False),
                       sum_metrics: str = Form("")) -> dict:
    base = _read_panel(baseline, "baseline")
    curr = _read_panel(current, "current")
    result = _investigate_pair(base, curr, "baseline", "current",
                               autopilot, sum_metrics)
    return {"panel": "upload", "dataset": _dataset_payload(base, curr), **result}


@app.post("/investigate/sql")
def investigate_sql(req: SqlRequest) -> dict:
    """Pull baseline/current straight from a SQL database (connectors/sql.py)
    instead of a CSV round trip: two read-only queries, each expected to
    already project onto the long-panel shape [metric, <dims...>, n, c].
    The caller supplies their own DSN/credentials — this endpoint adds no
    access beyond what that connection already grants, and restricts each
    query to a single SELECT statement."""
    try:
        base = fetch_sql_panel(req.dsn, req.baseline_query)
        curr = fetch_sql_panel(req.dsn, req.current_query)
    except SqlQueryError as exc:
        raise HTTPException(400, str(exc))

    base = _validate_panel(base, "baseline_query")
    curr = _validate_panel(curr, "current_query")
    result = _investigate_pair(base, curr, "baseline_query", "current_query",
                               req.autopilot, req.sum_metrics)
    return {"panel": "sql", "dataset": _dataset_payload(base, curr), **result}


@app.post("/investigate/sheets")
def investigate_sheets(req: SheetsRequest) -> dict:
    """Pull baseline/current straight from a Google Sheet (connectors/gsheets.py)
    instead of a CSV round trip — the sheet (or tab, via its gid) must
    already be shared as 'anyone with the link' or published to the web,
    and must already be in the long-panel shape [metric, <dims...>, n, c].
    Only docs.google.com URLs are accepted; anything else is rejected
    before any request is made."""
    try:
        base = fetch_sheet_panel(req.baseline_url)
        curr = fetch_sheet_panel(req.current_url)
    except SheetError as exc:
        raise HTTPException(400, str(exc))

    base = _validate_panel(base, "baseline_url")
    curr = _validate_panel(curr, "current_url")
    result = _investigate_pair(base, curr, "baseline_url", "current_url",
                               req.autopilot, req.sum_metrics)
    return {"panel": "sheets", "dataset": _dataset_payload(base, curr), **result}


@app.post("/investigate/series")
def investigate_series(series: UploadFile = File(...),
                       window: int | None = Form(None),
                       autopilot: bool = Form(False),
                       sum_metrics: str = Form("")) -> dict:
    """One multi-period CSV (a 'period' column). The last period is
    investigated against a rolling baseline: the preceding `window`
    periods pooled together (all of them if window is absent)."""
    panel = _read_panel(series, "series")
    if "period" not in panel.columns:
        raise HTTPException(400, "series: missing required column 'period'")

    try:
        base, curr = split_series(panel, window=window)
    except ValueError as exc:
        raise HTTPException(400, f"series: {exc}")

    result = _investigate_pair(base, curr, "series baseline", "series current",
                               autopilot, sum_metrics)
    return {"panel": "series", **result}


# ----------------------------------------------------------- autonomous autopilot
@app.post("/investigate/check")
def investigate_check() -> dict:
    """Autonomous monitoring endpoint — runs the investigation on ALL four
    built-in panels with autopilot ON. Designed to be called by a scheduler
    (cron, Alibaba Cloud SchedulerX, etc.). Returns a summary of what was
    detected and any actions taken.

    An ASSERT+EXECUTE that fires creates an execution record visible at
    GET /dashboard and POST /executions/{id}/resolve."""
    results = []
    for panel_name, panel_df in _PANELS.items():
        result = _run_investigation(
            BASELINE, panel_df,
            metrics=_METRICS,
            dims=["device", "segment"],
            autopilot=True,
        )
        result["panel"] = panel_name
        results.append(result)

    verdicts = [r["verdict"] for r in results]
    summary_verdict = "ASSERT_ACTED" if "ASSERT" in verdicts else "NO_ANOMALY"
    record_check(summary_verdict)
    return {"verdict": summary_verdict, "panels": results}


@app.get("/dashboard", include_in_schema=True)
def dashboard() -> dict:
    return asdict(get_dashboard())


@app.get("/executions", include_in_schema=True)
def list_executions() -> dict:
    ex = get_executions()
    return {"executions": [asdict(e) for e in ex.values()]}


class ResolveRequest(BaseModel):
    id: str
    resolved_by: str = Field(default="human")


@app.post("/executions/{exec_id}/resolve", include_in_schema=True)
def resolve_exec(exec_id: str, body: ResolveRequest) -> dict:
    entry = resolve_execution(exec_id, body.resolved_by)
    if entry is None:
        raise HTTPException(404, f"execution {exec_id!r} not found or already resolved")
    return {"resolved": asdict(entry)}
