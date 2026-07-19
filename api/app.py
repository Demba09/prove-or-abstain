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
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from prove_or_abstain import ingest, memory, reference
from prove_or_abstain.agent_loop import investigate_agentic
from prove_or_abstain.autopilot import get_dashboard, record_check, resolve_execution, get_executions
from prove_or_abstain.connectors.gsheets import SheetError
from prove_or_abstain.connectors.gsheets import fetch_panel as fetch_sheet_panel
from prove_or_abstain.connectors.sql import SqlQueryError, fetch_panel as fetch_sql_panel
from prove_or_abstain.investigate import _jsonable, _run_investigation
from prove_or_abstain.llm import get_client
from prove_or_abstain.panels import (BASELINE, CLEAN, DEEP, DIFFUSE, MIXSHIFT,
                                     DEVICES, SEGMENTS, split_series)

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
# Known segment values of the built-in panels' dimensions — handed to the LLM
# router (/investigate/query) so a follow-up ("and on mobile only?") can SELECT
# a filter among these, never invent one outside them.
_DIM_VALUES = {"device": DEVICES, "segment": SEGMENTS}
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
    # Set this to the `panel` field of a previous /investigate/query response
    # to ask a follow-up in the same conversation ("and on mobile only?")
    # instead of re-routing from scratch.
    previous_panel: Literal["clean", "diffuse", "mixshift", "deep"] | None = None
    # Set this instead of previous_panel to ask about a "Watch a source" id
    # (POST /sources/{id}/observe) rather than one of the 4 built-in panels —
    # there's exactly one dataset in play, so Qwen only extracts a filter,
    # it doesn't route. Needs >= 2 observations already recorded for the id.
    source_id: str | None = None


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
    rather than a hardcoded panel name.

    Conversational follow-up: pass `previous_panel` (the `panel` field of an
    earlier response) and a text like "and on mobile only?" — Qwen may also
    SELECT a (dim, segment) filter from the panel's known values (never
    invents one), and the same panel is filtered to it before re-running the
    unchanged pipeline. A single stateless call: the caller carries the
    conversation, not the server.

    Pass `source_id` instead of `previous_panel` to ask about a "Watch a
    source" id (see _query_source) — there's one active dataset, not 4
    named panels, so only the filter half of routing applies."""
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    if req.source_id:
        return _query_source(req)
    routed = get_client().route_query(req.query, _PANEL_DESCRIPTIONS, _METRICS,
                                      dims=_DIM_VALUES, previous_panel=req.previous_panel)
    base, curr = BASELINE, _PANELS[routed["panel"]]
    filt = routed.get("filter")
    if filt:
        base = base[base[filt["dim"]] == filt["segment"]]
        curr = curr[curr[filt["dim"]] == filt["segment"]]
    result = _run_investigation(
        base, curr,
        metrics=_METRICS,
        dims=["device", "segment"],
        autopilot=req.autopilot,
    )
    return {"panel": routed["panel"], "routing": routed, **result}


def _query_source(req: QueryRequest) -> dict:
    """Ask a free-text question about a "Watch a source" id: re-run the
    same comparison /sources/{id}/observe would have made for the latest
    observation (pooled prior observations vs. the latest one), optionally
    narrowed to a (dim, segment) Qwen extracts from the text. Needs >= 2
    recorded observations — with only 1 (cold start), there's nothing yet
    to compare, same as the /observe endpoint itself."""
    observations = memory.get_observations(req.source_id)
    if len(observations) < 2:
        raise HTTPException(
            400, f"source {req.source_id!r} has {len(observations)} observation(s) "
                f"— send at least one more before asking a question about it")
    dims = observations[-1]["dims"]
    dim_values = {d: sorted(set(pd.concat([o["panel"] for o in observations])[d].dropna()))
                 for d in dims}
    filt = get_client().extract_filter(req.query, dim_values)

    base = reference.pool_observations(observations[:-1])
    curr = observations[-1]["panel"]
    if filt:
        base = base[base[filt["dim"]] == filt["segment"]]
        curr = curr[curr[filt["dim"]] == filt["segment"]]

    result = _run_investigation(
        base, curr,
        metrics=observations[-1]["metrics"],
        dims=dims,
        autopilot=req.autopilot,
    )
    routing = {"source_id": req.source_id, "filter": filt}
    return {"source_id": req.source_id, "routing": routing, **result}


@app.post("/investigate/suggest")
def investigate_suggest(baseline: UploadFile = File(...)) -> dict:
    """Setup helper for bring-your-own-data. Dimensions need no suggestion —
    every non-reserved CSV column IS a dimension, exactly (see
    _investigate_pair below). What genuinely needs judgment is classifying
    each metric NAME as rate or sum, so Qwen does that one step; the caller
    still confirms before /investigate/upload runs anything."""
    df = _read_panel(baseline, "baseline")
    dims = [c for c in df.columns if c not in _RESERVED]
    metrics = sorted(df["metric"].unique())
    suggestion = get_client().suggest_setup(metrics)
    client = get_client()
    return {"dims": dims, "metrics": metrics, **suggestion,
            "llm": {"model": client.model, "mode": client.last_mode}}


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


# ------------------------------------------------------------ watch a source
def _read_raw_csv(upload: UploadFile, name: str) -> pd.DataFrame:
    """Like _read_panel, but WITHOUT the long-panel contract check — the
    caller decides whether the columns are already conformant or need
    map_schema() first."""
    try:
        return pd.read_csv(io.BytesIO(upload.file.read()))
    except Exception as exc:
        raise HTTPException(400, f"{name}: not a readable CSV ({exc})")


def _apply_schema_mapping(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Reshape a raw upload onto [metric, dims..., n, c] via llm.map_schema()
    — Qwen's decision is used directly (no human confirmation gate) but is
    always run through the same _validate_panel every other data source
    goes through, so an incoherent mapping is rejected before it reaches
    the math either way."""
    columns = list(df.columns)
    sample_rows = df.head(5).to_dict(orient="records")
    mapping = get_client().map_schema(columns, sample_rows)
    metric_col, n_col, c_col = (mapping.get("metric_column"),
                                mapping.get("n_column"), mapping.get("c_column"))
    if not (metric_col and n_col and c_col):
        raise HTTPException(
            400, f"{name}: could not identify metric/n/c columns among "
                f"{columns} (mapping: {mapping})")
    renamed = df.rename(columns={metric_col: "metric", n_col: "n", c_col: "c"})
    return _validate_panel(renamed, name)


@app.post("/sources/{source_id}/observe")
def observe_source(source_id: str,
                   panel: UploadFile = File(...),
                   sum_metrics: str = Form(""),
                   autopilot: bool = Form(False),
                   mode: Literal["graph", "agent"] = Form("graph"),
                   window: int | None = Form(None)) -> dict:
    """"Watch a source": send ONE observation tagged `source_id`. The first
    observation for a new source_id only seeds (cold_start=True, no
    verdict yet — nothing to compare against). Every observation after that
    is compared automatically to the pooled reference window of everything
    persisted so far for this source_id — no baseline file to supply by
    hand. Complements, not replaces, the explicit two-file endpoints above
    (/investigate/upload etc.), which remain the right tool for a one-off
    "compare these two periods" analysis with no ingestion history."""
    raw = _read_raw_csv(panel, "panel")
    if _REQUIRED - set(raw.columns):
        df = _apply_schema_mapping(raw, "panel")
    else:
        df = _validate_panel(raw, "panel")

    dims = [c for c in df.columns if c not in _RESERVED]
    metrics = sorted(df["metric"].unique())
    kinds = _parse_kinds(sum_metrics, metrics)
    _validate_rate_counts(df, "panel", kinds)

    result = ingest.ingest_and_investigate(
        source_id, df, metrics=metrics, dims=dims, metric_kinds=kinds,
        autopilot=autopilot, mode=mode, window=window)
    return result


@app.get("/sources/{source_id}/observations")
def source_observations(source_id: str, limit: int | None = None) -> dict:
    """Read-only: the persisted observation history for a watched source."""
    obs = memory.get_observations(source_id, limit=limit)
    return _jsonable({"source_id": source_id, "count": len(obs), "observations": [
        {"id": o["id"], "observed_at": o["observed_at"], "dims": o["dims"],
         "metrics": o["metrics"], "panel": o["panel"].to_dict(orient="records")}
        for o in obs
    ]})


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
