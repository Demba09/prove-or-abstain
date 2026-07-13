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

Expected CSV format (long panel, same columns as panels.py):
    metric, <dim1>, [<dim2>, ...], n, c
Dimensions are inferred: every column except {metric, n, c, period}.

Run: uvicorn api.app:app --reload
"""

from __future__ import annotations

import io
import math
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()  # local: reads .env; in the container the file is absent
               # (.dockerignore) and runtime-injected variables win anyway.

from graph import APP as INVESTIGATION_GRAPH
from llm import get_client
from panels import BASELINE, CLEAN, DEEP, DIFFUSE, MIXSHIFT, split_series

app = FastAPI(title="prove-or-abstain", version="0.4.0")

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


class QueryRequest(BaseModel):
    query: str                # free-text question, e.g. "why did conversion drop?"
    autopilot: bool = False


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
                       metric_kinds: dict | None = None) -> dict:
    state = {
        "baseline": baseline,
        "current": current,
        "metrics": metrics,
        "metric_kinds": metric_kinds or {},
        "dims": dims,
        "autopilot_enabled": autopilot,
        "trace": [],
    }
    final = INVESTIGATION_GRAPH.invoke(state)

    win = final.get("winning_report")
    drill = final.get("drilldown")
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
    })


@app.get("/")
def home() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/investigate")
def investigate(req: InvestigateRequest) -> dict:
    result = _run_investigation(
        BASELINE, _PANELS[req.panel],
        metrics=_METRICS,
        dims=["device", "segment"],
        autopilot=req.autopilot,
    )
    return {"panel": req.panel, **result}


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


def _read_panel(upload: UploadFile, name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(upload.file.read()))
    except Exception as exc:
        raise HTTPException(400, f"{name}: not a readable CSV ({exc})")
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise HTTPException(400, f"{name}: missing required column(s) {sorted(missing)}")
    for col in ("n", "c"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(400, f"{name}: column '{col}' must be numeric")
    if not [c for c in df.columns if c not in _RESERVED]:
        raise HTTPException(400, f"{name}: needs at least one dimension column "
                                 f"besides {sorted(_REQUIRED)}")
    return df


def _parse_kinds(sum_metrics: str, metrics: list[str]) -> dict:
    """Form field 'sum_metrics': comma-separated names of the SUM-kind
    metrics (revenue...). Everything else stays a rate."""
    kinds = {}
    for name in (s.strip() for s in sum_metrics.split(",") if s.strip()):
        if name not in metrics:
            raise HTTPException(400, f"sum_metrics: unknown metric '{name}'")
        kinds[name] = "sum"
    return kinds


@app.post("/investigate/upload")
def investigate_upload(baseline: UploadFile = File(...),
                       current: UploadFile = File(...),
                       autopilot: bool = Form(False),
                       sum_metrics: str = Form("")) -> dict:
    base = _read_panel(baseline, "baseline")
    curr = _read_panel(current, "current")

    if set(base.columns) != set(curr.columns):
        raise HTTPException(400, "baseline and current must have the same columns")
    if set(base["metric"].unique()) != set(curr["metric"].unique()):
        raise HTTPException(400, "baseline and current must cover the same metrics")

    dims = [c for c in base.columns if c not in _RESERVED]
    metrics = sorted(base["metric"].unique())
    result = _run_investigation(base, curr, metrics=metrics, dims=dims,
                                autopilot=autopilot,
                                metric_kinds=_parse_kinds(sum_metrics, metrics))
    return {"panel": "upload", **result}


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

    dims = [c for c in base.columns if c not in _RESERVED]
    metrics = sorted(base["metric"].unique())
    result = _run_investigation(base, curr, metrics=metrics, dims=dims,
                                autopilot=autopilot,
                                metric_kinds=_parse_kinds(sum_metrics, metrics))
    return {"panel": "series", **result}
