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
from panels import BASELINE, CLEAN, DEEP, DIFFUSE, MIXSHIFT, split_series

app = FastAPI(title="prove-or-abstain", version="0.4.0")

# The four demo panels: CLEAN/DEEP -> ASSERT, DIFFUSE/MIXSHIFT -> ABSTAIN.
_PANELS = {"clean": CLEAN, "diffuse": DIFFUSE, "mixshift": MIXSHIFT, "deep": DEEP}
_STATIC = Path(__file__).parent / "static"
_REQUIRED = {"metric", "n", "c"}             # mandatory long-panel columns
_RESERVED = _REQUIRED | {"period"}           # non-dimension columns


class InvestigateRequest(BaseModel):
    panel: Literal["clean", "diffuse", "mixshift", "deep"] = "clean"
    autopilot: bool = False   # only takes effect on ASSERT + confidence >= 0.70


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

    # per-segment contribution breakdown of the dimension that was decomposed
    # (the winning one on ASSERT, the last tried on ABSTAIN) — feeds the
    # waterfall chart on the demo page. Numbers only, straight from decompose().
    contributions = None
    inv = final.get("investigation")
    if inv is not None:
        out = inv["out"]
        ordered = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
        contributions = {
            "dimension": final.get("current_dim"),
            "kind": inv.get("kind", "rate"),
            "total": float(final["anomalies"][0].R1 - final["anomalies"][0].R0)
                     if final.get("anomalies") else None,
            "segments": [
                {"segment": idx,
                 "rate": float(row["rate"]),
                 "mix": float(row["mix"]),
                 "interaction": float(row["interaction"]),
                 "contribution": float(row["contribution"])}
                for idx, row in ordered.iterrows()
            ],
        }

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
        "dispatch": final.get("dispatch"),
        "contributions": contributions,
        "report": final.get("report"),
        "speculations": final.get("speculations", []),
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
        metrics=["conversion", "activation"],
        dims=["device", "segment"],
        autopilot=req.autopilot,
    )
    return {"panel": req.panel, **result}


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
    # counts must be non-negative (universal). The c <= n proportion check is
    # metric-aware and applied later (_validate_rate_counts): it holds for rate
    # metrics but NOT for sum metrics, where c is a total (revenue) that
    # legitimately exceeds n (customers).
    if (df["n"] < 0).any():
        raise HTTPException(400, f"{name}: column 'n' has negative value(s)")
    if (df["c"] < 0).any():
        raise HTTPException(400, f"{name}: column 'c' has negative value(s)")
    if not [c for c in df.columns if c not in _RESERVED]:
        raise HTTPException(400, f"{name}: needs at least one dimension column "
                                 f"besides {sorted(_REQUIRED)}")
    return df


def _validate_rate_counts(df: pd.DataFrame, name: str, kinds: dict) -> None:
    """For rate metrics only, c is a numerator bounded by the population n.
    A row with c > n reaches the z-test as an invalid proportion (a 500);
    reject it as a 400 instead. Sum metrics are exempt — there c is a total."""
    rate = df[~df["metric"].isin(k for k, v in kinds.items() if v == "sum")]
    over = rate["c"] > rate["n"]
    if over.any():
        raise HTTPException(400, f"{name}: {int(over.sum())} rate-metric row(s) "
                                 f"with c > n (numerator exceeds population)")


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
    kinds = _parse_kinds(sum_metrics, metrics)
    _validate_rate_counts(base, "baseline", kinds)
    _validate_rate_counts(curr, "current", kinds)
    result = _run_investigation(base, curr, metrics=metrics, dims=dims,
                                autopilot=autopilot, metric_kinds=kinds)
    return {"panel": "upload", **result}


@app.post("/investigate/suggest")
def investigate_suggest(file: UploadFile = File(...)) -> dict:
    """Framing aid — NOT an investigation. Given one CSV, return the
    deterministic default schema (dimensions = every column except
    metric/n/c/period) alongside an optional Qwen suggestion the user can
    ratify before running /investigate/upload. Qwen decides nothing here; the
    verdict later depends only on the confirmed inputs."""
    from llm import get_client
    df = _read_panel(file, "file")
    dims = [c for c in df.columns if c not in _RESERVED]
    metrics = sorted(df["metric"].unique())
    sample = df.head(8).to_dict(orient="records")
    suggestion = get_client().suggest_setup(list(df.columns), sample)
    return _jsonable({
        "columns": list(df.columns),
        "default": {"dimensions": dims, "sum_metrics": [], "metrics": metrics},
        "suggestion": suggestion,   # None in mock mode / on error
    })


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
    kinds = _parse_kinds(sum_metrics, metrics)
    _validate_rate_counts(base, "series (baseline window)", kinds)
    _validate_rate_counts(curr, "series (current period)", kinds)
    result = _run_investigation(base, curr, metrics=metrics, dims=dims,
                                autopilot=autopilot, metric_kinds=kinds)
    return {"panel": "series", **result}
