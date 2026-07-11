"""FastAPI surface : wrapper mince autour du graph LangGraph.

GET  /                    -> page de démo statique (api/static/index.html)
POST /investigate         -> { panel: "clean"|"diffuse"|"mixshift", autopilot?: bool }
POST /investigate/upload  -> multipart : baseline.csv + current.csv (+ autopilot)
GET  /health              -> healthcheck pour Docker / Alibaba Cloud

Les deux endpoints d'investigation construisent l'état initial, appellent
graph.invoke(state) et sérialisent l'AgentState final (verdict ASSERT/ABSTAIN,
gates par dimension, root cause, rapport, trace). Aucune logique métier ici —
tout est dans les nodes.

Format CSV attendu (panel long, mêmes colonnes que panels.py) :
    metric, <dim1>, [<dim2>, ...], n, c
Les dimensions sont inférées : toutes les colonnes sauf {metric, n, c}.

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

load_dotenv()  # local : lit .env ; en conteneur le fichier est absent (.dockerignore)
               # et les variables injectées au runtime priment.

from graph import APP as INVESTIGATION_GRAPH
from panels import BASELINE, CLEAN, DEEP, DIFFUSE, MIXSHIFT, split_series

app = FastAPI(title="prove-or-abstain", version="0.4.0")

# Les quatre panels de démo : CLEAN/DEEP -> ASSERT, DIFFUSE/MIXSHIFT -> ABSTAIN.
_PANELS = {"clean": CLEAN, "diffuse": DIFFUSE, "mixshift": MIXSHIFT, "deep": DEEP}
_STATIC = Path(__file__).parent / "static"
_REQUIRED = {"metric", "n", "c"}             # colonnes obligatoires d'un panel long
_RESERVED = _REQUIRED | {"period"}           # colonnes non-dimension


class InvestigateRequest(BaseModel):
    panel: Literal["clean", "diffuse", "mixshift", "deep"] = "clean"
    autopilot: bool = False   # ne prend effet que sur ASSERT + confiance >= 0.70


def _jsonable(v):
    """Aplati les types numpy/NaN d'un état final vers du JSON strict —
    nécessaire dès que les panels viennent d'un CSV utilisateur."""
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
    if not [c for c in df.columns if c not in _RESERVED]:
        raise HTTPException(400, f"{name}: needs at least one dimension column "
                                 f"besides {sorted(_REQUIRED)}")
    return df


def _parse_kinds(sum_metrics: str, metrics: list[str]) -> dict:
    """Champ de formulaire 'sum_metrics' : noms (séparés par des virgules) des
    métriques de type SOMME (revenu...). Les autres restent des taux."""
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
    """Un seul CSV multi-périodes (colonne 'period'). La dernière période est
    investiguée contre une baseline glissante : les `window` périodes
    précédentes poolées (toutes si window absent)."""
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
