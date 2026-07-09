"""FastAPI surface (Phase 8) : wrapper mince autour du graph LangGraph.

POST /investigate  { panel: "clean" | "diffuse" | "mixshift", autopilot?: bool }
    -> graph.invoke(state), puis sérialise l'AgentState final :
       verdict ASSERT/ABSTAIN, gates par dimension, root cause, rapport, trace.
GET  /health       -> healthcheck pour Docker / Alibaba Cloud.

Aucune logique métier ici — tout est dans les nodes.

Run: uvicorn api.app:app --reload
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()  # local : lit .env ; en conteneur le fichier est absent (.dockerignore)
               # et les variables injectées au runtime priment.

from graph import APP as INVESTIGATION_GRAPH
from panels import BASELINE, CLEAN, DIFFUSE, MIXSHIFT

app = FastAPI(title="prove-or-abstain", version="0.2.0")

# Les trois panels de démo : CLEAN -> ASSERT, DIFFUSE/MIXSHIFT -> ABSTAIN.
_PANELS = {"clean": CLEAN, "diffuse": DIFFUSE, "mixshift": MIXSHIFT}


class InvestigateRequest(BaseModel):
    panel: Literal["clean", "diffuse", "mixshift"] = "clean"
    autopilot: bool = False   # ne prend effet que sur ASSERT + confiance >= 0.70


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/investigate")
def investigate(req: InvestigateRequest) -> dict:
    state = {
        "baseline": BASELINE,
        "current": _PANELS[req.panel],
        "metrics": ["conversion", "activation"],
        "dims": ["device", "segment"],
        "autopilot_enabled": req.autopilot,
        "trace": [],
    }
    final = INVESTIGATION_GRAPH.invoke(state)

    win = final.get("winning_report")
    return {
        "panel": req.panel,
        "verdict": final.get("verdict"),
        "confidence": final.get("confidence"),
        "root_cause": (
            {"dimension": final.get("winning_dim"), "segment": win.leading_segment}
            if win is not None
            else None
        ),
        "gates": {dim: asdict(rep) for dim, rep in final.get("reports_by_dim", {}).items()},
        "action": asdict(final["actions"][0]) if final.get("actions") else None,
        "report": final.get("report"),
        "trace": final.get("trace", []),
    }
