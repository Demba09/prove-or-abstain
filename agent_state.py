"""
agent_state.py — contrat d'état du graphe Probatio (Phase 1).

AgentState est un TypedDict : c'est le schéma que LangGraph fait circuler
entre les nodes. Chaque node reçoit l'état courant et renvoie un dict partiel
des clés qu'il modifie ; LangGraph fusionne.

Aucun LLM ici (Phase 2). hypothesizer et reporter sont déterministes pour
l'instant — le LLM viendra enrichir l'ordre des hypothèses et la rédaction.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import pandas as pd

from gates import GateReport


@dataclass
class MetricAnomaly:
    """Une métrique dont le mouvement global dépasse le seuil de matérialité."""
    metric: str
    R0: float
    R1: float
    delta_rel: float        # |ΔR| / R0


@dataclass
class Action:
    """Décision typée produite par l'actuator. L'ABSTAIN ne produit jamais EXECUTE."""
    kind: str               # "EXECUTE" | "RECOMMEND" | "ESCALATE" | "NONE"
    metric: str
    dim: Optional[str] = None
    segment: Any = None
    detail: str = ""


class AgentState(TypedDict, total=False):
    # --- entrées ---
    baseline: pd.DataFrame          # panel long : [metric, *dims, n, c]
    current: pd.DataFrame
    metrics: list[str]
    metric_kinds: dict              # metric -> "rate" (défaut) | "sum"
    dims: list[str]                 # dimensions candidates, dans l'ordre d'essai
    autopilot_enabled: bool

    # --- détection ---
    anomalies: list[MetricAnomaly]
    target_metric: str              # la métrique qu'on investigue

    # --- boucle d'investigation ---
    dims_to_try: list[str]          # file des dimensions restantes
    iteration: int
    max_iterations: int
    current_dim: str
    hypothesis: str                 # texte (déterministe en Phase 1)
    dims_planned: bool              # le LLM a-t-il déjà ordonné les dimensions ?
    investigation: dict             # {out, agg, baseline_n} pour la dim courante
    gate_report: GateReport         # rapport de la dim courante
    reports_by_dim: dict            # dim -> GateReport (toutes les dims essayées)

    # --- verdict final ---
    verdict: str                    # "ASSERT" | "ABSTAIN" | "NO_ANOMALY"
    confidence: float
    winning_dim: Optional[str]
    winning_report: Optional[GateReport]

    # --- drill-down (après ASSERT : affiner au sein du segment gagnant) ---
    drilldown: Optional[dict]       # {parent, reports_by_dim, refined}

    # --- sortie ---
    actions: list[Action]
    report: str
    speculations: list[str]         # hypothèses métier LLM, étiquetées spéculation
    trace: list[str]                # journal lisible de l'investigation
