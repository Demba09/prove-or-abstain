"""
nodes.py — les six nodes de Probatio, purs et découplés de LangGraph.

Chaque node : (state) -> dict partiel à fusionner. Aucun n'importe langgraph,
donc on peut les chaîner à la main pour tester (voir _simulate.py).

Flux :
  detector ──(anomalie ?)──> hypothesizer -> investigator -> verifier
                  │ non                              │
                  └────────────> reporter            └──(ASSERT ? sinon dim suivante)
                                                         │
                                          actuator -> reporter -> END

Pas de LLM (Phase 2). hypothesizer/reporter sont déterministes.
"""
from __future__ import annotations

from attribution import decompose
from metrics import aggregate
from gates import evaluate_gates, MATERIAL_REL
from agent_state import AgentState, MetricAnomaly, Action
from panels import metric_totals, project
from llm import get_client, template_report


def _log(state: AgentState, msg: str) -> list[str]:
    return state.get("trace", []) + [msg]


# ---------------------------------------------------------------- detector
def detector(state: AgentState) -> dict:
    """Scanne chaque métrique, repère celles dont le mouvement global est matériel."""
    base, curr = state["baseline"], state["current"]
    anomalies = []
    for m in state["metrics"]:
        n0, c0 = metric_totals(base, m)
        n1, c1 = metric_totals(curr, m)
        R0 = c0 / n0 if n0 else 0.0
        R1 = c1 / n1 if n1 else 0.0
        delta_rel = abs(R1 - R0) / (abs(R0) + 1e-12)
        if delta_rel >= MATERIAL_REL:
            anomalies.append(MetricAnomaly(m, R0, R1, delta_rel))

    anomalies.sort(key=lambda a: a.delta_rel, reverse=True)

    if not anomalies:
        return {"anomalies": [], "verdict": "NO_ANOMALY",
                "trace": _log(state, "detector : aucune anomalie matérielle.")}

    target = anomalies[0].metric
    dims = list(state["dims"])
    return {
        "anomalies": anomalies,
        "target_metric": target,
        "dims_to_try": dims,
        "iteration": 0,
        "max_iterations": len(dims),
        "reports_by_dim": {},
        "trace": _log(state,
            f"detector : anomalie sur '{target}' "
            f"(ΔR={anomalies[0].R1 - anomalies[0].R0:+.3%}, rel={anomalies[0].delta_rel:.1%}). "
            f"Dimensions à tester : {dims}."),
    }


def route_after_detect(state: AgentState) -> str:
    return "report" if not state.get("anomalies") else "investigate"


# ------------------------------------------------------------ hypothesizer
def hypothesizer(state: AgentState) -> dict:
    """Au 1er passage : le LLM PROPOSE l'ordre d'exploration (il ne décide rien).
    Puis on prend la prochaine dimension et on formule l'hypothèse."""
    queue = list(state["dims_to_try"])
    updates: dict = {}
    trace = state.get("trace", [])

    if not state.get("dims_planned"):
        a = state["anomalies"][0]
        ordered = get_client().plan_dimensions(state["target_metric"], a.delta_rel, queue)
        if ordered != queue:
            trace = trace + [f"hypothesizer : plan LLM des dimensions -> {ordered}"]
        queue = ordered
        updates["dims_planned"] = True

    dim = queue.pop(0)
    metric = state["target_metric"]
    hyp = (f"La variation de '{metric}' est portée par un segment localisé "
           f"le long de la dimension '{dim}'.")
    updates.update({
        "current_dim": dim,
        "dims_to_try": queue,
        "iteration": state.get("iteration", 0) + 1,
        "hypothesis": hyp,
        "trace": trace + [f"hypothesizer : essai dim '{dim}' — {hyp}"],
    })
    return updates


# ------------------------------------------------------------- investigator
def investigator(state: AgentState) -> dict:
    """Projette sur (métrique, dim courante) et décompose (math exacte)."""
    metric, dim = state["target_metric"], state["current_dim"]
    base_p = project(state["baseline"], metric, dim)
    curr_p = project(state["current"], metric, dim)

    out = decompose(base_p, curr_p, dims=dim)
    agg = aggregate(out)
    baseline_n = base_p.set_index(dim)["n"]

    return {
        "investigation": {"out": out, "agg": agg, "baseline_n": baseline_n},
        "trace": _log(state,
            f"investigator : décomposition sur '{dim}' "
            f"(résidu={agg['residual']:+.1e})."),
    }


# ----------------------------------------------------------------- verifier
def verifier(state: AgentState) -> dict:
    """Passe la décomposition par les gates -> verdict de la dimension courante."""
    inv = state["investigation"]
    rep = evaluate_gates(inv["agg"], inv["out"], baseline_n=inv["baseline_n"])
    dim = state["current_dim"]

    reports = dict(state.get("reports_by_dim", {}))
    reports[dim] = rep

    update = {
        "gate_report": rep,
        "reports_by_dim": reports,
        "trace": _log(state,
            f"verifier : dim '{dim}' -> {rep.verdict} "
            f"(concentration={rep.concentration:.2f}, "
            f"meneur={rep.leading_segment}, conf={rep.confidence:.2f})."),
    }
    if rep.verdict == "ASSERT":
        update.update({"winning_dim": dim, "winning_report": rep})
    return update


def route_after_verify(state: AgentState) -> str:
    """ASSERT -> on agit. Sinon on essaie la dimension suivante, dans la borne."""
    if state["gate_report"].verdict == "ASSERT":
        return "actuate"
    if state["dims_to_try"] and state["iteration"] < state["max_iterations"]:
        return "loop"
    return "actuate"


# ----------------------------------------------------------------- actuator
def actuator(state: AgentState) -> dict:
    """Mappe verdict + confiance -> Action typée. ABSTAIN n'exécute JAMAIS."""
    metric = state["target_metric"]
    win = state.get("winning_report")
    autopilot = state.get("autopilot_enabled", False)

    if win is not None:  # au moins une dimension a localisé -> ASSERT
        verdict, conf = "ASSERT", win.confidence
        seg, dim = win.leading_segment, state.get("winning_dim")
        if conf >= 0.70 and autopilot:
            action = Action("EXECUTE", metric, dim, seg,
                            f"Autopilot : action ciblée sur {dim}={seg} (conf {conf:.2f}).")
        else:
            why = "autopilot désactivé" if conf >= 0.70 else f"confiance {conf:.2f} < 0.70"
            action = Action("RECOMMEND", metric, dim, seg,
                            f"Cause localisée {dim}={seg} ; recommandation seule ({why}).")
    else:  # aucune dimension n'a localisé -> ABSTAIN
        verdict, conf = "ABSTAIN", 0.0
        # on remonte la meilleure tentative pour expliquer le refus
        best = max(state.get("reports_by_dim", {}).values(),
                   key=lambda r: r.concentration, default=None)
        detail = "aucune dimension testée ne localise une cause."
        if best is not None:
            detail += f" Meilleure concentration={best.concentration:.2f} (< seuil)."
        action = Action("ESCALATE", metric, None, None, detail)

    return {
        "verdict": verdict,
        "confidence": conf,
        "actions": [action],
        "trace": _log(state, f"actuator : {verdict} -> action {action.kind}."),
    }


# ----------------------------------------------------------------- reporter
def reporter(state: AgentState) -> dict:
    """Synthèse lisible. Le LLM REFORMULE des chiffres déjà calculés ; en mock
    ou en cas d'erreur, on retombe sur la rédaction déterministe (template)."""
    verdict = state.get("verdict", "NO_ANOMALY")
    metric = state.get("target_metric", "—")

    if verdict == "NO_ANOMALY":
        payload = {"verdict": "NO_ANOMALY", "metric": metric}
    else:
        action = state["actions"][0]
        payload = {"verdict": verdict, "metric": metric,
                   "action_kind": action.kind, "action_detail": action.detail}
        if verdict == "ASSERT":
            rep = state["winning_report"]
            payload.update({
                "winning_dim": state["winning_dim"],
                "leading_segment": rep.leading_segment,
                "concentration": rep.concentration,
                "confidence": rep.confidence,
            })
        else:  # ABSTAIN
            reports = state.get("reports_by_dim", {})
            best = max((r.concentration for r in reports.values()), default=0.0)
            payload.update({
                "dims_tried": list(reports.keys()),
                "best_concentration": best,
            })

    text = get_client().write_report(payload)
    return {"report": text, "trace": _log(state, f"reporter : {verdict} rédigé.")}
