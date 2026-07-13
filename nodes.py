"""
nodes.py — the seven Probatio nodes, pure functions decoupled from LangGraph.

Each node: (state) -> partial dict to merge. None of them imports langgraph,
so they can be chained by hand for testing (see simulate.py).

Flow:
  detector ──(anomaly?)──> hypothesizer -> investigator -> verifier
                  │ no                               │
                  └────────────> reporter            └──(ASSERT? else next dim)
                                                         │
                              driller -> actuator -> reporter -> END

Division of labour: every number comes from pandas/numpy. The LLM only
orders the dimensions to try, phrases the report, and offers clearly
labelled speculation — it never computes and never decides a verdict.
"""
from __future__ import annotations

import time

from attribution import decompose, decompose_sum
from metrics import aggregate
from gates import evaluate_gates, MATERIAL_REL
from agent_state import AgentState, MetricAnomaly, Action
from panels import metric_totals, project
from llm import get_client, template_report


def _log(state: AgentState, msg: str) -> list[str]:
    return state.get("trace", []) + [msg]


def _kind(state: AgentState, metric: str) -> str:
    """"rate" (default) or "sum" — selects the attribution math and the
    significance gate variant."""
    return state.get("metric_kinds", {}).get(metric, "rate")


# ---------------------------------------------------------------- detector
def detector(state: AgentState) -> dict:
    """Scan every metric; flag the ones whose aggregate move is material."""
    base, curr = state["baseline"], state["current"]
    anomalies = []
    for m in state["metrics"]:
        n0, c0 = metric_totals(base, m)
        n1, c1 = metric_totals(curr, m)
        if _kind(state, m) == "sum":
            R0, R1 = c0, c1          # a sum compares directly
        else:
            R0 = c0 / n0 if n0 else 0.0
            R1 = c1 / n1 if n1 else 0.0
        delta_rel = abs(R1 - R0) / (abs(R0) + 1e-12)
        if delta_rel >= MATERIAL_REL:
            anomalies.append(MetricAnomaly(m, R0, R1, delta_rel))

    anomalies.sort(key=lambda a: a.delta_rel, reverse=True)

    if not anomalies:
        return {"anomalies": [], "verdict": "NO_ANOMALY",
                "trace": _log(state, "detector: no material anomaly.")}

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
            f"detector: anomaly on '{target}' "
            f"(ΔR={anomalies[0].R1 - anomalies[0].R0:+.3%}, rel={anomalies[0].delta_rel:.1%}). "
            f"Dimensions to test: {dims}."),
    }


def route_after_detect(state: AgentState) -> str:
    return "report" if not state.get("anomalies") else "investigate"


# ------------------------------------------------------------ hypothesizer
def hypothesizer(state: AgentState) -> dict:
    """On the first pass the LLM PROPOSES the exploration order (it decides
    nothing). Then take the next dimension and phrase the hypothesis."""
    queue = list(state["dims_to_try"])
    updates: dict = {}
    trace = state.get("trace", [])

    if not state.get("dims_planned"):
        a = state["anomalies"][0]
        client = get_client()
        t0 = time.perf_counter()
        ordered = client.plan_dimensions(state["target_metric"], a.delta_rel, queue)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not client.mock:
            trace = trace + [
                f"hypothesizer: Qwen ({client.model}) ranked dimensions in "
                f"{elapsed_ms:.0f} ms -> {ordered}"]
        elif ordered != queue:
            trace = trace + [f"hypothesizer: LLM dimension plan -> {ordered}"]
        queue = ordered
        updates["dims_planned"] = True

    dim = queue.pop(0)
    metric = state["target_metric"]
    hyp = (f"The move in '{metric}' is driven by a localized segment "
           f"along the '{dim}' dimension.")
    updates.update({
        "current_dim": dim,
        "dims_to_try": queue,
        "iteration": state.get("iteration", 0) + 1,
        "hypothesis": hyp,
        "trace": trace + [f"hypothesizer: trying dim '{dim}' — {hyp}"],
    })
    return updates


# ------------------------------------------------------------- investigator
def _decompose_on(state: AgentState, base, curr, metric: str, dim: str):
    """Project (metric, dim) and decompose with the metric-kind's math.
    Returns (out, agg, kind) — shared by investigator and driller."""
    kind = _kind(state, metric)
    base_p = project(base, metric, dim)
    curr_p = project(curr, metric, dim)
    fn = decompose_sum if kind == "sum" else decompose
    out = fn(base_p, curr_p, dims=dim)
    return out, aggregate(out), kind


def investigator(state: AgentState) -> dict:
    """Project on (metric, current dim) and decompose (exact math)."""
    metric, dim = state["target_metric"], state["current_dim"]
    out, agg, kind = _decompose_on(state, state["baseline"], state["current"],
                                   metric, dim)

    return {
        "investigation": {"out": out, "agg": agg, "kind": kind},
        "trace": _log(state,
            f"investigator: decomposition along '{dim}' "
            f"(residual={agg['residual']:+.1e})."),
    }


# ----------------------------------------------------------------- verifier
def verifier(state: AgentState) -> dict:
    """Run the decomposition through the gates -> verdict for this dimension."""
    inv = state["investigation"]
    rep = evaluate_gates(inv["agg"], inv["out"], kind=inv.get("kind", "rate"))
    dim = state["current_dim"]

    reports = dict(state.get("reports_by_dim", {}))
    reports[dim] = rep

    update = {
        "gate_report": rep,
        "reports_by_dim": reports,
        "trace": _log(state,
            f"verifier: dim '{dim}' -> {rep.verdict} "
            f"(concentration={rep.concentration:.2f}, "
            f"leader={rep.leading_segment}, conf={rep.confidence:.2f})."),
    }
    if rep.verdict == "ASSERT":
        update.update({"winning_dim": dim, "winning_report": rep})
    return update


def route_after_verify(state: AgentState) -> str:
    """ASSERT -> act. Otherwise try the next dimension, within the bound."""
    if state["gate_report"].verdict == "ASSERT":
        return "actuate"
    if state["dims_to_try"] and state["iteration"] < state["max_iterations"]:
        return "loop"
    return "actuate"


# ------------------------------------------------------------------ driller
def driller(state: AgentState) -> dict:
    """Drill-down after an ASSERT: filter the panel to the winning segment and
    re-decompose along the OTHER dimensions to refine the localization.
    One level, bounded by len(dims) — and a no-op on ABSTAIN."""
    win = state.get("winning_report")
    if win is None:
        return {"drilldown": None}

    metric = state["target_metric"]
    dim, seg = state["winning_dim"], win.leading_segment
    base = state["baseline"]
    curr = state["current"]
    base_f = base[base[dim] == seg]
    curr_f = curr[curr[dim] == seg]

    sub_reports, refined = {}, None
    for other in state["dims"]:
        if other == dim:
            continue
        out, agg, kind = _decompose_on(state, base_f, curr_f, metric, other)
        rep = evaluate_gates(agg, out, kind=kind)
        sub_reports[other] = rep
        if rep.verdict == "ASSERT" and refined is None:
            refined = {"dim": other, "segment": rep.leading_segment,
                       "concentration": rep.concentration,
                       "confidence": rep.confidence}

    if refined:
        msg = (f"driller: within {dim}={seg}, the cause narrows to "
               f"{refined['dim']}={refined['segment']} "
               f"(concentration {refined['concentration']:.2f}).")
    else:
        msg = (f"driller: within {dim}={seg}, no sub-dimension localizes "
               f"further — the whole segment is affected.")

    return {
        "drilldown": {"parent": {"dim": dim, "segment": seg},
                      "reports_by_dim": sub_reports, "refined": refined},
        "trace": _log(state, msg),
    }


# ----------------------------------------------------------------- actuator
def actuator(state: AgentState) -> dict:
    """Map verdict + confidence to a typed Action. ABSTAIN NEVER executes."""
    metric = state["target_metric"]
    win = state.get("winning_report")
    autopilot = state.get("autopilot_enabled", False)

    if win is not None:  # at least one dimension localized -> ASSERT
        verdict, conf = "ASSERT", win.confidence
        seg, dim = win.leading_segment, state.get("winning_dim")
        target = f"{dim}={seg}"
        refined = (state.get("drilldown") or {}).get("refined")
        if refined:
            target += f" ∧ {refined['dim']}={refined['segment']}"
        if conf >= 0.70 and autopilot:
            action = Action("EXECUTE", metric, dim, seg,
                            f"Autopilot: scoped action on {target} (conf {conf:.2f}).")
        else:
            why = "autopilot disabled" if conf >= 0.70 else f"confidence {conf:.2f} < 0.70"
            action = Action("RECOMMEND", metric, dim, seg,
                            f"Localized cause {target}; recommendation only ({why}).")
    else:  # no dimension localized -> ABSTAIN
        verdict, conf = "ABSTAIN", 0.0
        # surface the best attempt to explain the refusal
        best = max(state.get("reports_by_dim", {}).values(),
                   key=lambda r: r.concentration, default=None)
        detail = "no tested dimension localizes a cause."
        if best is not None:
            detail += f" Best concentration={best.concentration:.2f} (below threshold)."
        action = Action("ESCALATE", metric, None, None, detail)

    return {
        "verdict": verdict,
        "confidence": conf,
        "actions": [action],
        "trace": _log(state, f"actuator: {verdict} -> action {action.kind}."),
    }


# ----------------------------------------------------------------- reporter
def reporter(state: AgentState) -> dict:
    """Readable synthesis. The LLM REPHRASES numbers that are already computed;
    in mock mode or on any error it falls back to the deterministic template.
    On ASSERT the LLM adds BUSINESS hypotheses labelled as speculation —
    kept apart from the proven verdict, never mixed with the numbers."""
    verdict = state.get("verdict", "NO_ANOMALY")
    metric = state.get("target_metric", "—")
    speculations: list[str] = []

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
            refined = (state.get("drilldown") or {}).get("refined")
            if refined:
                payload["refined"] = f"{refined['dim']}={refined['segment']}"
            speculations = get_client().speculate_causes(payload)
        else:  # ABSTAIN
            reports = state.get("reports_by_dim", {})
            best = max((r.concentration for r in reports.values()), default=0.0)
            payload.update({
                "dims_tried": list(reports.keys()),
                "best_concentration": best,
            })

    text = get_client().write_report(payload)
    return {"report": text, "speculations": speculations,
            "trace": _log(state, f"reporter: {verdict} written.")}
