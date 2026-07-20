"""
agent_loop.py — Qwen-orchestrated investigation (the "chief investigator").

A drop-in alternative to the LangGraph state machine in graph.py. Instead of a
fixed detector→hypothesizer→investigator→verifier loop, Qwen drives the
investigation through tool calls: it decides which dimension to test, in what
order, and when to stop, calling tools that run the EXACT same deterministic
math (_decompose_on + evaluate_gates) the graph uses.

Division of labour (unchanged from the rest of the project):
    Qwen orchestrates the PATH. The math decides the VERDICT.

Two guarantees keep the verdict LLM-independent, so the offline/mock result is
identical to a live Qwen run:
  1. Every gate verdict comes from evaluate_gates — Qwen cannot fabricate an
     ASSERT the tools did not return.
  2. Determinism guard (_finalize_verdict): Qwen can never cause a false
     ABSTAIN by skipping a dimension. If nothing asserted, every remaining
     dimension is tested deterministically before concluding, and the winner
     is the first dimension — in the original dims order — that asserts.

Offline (mock mode / no key / any error mid-loop) the loop is replayed
deterministically by _run_mock_loop, reproducing the graph's behaviour exactly.
The drill-down, actuation and report reuse the existing nodes verbatim.
"""
from __future__ import annotations

import json
import math

import numpy as np

from prove_or_abstain.gates import evaluate_gates, GateReport
from prove_or_abstain.llm import get_client
from prove_or_abstain.nodes import (
    _decompose_on, detector, driller, actuator, reporter,
)


def _noop(*_a, **_k) -> None:
    """Default event sink — investigate_agentic streams only if given a callback."""


# --- tool schemas exposed to Qwen (OpenAI function-calling format) ---
TOOLS = [
    {"type": "function", "function": {
        "name": "list_dimensions",
        "description": "List the candidate dimensions still available to test "
                       "and which have been tested so far.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "test_dimension",
        "description": "Decompose the anomalous metric along one dimension and "
                       "return the gate verdict (ASSERT means the cause "
                       "localizes to a segment) with concentration, "
                       "significance p-value, interaction share and confidence.",
        "parameters": {"type": "object", "properties": {
            "dimension": {"type": "string",
                          "description": "one of the candidate dimensions"},
        }, "required": ["dimension"]},
    }},
    {"type": "function", "function": {
        "name": "drill",
        "description": "After a dimension ASSERTS, refine the cause within its "
                       "leading segment along another dimension.",
        "parameters": {"type": "object", "properties": {
            "dimension": {"type": "string",
                          "description": "the dimension that asserted"},
            "sub_dimension": {"type": "string",
                              "description": "another dimension to refine along"},
            "segment": {"type": "string",
                        "description": "optional: the segment to drill into; "
                                       "defaults to the leading segment"},
        }, "required": ["dimension", "sub_dimension"]},
    }},
    {"type": "function", "function": {
        "name": "finalize",
        "description": "Conclude the investigation once enough dimensions have "
                       "been tested. The verdict is computed from the gates.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

_SYSTEM = (
    "You are the lead investigator for a business-metric anomaly. Decide which "
    "dimensions to test, and in what order, by calling test_dimension. A "
    "dimension ASSERTS only when the math localizes the cause to a segment; "
    "otherwise it ABSTAINS. After a dimension asserts you may call drill to "
    "refine the cause within its leading segment. Call finalize when you have "
    "tested enough. You choose the path — the tools compute every verdict, so "
    "never claim a cause the tools did not confirm. Be efficient: a handful of "
    "calls is enough."
)


def _py(v):
    """Coerce numpy scalars / NaN to plain JSON-safe Python."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _abstain_report(dim: str) -> GateReport:
    """A safe ABSTAIN report for a dimension whose math is unrecoverable — used
    only by the finalize sweep, so an unfixable tool error concludes safely."""
    return GateReport(
        verdict="ABSTAIN", confidence=0.0, leading_segment=None,
        concentration=0.0, interaction_share=0.0, leading_sample_n=0.0,
        leading_z=float("nan"), leading_p=float("nan"), significant=False,
        delta_R_relative=0.0, reasons=["tool_error"], subscores={})


def _emit_gate(emit, state, reports, dim: str) -> None:
    """Stream the 'testing' + 'gate_result' events for one dimension."""
    emit("testing", {"dimension": dim, "progress": f"{len(reports)}/{len(state['dims'])}"})
    rep = reports.get(dim)
    if rep is not None:
        emit("gate_result", {"dimension": dim, "verdict": rep.verdict,
                             "concentration": round(float(rep.concentration), 3),
                             "confidence": round(float(rep.confidence), 3)})


def _report_summary(label: str, rep) -> dict:
    """The compact, JSON-safe view of a GateReport handed back to Qwen."""
    return {
        "target": label,
        "verdict": rep.verdict,
        "leading_segment": _py(rep.leading_segment),
        "concentration": round(float(rep.concentration), 3),
        "significance_p": _py(round(float(rep.leading_p), 4))
        if not math.isnan(rep.leading_p) else None,
        "interaction_share": round(float(rep.interaction_share), 3),
        "confidence": round(float(rep.confidence), 3),
        "reasons": list(rep.reasons),
    }


# ------------------------------------------------------------ tool bodies
def _test_dimension(state, reports, dim: str) -> dict:
    if dim not in state["dims"]:
        return {"error": f"unknown dimension '{dim}'. "
                         f"Choose from {state['dims']}."}
    if dim not in reports:
        out, agg, kind = _decompose_on(
            state, state["baseline"], state["current"],
            state["target_metric"], dim)
        reports[dim] = evaluate_gates(agg, out, kind=kind)
    return _report_summary(dim, reports[dim])


def _drill(state, reports, args: dict) -> dict:
    dim = str(args.get("dimension", ""))
    sub = str(args.get("sub_dimension", ""))
    if dim not in reports or reports[dim].verdict != "ASSERT":
        return {"error": f"drill needs an ASSERTed dimension first; "
                         f"'{dim}' has not asserted."}
    if sub not in state["dims"] or sub == dim:
        return {"error": f"sub_dimension must be another dimension "
                         f"from {state['dims']}."}
    seg = args.get("segment") or reports[dim].leading_segment
    metric = state["target_metric"]
    base = state["baseline"]
    curr = state["current"]
    base_f = base[base[dim] == seg]
    curr_f = curr[curr[dim] == seg]
    if base_f.empty or curr_f.empty:
        return {"error": f"no rows for {dim}={seg}."}
    out, agg, kind = _decompose_on(state, base_f, curr_f, metric, sub)
    rep = evaluate_gates(agg, out, kind=kind)
    return _report_summary(f"{dim}={seg} → {sub}", rep)


def _dispatch(state, reports, name: str, args: dict) -> dict:
    if name == "list_dimensions":
        return {"metric": state["target_metric"],
                "dimensions": state["dims"],
                "tested": list(reports.keys())}
    if name == "test_dimension":
        dim = str(args.get("dimension", ""))
        try:
            return _test_dimension(state, reports, dim)
        except Exception as exc:
            # Report a tool_error to Qwen but DON'T poison reports[dim]: leaving
            # it unset lets the finalize sweep re-test it deterministically, so a
            # transient tool error can never cause a false ABSTAIN.
            return {"target": dim, "verdict": "ABSTAIN",
                    "reason": "tool_error", "error": str(exc)}
    if name == "drill":
        try:
            return _drill(state, reports, args)
        except Exception as exc:
            return {"error": str(exc), "fallback": True}
    if name == "finalize":
        return {"ok": True}
    return {"error": f"unknown tool '{name}'"}


# --------------------------------------------------------------- drivers
def _run_real_loop(client, state, reports, agent_trace, emit=_noop) -> None:
    """Qwen drives via tool calls. Raises on transport error (caller falls
    back to the deterministic driver)."""
    a = state["anomalies"][0]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": json.dumps({
            "metric": state["target_metric"],
            "relative_move": round(a.delta_rel, 4),
            "dimensions": state["dims"],
        })},
    ]
    # Cap the loop so a rambling model still terminates: every dimension plus
    # a little headroom for drilling and the finalize turn.
    for _ in range(len(state["dims"]) + 4):
        turn = client.chat_with_tools(messages, TOOLS)
        messages.append(turn["message"])
        if not turn["tool_calls"]:
            break  # a plain text turn means Qwen considers itself done
        done = False
        for tc in turn["tool_calls"]:
            # One failing tool must never kill the investigation.
            try:
                result = _dispatch(state, reports, tc["name"], tc["arguments"])
            except Exception as exc:
                result = {"error": str(exc), "fallback": True}
            agent_trace.append({
                "step": len(agent_trace) + 1,
                "tool": tc["name"],
                "arguments": tc["arguments"],
                "result": result,
            })
            if tc["name"] == "test_dimension":
                _emit_gate(emit, state, reports, str(tc["arguments"].get("dimension", "")))
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result)})
            if tc["name"] == "finalize":
                done = True
        if done:
            break


def _run_mock_loop(state, reports, agent_trace, emit=_noop) -> None:
    """Deterministic offline replay: test dimensions in the original order and
    stop at the first ASSERT — exactly what the graph does with QWEN_MOCK=1."""
    for dim in state["dims"]:
        result = _test_dimension(state, reports, dim)
        agent_trace.append({
            "step": len(agent_trace) + 1,
            "tool": "test_dimension",
            "arguments": {"dimension": dim},
            "result": result,
            "mode": "deterministic",
        })
        _emit_gate(emit, state, reports, dim)
        if reports[dim].verdict == "ASSERT":
            break


def _finalize_verdict(state, reports, agent_trace, emit=_noop) -> str | None:
    """Pick the winning dimension with the determinism guard: if nothing has
    asserted, deterministically test every remaining dimension so the LLM can
    never cause a false ABSTAIN by skipping one. Winner = the first dimension,
    in the original order, that asserts."""
    dims = state["dims"]
    if not any(r.verdict == "ASSERT" for r in reports.values()):
        for dim in dims:
            if dim in reports:
                continue
            try:
                out, agg, kind = _decompose_on(
                    state, state["baseline"], state["current"],
                    state["target_metric"], dim)
                reports[dim] = evaluate_gates(agg, out, kind=kind)
            except Exception:
                reports[dim] = _abstain_report(dim)   # unrecoverable -> safe ABSTAIN
            agent_trace.append({
                "step": len(agent_trace) + 1,
                "tool": "test_dimension",
                "arguments": {"dimension": dim},
                "result": _report_summary(dim, reports[dim]),
                "mode": "determinism-sweep",
            })
            _emit_gate(emit, state, reports, dim)
            if reports[dim].verdict == "ASSERT":
                break
    return next((d for d in dims
                 if d in reports and reports[d].verdict == "ASSERT"), None)


def investigate_agentic(state: dict, on_event=None) -> dict:
    """Run the investigation with Qwen orchestrating the path. Returns the same
    final-state shape as graph.APP.invoke(state), plus `agent_trace`.

    `on_event(event_type, data)` (optional) streams each step live — used by the
    SSE endpoint. When None, the run is silent and unchanged."""
    emit = on_event or _noop
    state = dict(state)
    state.setdefault("trace", [])

    state.update(detector(state))
    if not state.get("anomalies"):
        emit("detector", {"material": False})
        state.update(reporter(state))
        state["agent_trace"] = []
        emit("verdict", {"verdict": state.get("verdict"), "confidence": 0.0,
                         "cause": None})
        emit("done", {"verdict": state.get("verdict")})
        return state

    an = state["anomalies"][0]
    emit("detector", {"metric": state["target_metric"],
                      "delta_rel": round(an.delta_rel, 4), "material": True})

    reports: dict = {}
    agent_trace: list[dict] = []
    client = get_client()

    if client.mock:
        _run_mock_loop(state, reports, agent_trace, emit)
        client.last_mode, client.last_error = "mock", None
    else:
        try:
            _run_real_loop(client, state, reports, agent_trace, emit)
            client.last_mode, client.last_error = "real", None
        except Exception as exc:  # transport/parse error -> deterministic replay
            reports.clear()
            agent_trace.clear()
            _run_mock_loop(state, reports, agent_trace, emit)
            client.last_mode, client.last_error = "fallback", str(exc)

    winning_dim = _finalize_verdict(state, reports, agent_trace, emit)
    state["reports_by_dim"] = reports
    if winning_dim is not None:
        state["winning_dim"] = winning_dim
        state["winning_report"] = reports[winning_dim]

    # Drill-down, actuation and reporting stay deterministic — reuse the nodes.
    state.update(driller(state))
    state.update(actuator(state))
    state.update(reporter(state))
    state["agent_trace"] = agent_trace

    win = state.get("winning_report")
    emit("verdict", {
        "verdict": state.get("verdict"),
        "confidence": round(float(state.get("confidence") or 0.0), 3),
        "cause": f"{state.get('winning_dim')}={_py(win.leading_segment)}" if win else None,
    })
    refined = (state.get("drilldown") or {}).get("refined")
    if refined:
        emit("drill", {"dimension": refined["dim"], "segment": _py(refined["segment"])})
    act = (state.get("actions") or [None])[0]
    if act is not None and act.kind == "EXECUTE":
        emit("action", {"kind": act.kind, "detail": act.detail})
    emit("done", {"verdict": state.get("verdict")})
    return state
