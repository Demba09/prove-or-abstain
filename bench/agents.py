"""
bench/agents.py — the two agents under test, behind one interface.

Both receive a scenario and return the same verdict shape:
    {"found": bool, "segment": str|None, "reason_gate": str|None}
so the harness can score them identically.

  our_agent   : the full prove-or-abstain graph (deterministic, offline).
  NaiveAgent  : the SAME Qwen model, handed the same per-segment numbers with a
                direct, honest prompt and structured output — explicitly allowed
                to answer "no single segment". This is the baseline a competent
                engineer would actually ship, not a strawman: if it invents
                causes anyway, that is the model's disposition, not a rigged prompt.
"""
from __future__ import annotations

import json

from graph import APP

# gate names in evaluation order (see gates.py GateReport.checks)
GATE_ORDER = ["material", "localized", "significant", "clean", "confident"]


# ------------------------------------------------------------ our agent
def _failing_gate(final) -> str | None:
    """The first gate (in order) that the best-concentrated tried dimension
    failed on — the reason our agent abstained."""
    reports = final.get("reports_by_dim", {})
    if not reports:
        return None
    best = max(reports.values(), key=lambda r: r.concentration)
    for name in GATE_ORDER:
        if not best.checks.get(name, True):
            return name
    return None


def our_agent(scenario) -> dict:
    final = APP.invoke({
        "baseline": scenario.baseline, "current": scenario.current,
        "metrics": ["conversion"], "metric_kinds": {}, "dims": scenario.dims,
        "autopilot_enabled": False, "trace": []})
    verdict = final.get("verdict")
    if verdict == "ASSERT":
        rep = final["winning_report"]
        return {"found": True, "segment": rep.leading_segment, "reason_gate": None}
    if verdict == "NO_ANOMALY":
        # nothing moved materially — the "material" criterion is why we abstain
        return {"found": False, "segment": None, "reason_gate": "material"}
    return {"found": False, "segment": None, "reason_gate": _failing_gate(final)}


# ------------------------------------------------------------ naive baseline
class NaiveAgent:
    """A plain-prompt analyst on the same Qwen. Needs a live client."""

    def __init__(self, client):
        self.client = client

    def __call__(self, scenario) -> dict:
        segs = _serialize(scenario)
        names = [s["segment"] for s in segs]
        system = (
            "You are a senior data analyst. You are given a conversion metric "
            "broken down by segment, for a baseline period and a current period "
            "(n = visitors, c = conversions). Decide whether the change is caused "
            "by ONE specific segment, or whether no single segment is responsible "
            "— because the change is spread across segments, or too small or noisy "
            "to attribute. Answer honestly: if the data does not clearly point to "
            "one segment, set found=false. Do not force a cause."
        )
        tool = {
            "type": "function",
            "function": {
                "name": "report_finding",
                "description": "Report whether one segment caused the change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "found": {"type": "boolean"},
                        "segment": {"type": ["string", "null"], "enum": names + [None]},
                        "reason": {"type": "string"},
                    },
                    "required": ["found", "segment", "reason"],
                },
            },
        }
        user = json.dumps({"segments": segs}, ensure_ascii=False)
        try:
            out = self.client.complete_tool(system, user, tool, max_tokens=250)
            found = bool(out.get("found"))
            seg = out.get("segment") if found else None
            if seg not in names:
                seg = None
            return {"found": found and seg is not None,
                    "segment": seg, "reason_gate": None}
        except Exception:
            # a broken call is scored as an abstention, not a crash
            return {"found": False, "segment": None, "reason_gate": None}


def _serialize(scenario) -> list[dict]:
    """Per-segment baseline vs current counts, the exact numbers our agent sees."""
    b = scenario.baseline.set_index("segment")
    c = scenario.current.set_index("segment")
    segs = []
    for s in b.index:
        segs.append({"segment": s,
                     "baseline_n": int(b.loc[s, "n"]), "baseline_c": int(b.loc[s, "c"]),
                     "current_n": int(c.loc[s, "n"]), "current_c": int(c.loc[s, "c"])})
    return segs
