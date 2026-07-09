"""Actuator node — the terminal decision (replaces the passive report-only end).

Maps the verifier's verdict onto a typed Action:
  - asserted cause           -> a scoped recommendation
  - asserted + high conf     -> may auto-execute, but ONLY if AUTOPILOT_ENABLED
  - abstained                -> escalate to a human, execute nothing

The safety invariant: an autopilot that can act must be able to refuse to act.
Abstention always yields requires_human=True and executed=False. No LLM computes
anything here; this is deterministic mapping from an already-computed verdict.

The actual side effect is delegated to a pluggable actuator. The default one just
writes a structured decision artifact to data/decisions/ — honest and demoable.
Real targets (Slack, Jira, Stripe, ad platforms) are roadmap, behind the same
interface.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import AUTOPILOT_ENABLED


def _default_actuator(action: dict) -> None:
    """Demo actuator: persist the decision as a visible artifact. Swap this for a
    Slack/Jira/etc. dispatcher behind the same single-call interface."""
    out_dir = Path("data") / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = (action.get("target") or "none").replace("=", "_").replace("/", "_")
    path = out_dir / f"action_{action['type']}_{target}.json"
    path.write_text(json.dumps(action, indent=2))


def decide(state: dict, actuator=_default_actuator) -> dict:
    verdict = state.get("verdict") or {}
    abstained = verdict.get("abstained", True)
    confidence = verdict.get("confidence", "low")
    top_cause = verdict.get("top_cause")

    if abstained or top_cause is None:
        action = {
            "type": "escalate_to_human",
            "target": None,
            "rationale": "Evidence insufficient to attribute a cause. Handing back "
                         "to a human rather than acting on an unproven diagnosis.",
            "payload": {
                "explained_share": verdict.get("explained_share", 0.0),
                "separation": verdict.get("separation", 0.0),
                "rejected": verdict.get("rejected", []),
            },
            "requires_human": True,
            "executed": False,
        }
    else:
        action = {
            "type": "recommend",
            "target": top_cause,
            "rationale": f"Attributed cause: {top_cause}. Recommended scoped action "
                         f"with confidence={confidence}.",
            "payload": {
                "explained_share": verdict.get("explained_share", 0.0),
                "supporting": verdict.get("supporting", []),
            },
            "requires_human": confidence != "high",
            "executed": False,
        }
        # Auto-execute only under explicit autopilot AND high confidence.
        if AUTOPILOT_ENABLED and confidence == "high":
            action["executed"] = True
            actuator(action)

    state["action"] = action
    return state
