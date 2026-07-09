"""LangGraph assembly: 5 nodes + a bounded conditional loop.

    detector -> hypothesizer -> investigator -> verifier --(done)--> reporter -> END
                     ^                              |
                     +----------(loop)--------------+

The loop is governed by a bounded iteration counter (verifier.route), so the
back-tracking can never run away. This conditional-cycle shape is the reason
LangGraph is the right tool here rather than a linear chain.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import actuator, detector, hypothesizer, investigator, reporter, verifier
from .state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("detector", detector.detect)
    g.add_node("hypothesizer", hypothesizer.hypothesize)
    g.add_node("investigator", investigator.investigate)
    g.add_node("verifier", verifier.verify)
    g.add_node("actuator", actuator.decide)
    g.add_node("reporter", reporter.report)

    g.set_entry_point("detector")
    g.add_edge("detector", "hypothesizer")
    g.add_edge("hypothesizer", "investigator")
    g.add_edge("investigator", "verifier")

    # Conditional cycle: loop back to hypothesizer, or proceed to the decision.
    g.add_conditional_edges(
        "verifier",
        verifier.route,
        {"loop": "hypothesizer", "done": "actuator"},
    )
    g.add_edge("actuator", "reporter")
    g.add_edge("reporter", END)

    return g.compile()


def initial_state(dataset_ref: str, metric_config: dict) -> dict:
    return {
        "dataset_ref": dataset_ref,
        "metric_config": metric_config,
        "anomaly": None,
        "hypotheses": [],
        "evidence": {},
        "verdict": None,
        "action": None,
        "iteration": 0,
        "report": None,
    }
