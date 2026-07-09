"""
graph.py — assemble les six nodes dans un StateGraph LangGraph.

Topologie :

        START
          │
       detector ──route_after_detect──┐
          │ "investigate"             │ "report"
          ▼                           │
     hypothesizer ◄──────┐            │
          │              │ "loop"     │
          ▼              │            │
     investigator        │            │
          │              │            │
          ▼              │            │
       verifier ─route_after_verify───┤
          │ "actuate"                 │
          ▼                           │
       actuator                       │
          │                           │
          ▼                           ▼
       reporter ───────────────────► END

Les nodes vivent dans nodes.py et n'importent pas langgraph : ce fichier
est le SEUL point de couplage au framework.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from agent_state import AgentState
from nodes import (detector, route_after_detect,
                   hypothesizer, investigator,
                   verifier, route_after_verify,
                   actuator, reporter)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("detector", detector)
    g.add_node("hypothesizer", hypothesizer)
    g.add_node("investigator", investigator)
    g.add_node("verifier", verifier)
    g.add_node("actuator", actuator)
    g.add_node("reporter", reporter)

    g.add_edge(START, "detector")
    g.add_conditional_edges(
        "detector", route_after_detect,
        {"investigate": "hypothesizer", "report": "reporter"},
    )
    g.add_edge("hypothesizer", "investigator")
    g.add_edge("investigator", "verifier")
    g.add_conditional_edges(
        "verifier", route_after_verify,
        {"loop": "hypothesizer", "actuate": "actuator"},
    )
    g.add_edge("actuator", "reporter")
    g.add_edge("reporter", END)

    return g.compile()


APP = build_graph()
