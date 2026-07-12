"""
graph.py — assembles the seven nodes into a LangGraph StateGraph.

Topology:

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
       driller  (refines within the   │
          │      winning segment)     │
          ▼                           │
       actuator                       │
          │                           │
          ▼                           ▼
       reporter ───────────────────► END

The nodes live in nodes.py and do not import langgraph: this file is the
ONLY coupling point to the framework.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, START, END

from agent_state import AgentState
from nodes import (detector, route_after_detect,
                   hypothesizer, investigator,
                   verifier, route_after_verify,
                   driller, actuator, reporter)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("detector", detector)
    g.add_node("hypothesizer", hypothesizer)
    g.add_node("investigator", investigator)
    g.add_node("verifier", verifier)
    g.add_node("driller", driller)
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
        {"loop": "hypothesizer", "actuate": "driller"},
    )
    g.add_edge("driller", "actuator")
    g.add_edge("actuator", "reporter")
    g.add_edge("reporter", END)

    return g.compile()


APP = build_graph()
