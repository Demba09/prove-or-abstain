"""Verifier node (J8-J9, STRICT timebox). THE differentiator — ~80% of the signal.
Deterministic gating on explained_share + separation; the LLM only articulates the
verdict in words. Decides: assert (high/medium confidence), loop back, or ABSTAIN.

STUB: returns a non-looping placeholder verdict so the skeleton terminates.
Real gating logic (TAU_HIGH / TAU_LOW / MIN_SEPARATION + loopback) comes J8.
"""
from __future__ import annotations

from ..config import MAX_ITERATIONS, MIN_SEPARATION, TAU_HIGH, TAU_LOW  # noqa: F401


def verify(state: dict) -> dict:
    # TODO J8-J9: read state["evidence"], rank by explained_share, apply gating:
    #   share >= TAU_HIGH and separation >= MIN_SEPARATION -> assert high confidence
    #   TAU_LOW <= share < TAU_HIGH and iteration < MAX_ITERATIONS -> loop back
    #   otherwise -> abstain (verdict.abstained = True)
    state["verdict"] = {
        "top_cause": None,
        "confidence": "low",
        "explained_share": 0.0,
        "separation": 0.0,
        "supporting": [],
        "rejected": [],
        "abstained": True,
    }  # placeholder: abstain, no loop
    return state


def route(state: dict) -> str:
    """Conditional edge: 'loop' back to hypothesizer, or 'done' to the reporter."""
    verdict = state.get("verdict") or {}
    needs_more = (
        not verdict.get("abstained", True)
        and verdict.get("confidence") == "low"
        and state.get("iteration", 0) < MAX_ITERATIONS
    )
    if needs_more:
        state["iteration"] = state.get("iteration", 0) + 1
        return "loop"
    return "done"
