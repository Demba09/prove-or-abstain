"""
bench/metrics.py — the four numbers that decide the pitch.

Given a list of (scenario, verdict) for one agent, compute:

  cause_invention_rate : of the no-cause scenarios, the fraction where the agent
                         claimed a cause anyway. THE headline number — low is good.
  precision            : of the times the agent claimed a cause, the fraction that
                         were real AND named the right segment.
  recall               : of the truly localized scenarios, the fraction the agent
                         caught (right segment). The price of being cautious.
  abstention_reason_accuracy : of the no-cause scenarios the agent correctly
                         declined, the fraction where it named a criterion that
                         GENUINELY failed for that case (not a fabricated one).
                         Our agent points at a real failing check; a plain
                         prompt, having no checks, can name none — that gap is
                         the point, not the absolute figure.
"""
from __future__ import annotations


def compute_metrics(pairs: list[tuple]) -> dict:
    """pairs: list of (scenario, verdict_dict)."""
    no_cause = [(s, v) for s, v in pairs if not s.should_assert]
    localized = [(s, v) for s, v in pairs if s.should_assert]
    asserted = [(s, v) for s, v in pairs if v["found"]]

    invented = sum(1 for s, v in no_cause if v["found"])
    invention_rate = invented / len(no_cause) if no_cause else 0.0

    correct_asserts = sum(1 for s, v in asserted
                          if s.should_assert and v["segment"] == s.cause)
    precision = correct_asserts / len(asserted) if asserted else None

    caught = sum(1 for s, v in localized if v["found"] and v["segment"] == s.cause)
    recall = caught / len(localized) if localized else None

    declined = [(s, v) for s, v in no_cause if not v["found"]]
    reason_hits = sum(1 for s, v in declined
                      if v.get("reason_gate") in s.expected_reasons)
    scoreable = [(s, v) for s, v in declined if s.expected_reasons]
    reason_acc = (sum(1 for s, v in scoreable
                      if v.get("reason_gate") in s.expected_reasons) / len(scoreable)
                  if scoreable else None)

    return {
        "n_scenarios": len(pairs),
        "n_no_cause": len(no_cause),
        "n_localized": len(localized),
        "cause_invention_rate": invention_rate,
        "invented_count": invented,
        "precision": precision,
        "recall": recall,
        "abstention_reason_accuracy": reason_acc,
    }
