"""
evidence.py — synthetic operational log, standing in for a real external
system (a campaign calendar, a deploy log, a pricing changelog...).

This is the missing link the README's roadmap names ("downstream actions
wired to real systems") pointed the other way: instead of speculating in a
vacuum, an ASSERTed cause can be checked against whatever a business already
logs elsewhere. In this repo there is no live calendar/deploy-log
integration yet, so `_EVENTS` is a small embedded table mirroring the
built-in demo panels (prove_or_abstain/panels.py) — enough to prove the
wiring end-to-end offline. A real deployment would replace `find_events()`
with a call to Slack/Jira/a deploy log/a marketing calendar, keeping the
same (dim, segment) -> [events] contract.

Computes nothing, decides nothing: a lookup, not a verdict.
"""
from __future__ import annotations

# One row per operational event a real business would already have logged
# somewhere. Dates are illustrative, deliberately close to the built-in
# panels' "current" period so they read as plausible triggers.
_EVENTS: list[dict] = [
    {"dim": "segment", "segment": "paid", "event": "campaign_budget_cut",
     "date": "2026-07-17", "source": "ads_manager (synthetic)"},
    {"dim": "segment", "segment": "organic", "event": "search_algorithm_update",
     "date": "2026-07-16", "source": "seo_tracker (synthetic)"},
    {"dim": "device", "segment": "mobile", "event": "app_release_v4.2",
     "date": "2026-07-17", "source": "deploy_log (synthetic)"},
]


def find_events(dim: str | None, segment) -> list[dict]:
    """Events logged for exactly this (dim, segment) — [] if none, or if
    dim/segment is missing (e.g. no winning dimension on ABSTAIN)."""
    if dim is None or segment is None:
        return []
    return [e for e in _EVENTS if e["dim"] == dim and e["segment"] == segment]
