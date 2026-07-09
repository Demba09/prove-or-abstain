"""Validation tests for the demo datasets.

These are the safety net for the whole demo. If scenario 1 stops being
localizable or scenario 2 stops being diffuse, attribution and abstention both
break — and these tests fail loudly before that reaches the judges.

Run: pytest -q   (or: python -m tests.test_datagen)
"""

from __future__ import annotations

from investigator import datagen
from investigator.attribution import decompose_rate_metric, aggregate_rate

DIMENSIONS = ["plan", "channel", "region", "cohort"]


def _scan(df):
    """Return (max_explained_share, max_rate_move) across all dimensions/segments."""
    max_share = 0.0
    max_move = 0.0
    for dim in DIMENSIONS:
        d = decompose_rate_metric(df, dim)
        max_share = max(max_share, d["explained_share"].abs().max())
        max_move = max(max_move, (d["rate_after"] - d["rate_before"]).abs().max())
    return max_share, max_move


def test_both_scenarios_produce_a_real_anomaly():
    for scenario in ("clean", "diffuse"):
        df = datagen.generate(scenario, seed=42)
        r0 = aggregate_rate(df, "churned", "period", 0)
        r1 = aggregate_rate(df, "churned", "period", 1)
        assert r1 - r0 > 0.015, f"{scenario}: anomaly too small ({r1 - r0:.4f})"


def test_clean_scenario_is_localizable():
    df = datagen.generate("clean", seed=42)
    max_share, max_move = _scan(df)
    # one segment must both move a lot and explain most of the change
    assert max_move >= 0.15, f"clean: no segment moved enough ({max_move:.3f})"
    assert max_share >= 0.60, f"clean: top segment explains too little ({max_share:.2%})"


def test_diffuse_scenario_triggers_abstention():
    df = datagen.generate("diffuse", seed=42)
    max_share, max_move = _scan(df)
    # no segment moves much: the rate rose uniformly, so nothing is attributable
    assert max_move <= 0.06, f"diffuse: a segment moved too much ({max_move:.3f})"
    assert max_share < 0.55, f"diffuse: a segment explains too much ({max_share:.2%})"


def test_contributions_sum_to_delta_r_exactly():
    # the auditability property: sum of contributions == R1 - R0
    df = datagen.generate("clean", seed=42)
    for dim in DIMENSIONS:
        d = decompose_rate_metric(df, dim)
        delta_r = d.attrs["delta_r"]
        assert abs(d["contribution"].sum() - delta_r) < 1e-9


if __name__ == "__main__":
    test_both_scenarios_produce_a_real_anomaly()
    test_clean_scenario_is_localizable()
    test_diffuse_scenario_triggers_abstention()
    test_contributions_sum_to_delta_r_exactly()
    print("All datagen validation checks passed.")
