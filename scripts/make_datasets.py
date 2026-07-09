"""Generate and validate the two demo datasets.

Run: python -m scripts.make_datasets
"""

from __future__ import annotations

from investigator import datagen
from investigator.attribution import aggregate_rate, decompose_rate_metric

DIMENSIONS = ["plan", "channel", "region", "cohort"]


def summarize(scenario: str) -> None:
    path = datagen.write(scenario)
    df = datagen.generate(scenario, seed=42)
    r0 = aggregate_rate(df, "churned", "period", 0)
    r1 = aggregate_rate(df, "churned", "period", 1)
    print(f"\n[{scenario}] -> {path}")
    print(f"  aggregate churn {r0:.4f} -> {r1:.4f}  (dR={r1 - r0:+.4f})")
    best = ("", 0.0)
    for dim in DIMENSIONS:
        d = decompose_rate_metric(df, dim)
        top = d.iloc[0]
        if abs(top["explained_share"]) > abs(best[1]):
            best = (f"{dim}={top['segment']}", float(top["explained_share"]))
    print(f"  strongest single segment: {best[0]} explains {best[1]:.1%} of dR")


if __name__ == "__main__":
    for s in ("clean", "diffuse"):
        summarize(s)
    print("\nDatasets written to data/. Validate with: pytest -q")
