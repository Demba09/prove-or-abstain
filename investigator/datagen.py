"""Dataset generator — the make-or-break day-1 artifact.

Produces customer-window records for two periods (0 = baseline, 1 = current) for
a SaaS 30-day churn demo, in two scenarios:

  "clean"   : aggregate churn rises, driven by ONE segment whose rate collapses
              (channel = paid_partner_x). Localizable -> Investigator attributes,
              Verifier concludes with HIGH confidence.

  "diffuse" : aggregate churn rises by a similar amount, but spread roughly
              uniformly across every segment of every dimension. No single
              hypothesis clears the bar -> Verifier ABSTAINS.

The population composition is held stable between periods so the mix effect stays
near zero and the story is a clean rate effect — exactly what the decomposition
should surface. Everything is seeded and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Dimension universe. Segment shares are intentionally balanced (no single segment
# dominates the population) so that a uniform bump cannot masquerade as a localized
# cause in the diffuse scenario.
PLANS = ["free", "pro", "enterprise"]
PLAN_P = [0.45, 0.40, 0.15]

CHANNELS = ["organic", "paid_search", "paid_partner_x", "referral"]
CHANNEL_P = [0.34, 0.34, 0.12, 0.20]   # paid_partner_x is SMALL on purpose

REGIONS = ["na", "emea", "apac", "latam"]
REGION_P = [0.30, 0.30, 0.25, 0.15]

COHORTS = ["lt_3m", "3_to_12m", "gt_12m"]
COHORT_P = [0.35, 0.40, 0.25]

# Baseline churn modulation by segment (additive on a base rate). Mild and realistic:
# free churns more than enterprise, newest cohort churns more, etc.
PLAN_EFFECT = {"free": 0.03, "pro": 0.00, "enterprise": -0.02}
CHANNEL_EFFECT = {"organic": -0.01, "paid_search": 0.01, "paid_partner_x": 0.01, "referral": -0.01}
REGION_EFFECT = {"na": 0.0, "emea": 0.0, "apac": 0.005, "latam": 0.005}
COHORT_EFFECT = {"lt_3m": 0.03, "3_to_12m": 0.0, "gt_12m": -0.02}

BASE_RATE = 0.05


def _draw_population(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plan": rng.choice(PLANS, size=n, p=PLAN_P),
            "channel": rng.choice(CHANNELS, size=n, p=CHANNEL_P),
            "region": rng.choice(REGIONS, size=n, p=REGION_P),
            "cohort": rng.choice(COHORTS, size=n, p=COHORT_P),
        }
    )


def _baseline_churn_prob(df: pd.DataFrame) -> np.ndarray:
    p = (
        BASE_RATE
        + df["plan"].map(PLAN_EFFECT).to_numpy()
        + df["channel"].map(CHANNEL_EFFECT).to_numpy()
        + df["region"].map(REGION_EFFECT).to_numpy()
        + df["cohort"].map(COHORT_EFFECT).to_numpy()
    )
    return np.clip(p, 0.005, 0.95)


def _generate_period(
    rng: np.random.Generator,
    n: int,
    period: int,
    scenario: str,
) -> pd.DataFrame:
    df = _draw_population(rng, n)
    prob = _baseline_churn_prob(df)

    if period == 1:
        if scenario == "clean":
            # Localized rate shock: paid_partner_x retention collapses.
            mask = (df["channel"] == "paid_partner_x").to_numpy()
            prob = prob.copy()
            prob[mask] = np.clip(prob[mask] + 0.26, 0.0, 0.95)
        elif scenario == "diffuse":
            # Uniform additive shock across the whole population.
            prob = np.clip(prob + 0.03, 0.0, 0.95)
        else:
            raise ValueError(f"unknown scenario: {scenario!r}")

    df["period"] = period
    df["churned"] = (rng.random(n) < prob).astype(int)
    return df


def generate(
    scenario: str,
    n_per_period: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return the long dataframe (period 0 + period 1) for a scenario."""
    rng = np.random.default_rng(seed)
    p0 = _generate_period(rng, n_per_period, period=0, scenario=scenario)
    p1 = _generate_period(rng, n_per_period, period=1, scenario=scenario)
    out = pd.concat([p0, p1], ignore_index=True)
    out.attrs["scenario"] = scenario
    return out


def write(scenario: str, out_dir: str | Path = "data", **kwargs) -> Path:
    """Write the scenario dataset. Prefers parquet; falls back to CSV if no
    parquet engine (pyarrow) is installed, so it runs in any environment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(scenario, **kwargs)
    try:
        path = out_dir / f"churn_{scenario}.parquet"
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = out_dir / f"churn_{scenario}.csv"
        df.to_csv(path, index=False)
    return path
