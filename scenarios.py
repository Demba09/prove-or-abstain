"""
scenarios.py — a seeded generator of labelled investigation scenarios.

Every "AI for analytics" claim lives or dies on one question: when there is no
real localized cause, does the agent invent one anyway? To measure that we
need scenarios whose GROUND TRUTH we control — where we know there is a
localized cause, and where we know there is not.

One panel = one metric (`conversion`) over a single `segment` dimension, in the
long format the pipeline already consumes (`metric, segment, n, c`). Each
scenario carries its truth: a label, the planted cause (or None), and — for the
no-cause classes — which verification gate SHOULD be the one that fails.

Five labels, along one intuition each:
  localized    : a real, concentrated cause -> the agent SHOULD assert it.
  diffuse      : every segment moves together -> no dimension localizes.
  mixshift     : population mix and rates move at once -> mechanism entangled.
  noise        : no real move, only sampling wobble -> nothing to explain.
  small_sample : a concentrated move, but too little data to prove it.

`generate_suite(seed)` returns 45 scenarios (9 per label) and is exactly
reproducible: same seed in, byte-identical panels out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEGMENTS = ["organic", "paid", "referral", "email", "social"]
BASE_RATE = 0.07          # baseline conversion rate, uniform across segments
BASE_N = 4000             # baseline population per segment (large -> powered)


@dataclass
class Scenario:
    name: str
    label: str                      # localized | diffuse | mixshift | noise | small_sample
    baseline: pd.DataFrame
    current: pd.DataFrame
    dims: list = field(default_factory=lambda: ["segment"])
    cause: str | None = None        # planted segment (localized only), else None
    expected_reasons: set = field(default_factory=set)  # acceptable failing gate names
    params: dict = field(default_factory=dict)

    @property
    def should_assert(self) -> bool:
        return self.label == "localized"


def _panel(rates: dict, ns: dict) -> pd.DataFrame:
    """Build a long panel from segment -> rate and segment -> n."""
    rows = []
    for s in SEGMENTS:
        n = int(round(ns[s]))
        c = int(round(n * float(np.clip(rates[s], 0.0, 1.0))))
        rows.append({"metric": "conversion", "segment": s, "n": n, "c": c})
    return pd.DataFrame(rows)


def _noisy(rate: float, n: int, noise: float, rng) -> float:
    """Perturb a rate by sampling noise (binomial standard error) times a
    noise factor. noise=0 leaves the rate untouched."""
    if noise <= 0:
        return rate
    se = np.sqrt(max(rate * (1 - rate), 1e-9) / max(n, 1))
    return float(rate + rng.normal(0.0, noise * se))


def make_scenario(label: str, seed: int, *, intensity: float = 0.30,
                  concentration: float = 1.0, noise: float = 0.0,
                  n: int = BASE_N, target: str = "paid") -> Scenario:
    """Build one scenario of `label`. Parameters:
      intensity     : relative size of the aggregate move (0.30 = -30%).
      concentration : share of the move sitting in `target` (1 localized, 0 diffuse).
      noise         : sampling-noise multiplier added to every segment rate.
      n             : baseline population per segment (small -> underpowered).
      target        : the segment that carries a localized cause.
    """
    rng = np.random.default_rng(seed)
    ns0 = {s: n for s in SEGMENTS}
    r0 = {s: BASE_RATE for s in SEGMENTS}

    if label == "noise":
        # no real move: same expected rate both periods, only sampling wobble.
        ns1 = dict(ns0)
        r1 = {s: _noisy(BASE_RATE, n, max(noise, 0.6), rng) for s in SEGMENTS}
        r0 = {s: _noisy(BASE_RATE, n, max(noise, 0.6), rng) for s in SEGMENTS}
        reasons = {"material", "localized", "significant", "confident"}
        cause = None

    elif label == "mixshift":
        # population mix shifts (one segment grows, one shrinks) AND rates move,
        # so rate/mix/interaction are all non-trivial -> entangled mechanism.
        ns1 = dict(ns0)
        ns1["organic"] = n * 2.2
        ns1["paid"] = n * 0.4
        r1 = dict(r0)
        r1["organic"] = BASE_RATE * 0.7
        r1["paid"] = BASE_RATE * 1.4
        r1 = {s: _noisy(r1[s], n, noise, rng) for s in SEGMENTS}
        reasons = {"clean"}
        cause = None

    else:
        # localized / diffuse / small_sample share one construction along the
        # concentration axis: distribute an aggregate loss between one target
        # segment (concentrated part) and all segments (uniform part).
        total_c = sum(r0[s] * ns0[s] for s in SEGMENTS)
        lost = total_c * intensity
        uniform_each = (1.0 - concentration) * lost / len(SEGMENTS)
        r1 = {}
        for s in SEGMENTS:
            loss = uniform_each + (concentration * lost if s == target else 0.0)
            r1[s] = (r0[s] * ns0[s] - loss) / ns0[s]
        r1 = {s: _noisy(r1[s], n, noise, rng) for s in SEGMENTS}
        ns1 = dict(ns0)
        if label == "localized":
            reasons, cause = set(), target
        elif label == "diffuse":
            reasons, cause = {"localized"}, None
        elif label == "small_sample":
            reasons, cause = {"significant"}, None
        else:
            raise ValueError(f"unknown label {label!r}")

    base = _panel(r0, ns0)
    curr = _panel(r1, ns1)
    name = f"{label}-{seed}"
    return Scenario(name=name, label=label, baseline=base, current=curr,
                    cause=cause, expected_reasons=reasons,
                    params={"intensity": intensity, "concentration": concentration,
                            "noise": noise, "n": n, "target": target})


def generate_suite(seed: int = 0) -> list[Scenario]:
    """45 scenarios: 9 per label, varied deterministically off `seed`."""
    out: list[Scenario] = []
    for i in range(9):
        s = seed * 1000 + i
        # localized: strongly concentrated, well powered, a little noise
        out.append(make_scenario("localized", s, intensity=0.25 + 0.03 * i,
                                  concentration=0.90 + 0.01 * (i % 3), noise=0.15))
        # diffuse: the SAME aggregate move, spread flat
        out.append(make_scenario("diffuse", s, intensity=0.25 + 0.03 * i,
                                  concentration=0.05, noise=0.15))
        # mixshift: entangled mechanism
        out.append(make_scenario("mixshift", s, noise=0.10))
        # noise: no real move, only wobble
        out.append(make_scenario("noise", s, noise=0.7 + 0.05 * (i % 4)))
        # small_sample: concentrated but starved of data
        out.append(make_scenario("small_sample", s, intensity=0.30,
                                  concentration=0.95, noise=0.15, n=40 + 5 * i))
    return out
