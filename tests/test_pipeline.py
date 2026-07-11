"""Filet de sécurité de la démo : math, gates et API sur les trois panels.

Si CLEAN cesse de localiser, si DIFFUSE/MIXSHIFT cessent d'être rejetés, ou si
l'ABSTAIN se met à exécuter une action, ces tests cassent avant que ça
n'atteigne les juges.

Run: pytest -q
"""
from __future__ import annotations

import os

os.environ["QWEN_MOCK"] = "1"   # avant tout import qui instancie le client LLM

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.app import app
from attribution import decompose
from attribution_reference import (BASELINE as REF_BASELINE, CLEAN as REF_CLEAN,
                                   DIFFUSE as REF_DIFFUSE, decompose as oracle)
from gates import evaluate_gates
from metrics import aggregate

# Même MIXSHIFT que gate_check.py : n ET r bougent -> rate, mix, interaction non nuls.
REF_MIXSHIFT = pd.DataFrame([
    {"segment": "organic",  "n": 14000, "c": 630},
    {"segment": "paid",     "n": 4000,  "c": 320},
    {"segment": "referral", "n": 3000,  "c": 240},
    {"segment": "email",    "n": 1000,  "c": 100},
])

SCENARIOS = {"CLEAN": REF_CLEAN, "DIFFUSE": REF_DIFFUSE, "MIXSHIFT": REF_MIXSHIFT}
EXPECTED = {"CLEAN": "ASSERT", "DIFFUSE": "ABSTAIN", "MIXSHIFT": "ABSTAIN"}

client = TestClient(app)


# ------------------------------------------------------------------ la math
def test_decompose_matches_oracle():
    cols = ["rate", "mix", "interaction", "contribution"]
    for curr in SCENARIOS.values():
        a = oracle(REF_BASELINE, curr, dims="segment")
        b = decompose(REF_BASELINE, curr, dims="segment").reindex(a.index)
        assert np.allclose(a[cols].values, b[cols].values, atol=1e-10)


def test_decompose_zero_residual():
    for curr in SCENARIOS.values():
        agg = aggregate(decompose(REF_BASELINE, curr, dims="segment"))
        assert abs(agg["residual"]) < 1e-10


# ----------------------------------------------------------------- les gates
def test_gates_verdicts():
    baseline_n = REF_BASELINE.set_index("segment")["n"]
    for name, curr in SCENARIOS.items():
        out = decompose(REF_BASELINE, curr, dims="segment")
        rep = evaluate_gates(aggregate(out), out, baseline_n=baseline_n)
        assert rep.verdict == EXPECTED[name], f"{name}: {rep.reasons}"


# ------------------------------------------------------------------- l'API
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_investigate_verdicts():
    for panel, want in [("clean", "ASSERT"), ("diffuse", "ABSTAIN"),
                        ("mixshift", "ABSTAIN")]:
        body = client.post("/investigate", json={"panel": panel}).json()
        assert body["verdict"] == want, f"{panel}: {body['trace']}"


def test_clean_localizes_paid_after_loop():
    body = client.post("/investigate", json={"panel": "clean"}).json()
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}
    # la boucle a bien tourné : device essayé (et rejeté) avant segment
    assert body["gates"]["device"]["verdict"] == "ABSTAIN"


def test_abstain_never_executes():
    # LA propriété de sûreté : même autopilot ON, un ABSTAIN n'exécute jamais.
    for panel in ("diffuse", "mixshift"):
        body = client.post("/investigate", json={"panel": panel,
                                                 "autopilot": True}).json()
        assert body["action"]["kind"] == "ESCALATE"


def test_autopilot_executes_on_proof():
    body = client.post("/investigate", json={"panel": "clean",
                                             "autopilot": True}).json()
    assert body["action"]["kind"] == "EXECUTE"
    assert body["confidence"] >= 0.70


def test_unknown_panel_rejected():
    assert client.post("/investigate", json={"panel": "bogus"}).status_code == 422


# ---------------------------------------------------------- page de démo
def test_home_serves_demo_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "prove-or-abstain" in r.text


# ------------------------------------------------------------ upload CSV
def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def test_upload_roundtrip_matches_panels():
    from panels import BASELINE, CLEAN, DIFFUSE
    for curr, want in [(CLEAN, "ASSERT"), (DIFFUSE, "ABSTAIN")]:
        r = client.post("/investigate/upload", files={
            "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
            "current": ("current.csv", _csv(curr), "text/csv"),
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["panel"] == "upload" and body["verdict"] == want


def test_upload_clean_localizes_paid():
    from panels import BASELINE, CLEAN
    body = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    }).json()
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_upload_rejects_missing_column():
    from panels import BASELINE, CLEAN
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE.drop(columns=["n"])), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    })
    assert r.status_code == 400
    assert "missing required column" in r.json()["detail"]


def test_upload_rejects_mismatched_columns():
    from panels import BASELINE, CLEAN
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
        "current": ("current.csv", _csv(CLEAN.rename(columns={"device": "browser"})), "text/csv"),
    })
    assert r.status_code == 400
    assert "same columns" in r.json()["detail"]


# ------------------------------------------------- gate de significativité
def test_significance_gate_rejects_small_samples():
    # Mêmes taux que CLEAN mais n divisé par 100 : la concentration est
    # parfaite, pourtant le mouvement n'est plus significatif -> ABSTAIN.
    # C'est exactement ce qu'un plancher n>=1000 ne savait pas expliquer.
    scale = 100
    small_base = REF_BASELINE.assign(n=REF_BASELINE.n // scale, c=REF_BASELINE.c // scale)
    small_curr = REF_CLEAN.assign(n=REF_CLEAN.n // scale, c=REF_CLEAN.c // scale)
    out = decompose(small_base, small_curr, dims="segment")
    rep = evaluate_gates(aggregate(out), out)
    assert rep.verdict == "ABSTAIN"
    assert any("non significatif" in r for r in rep.reasons)


def test_significance_gate_passes_on_clean():
    out = decompose(REF_BASELINE, REF_CLEAN, dims="segment")
    rep = evaluate_gates(aggregate(out), out)
    assert rep.verdict == "ASSERT"
    assert rep.leading_p < 0.01 and abs(rep.leading_z) > 2.576


# --------------------------------------------------------- métriques somme
def test_decompose_sum_exact():
    from attribution import decompose_sum
    base = pd.DataFrame([{"segment": "a", "n": 100, "c": 2000},
                         {"segment": "b", "n": 50, "c": 2500}])
    curr = pd.DataFrame([{"segment": "a", "n": 120, "c": 1800},
                         {"segment": "b", "n": 50, "c": 2500}])
    out = decompose_sum(base, curr, dims="segment")
    # contribution_s == c1 - c0, et somme == ΔV, résidu nul
    assert np.allclose(out["contribution"], out["c1"] - out["c0"])
    agg = aggregate(out)
    assert abs(agg["residual"]) < 1e-9
    assert agg["delta_R"] == (1800 + 2500) - (2000 + 2500)


def test_sum_metric_upload_asserts():
    r = client.post("/investigate/upload",
                    files={"baseline": open("examples/revenue_baseline.csv", "rb"),
                           "current": open("examples/revenue_current.csv", "rb")},
                    data={"sum_metrics": "revenue"})
    body = r.json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


# --------------------------------------------------------------- drill-down
def test_deep_panel_drills_down():
    body = client.post("/investigate", json={"panel": "deep"}).json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "device", "segment": "mobile"}
    refined = body["drilldown"]["refined"]
    assert refined["dim"] == "segment" and refined["segment"] == "paid"
    assert "segment=paid" in body["action"]["detail"]


def test_clean_panel_does_not_refine():
    # paid est réparti 50/50 mobile/desktop : rien à affiner, et le driller
    # doit le dire au lieu d'inventer une sous-cause.
    body = client.post("/investigate", json={"panel": "clean"}).json()
    assert body["verdict"] == "ASSERT"
    assert body["drilldown"]["refined"] is None


def test_abstain_has_no_drilldown():
    body = client.post("/investigate", json={"panel": "diffuse"}).json()
    assert body["drilldown"] is None


# ---------------------------------------------------------- série temporelle
def test_series_endpoint_rolling_baseline():
    from panels import make_series
    r = client.post("/investigate/series", files={
        "series": ("series.csv", _csv(make_series()), "text/csv"),
    })
    body = r.json()
    assert body["panel"] == "series"
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_series_window_limits_baseline():
    from panels import make_series
    r = client.post("/investigate/series",
                    files={"series": ("series.csv", _csv(make_series()), "text/csv")},
                    data={"window": "3"})
    assert r.json()["verdict"] == "ASSERT"


def test_series_requires_period_column():
    from panels import BASELINE
    r = client.post("/investigate/series", files={
        "series": ("series.csv", _csv(BASELINE), "text/csv"),
    })
    assert r.status_code == 400
    assert "period" in r.json()["detail"]


# ------------------------------------------------------------- spéculations
def test_speculations_only_on_assert():
    assert_body = client.post("/investigate", json={"panel": "clean"}).json()
    abstain_body = client.post("/investigate", json={"panel": "diffuse"}).json()
    assert len(assert_body["speculations"]) >= 1
    assert abstain_body["speculations"] == []
