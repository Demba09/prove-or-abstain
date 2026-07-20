"""Demo safety net: math, gates and API across the built-in panels.

If CLEAN stops localizing, if DIFFUSE/MIXSHIFT stop being rejected, or if
ABSTAIN ever starts executing an action, these tests break before that
reaches the judges.

Run: pytest -q
"""
from __future__ import annotations

import os

os.environ["QWEN_MOCK"] = "1"   # before any import that instantiates the LLM client
os.environ.setdefault("PROBATIO_DB", ":memory:")   # never touch disk in tests

import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import app
from prove_or_abstain.attribution import decompose
from scripts.attribution_reference import (BASELINE as REF_BASELINE, CLEAN as REF_CLEAN,
                                   DIFFUSE as REF_DIFFUSE, decompose as oracle)
from prove_or_abstain.gates import evaluate_gates
from prove_or_abstain.metrics import aggregate

# Same MIXSHIFT as gate_check.py: n AND r both move -> rate, mix, interaction
# all non-zero.
REF_MIXSHIFT = pd.DataFrame([
    {"segment": "organic",  "n": 14000, "c": 630},
    {"segment": "paid",     "n": 4000,  "c": 320},
    {"segment": "referral", "n": 3000,  "c": 240},
    {"segment": "email",    "n": 1000,  "c": 100},
])

SCENARIOS = {"CLEAN": REF_CLEAN, "DIFFUSE": REF_DIFFUSE, "MIXSHIFT": REF_MIXSHIFT}
EXPECTED = {"CLEAN": "ASSERT", "DIFFUSE": "ABSTAIN", "MIXSHIFT": "ABSTAIN"}

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Every test shares one TestClient/client_id, so without a reset the
    rate-limit middleware's global counter accumulates across the whole
    suite (100+ HTTP calls total) and starts 429ing unrelated tests partway
    through a run. Real usage doesn't have this problem — it's an artifact
    of one process serving the entire test suite as if it were one client."""
    from prove_or_abstain import ratelimit
    ratelimit.reset()
    yield


# -------------------------------------------------------------------- math
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


# ------------------------------------------------------------------- gates
def test_gates_verdicts():
    baseline_n = REF_BASELINE.set_index("segment")["n"]
    for name, curr in SCENARIOS.items():
        out = decompose(REF_BASELINE, curr, dims="segment")
        rep = evaluate_gates(aggregate(out), out, baseline_n=baseline_n)
        assert rep.verdict == EXPECTED[name], f"{name}: {rep.reasons}"


# --------------------------------------------------------------------- API
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
    # the loop really ran: device was tried (and rejected) before segment
    assert body["gates"]["device"]["verdict"] == "ABSTAIN"


def test_abstain_never_executes():
    # THE safety property: even with autopilot ON, an ABSTAIN never executes.
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


# --------------------------------------------------------------- demo page
def test_home_serves_demo_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "prove-or-abstain" in r.text


# -------------------------------------------------------------- CSV upload
def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def test_upload_roundtrip_matches_panels():
    from prove_or_abstain.panels import BASELINE, CLEAN, DIFFUSE
    for curr, want in [(CLEAN, "ASSERT"), (DIFFUSE, "ABSTAIN")]:
        r = client.post("/investigate/upload", files={
            "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
            "current": ("current.csv", _csv(curr), "text/csv"),
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["panel"] == "upload" and body["verdict"] == want


def test_upload_clean_localizes_paid():
    from prove_or_abstain.panels import BASELINE, CLEAN
    body = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    }).json()
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_upload_rejects_missing_column():
    from prove_or_abstain.panels import BASELINE, CLEAN
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE.drop(columns=["n"])), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    })
    assert r.status_code == 400
    assert "missing required column" in r.json()["detail"]


def test_upload_rejects_negative_counts():
    from prove_or_abstain.panels import BASELINE, CLEAN
    bad = BASELINE.copy()
    bad.loc[0, "n"] = -5
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(bad), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    })
    assert r.status_code == 400
    assert "negative" in r.json()["detail"]


def test_upload_rejects_missing_values():
    from prove_or_abstain.panels import BASELINE, CLEAN
    bad = BASELINE.copy()
    bad.loc[0, "c"] = np.nan
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(bad), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    })
    assert r.status_code == 400
    assert "missing values" in r.json()["detail"]


def test_upload_rejects_c_over_n_on_rate_metric():
    # c > n makes no sense for a rate (successes out of n); it must be a
    # 400, not a silent nonsense verdict. Sum metrics stay exempt —
    # test_sum_metric_upload_asserts covers that (revenue has c >> n).
    from prove_or_abstain.panels import BASELINE, CLEAN
    bad = CLEAN.copy()
    bad.loc[0, "c"] = bad.loc[0, "n"] + 1
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
        "current": ("current.csv", _csv(bad), "text/csv"),
    })
    assert r.status_code == 400
    assert "c > n" in r.json()["detail"]


def test_upload_rejects_mismatched_columns():
    from prove_or_abstain.panels import BASELINE, CLEAN
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
        "current": ("current.csv", _csv(CLEAN.rename(columns={"device": "browser"})), "text/csv"),
    })
    assert r.status_code == 400
    assert "same columns" in r.json()["detail"]


# --------------------------------------------------------- significance gate
def test_significance_gate_rejects_small_samples():
    # Same rates as CLEAN but n divided by 100: the concentration is perfect,
    # yet the move is no longer significant -> ABSTAIN.
    # This is exactly what an n>=1000 floor could not explain.
    scale = 100
    small_base = REF_BASELINE.assign(n=REF_BASELINE.n // scale, c=REF_BASELINE.c // scale)
    small_curr = REF_CLEAN.assign(n=REF_CLEAN.n // scale, c=REF_CLEAN.c // scale)
    out = decompose(small_base, small_curr, dims="segment")
    rep = evaluate_gates(aggregate(out), out)
    assert rep.verdict == "ABSTAIN"
    assert any("not significant" in r for r in rep.reasons)


def test_significance_gate_passes_on_clean():
    out = decompose(REF_BASELINE, REF_CLEAN, dims="segment")
    rep = evaluate_gates(aggregate(out), out)
    assert rep.verdict == "ASSERT"
    assert rep.leading_p < 0.01 and abs(rep.leading_z) > 2.576


# --------------------------------------------------------------- sum metrics
def test_decompose_sum_exact():
    from prove_or_abstain.attribution import decompose_sum
    base = pd.DataFrame([{"segment": "a", "n": 100, "c": 2000},
                         {"segment": "b", "n": 50, "c": 2500}])
    curr = pd.DataFrame([{"segment": "a", "n": 120, "c": 1800},
                         {"segment": "b", "n": 50, "c": 2500}])
    out = decompose_sum(base, curr, dims="segment")
    # contribution_s == c1 - c0, their sum == ΔV, zero residual
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


# ---------------------------------------------------------------- drill-down
def test_deep_panel_drills_down():
    body = client.post("/investigate", json={"panel": "deep"}).json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "device", "segment": "mobile"}
    refined = body["drilldown"]["refined"]
    assert refined["dim"] == "segment" and refined["segment"] == "paid"
    assert "segment=paid" in body["action"]["detail"]


def test_clean_panel_does_not_refine():
    # paid is split 50/50 mobile/desktop: nothing to refine, and the driller
    # must say so instead of inventing a sub-cause.
    body = client.post("/investigate", json={"panel": "clean"}).json()
    assert body["verdict"] == "ASSERT"
    assert body["drilldown"]["refined"] is None


def test_deep_panel_root_cause_and_refined_swap_with_dimension_order():
    """A single-cell collapse (paid x mobile) concentrates 100% on BOTH its
    defining dimensions -- mathematically inevitable, not a calibration
    quirk (found via a real, non-mock Qwen run that happened to order
    dimensions differently than the mock default and got marked "wrong" by
    the benchmark before this was understood). Whichever dimension is
    tested first becomes the top-level root_cause; the driller always finds
    the OTHER one as `refined` -- so the full (paid, mobile) diagnosis is
    recovered either way. Only the top-level/refined labelling depends on
    order, never whether a cause is found or which cell it points to."""
    from prove_or_abstain.panels import BASELINE
    from prove_or_abstain.benchmark import _deep
    from prove_or_abstain.graph import APP

    curr = _deep("paid", "mobile", 0.03)
    state = {"baseline": BASELINE, "current": curr,
             "metrics": ["conversion", "activation"], "metric_kinds": {},
             "dims": ["segment", "device"],  # segment tested BEFORE device
             "autopilot_enabled": False, "trace": []}
    final = APP.invoke(state)
    assert final["verdict"] == "ASSERT"
    assert final["winning_dim"] == "segment"
    assert final["winning_report"].leading_segment == "paid"
    refined = final["drilldown"]["refined"]
    assert refined["dim"] == "device" and refined["segment"] == "mobile"


def test_abstain_has_no_drilldown():
    body = client.post("/investigate", json={"panel": "diffuse"}).json()
    assert body["drilldown"] is None


# ----------------------------------------------------------------- time series
def test_series_endpoint_rolling_baseline():
    from prove_or_abstain.panels import make_series
    r = client.post("/investigate/series", files={
        "series": ("series.csv", _csv(make_series()), "text/csv"),
    })
    body = r.json()
    assert body["panel"] == "series"
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_series_window_limits_baseline():
    from prove_or_abstain.panels import make_series
    r = client.post("/investigate/series",
                    files={"series": ("series.csv", _csv(make_series()), "text/csv")},
                    data={"window": "3"})
    assert r.json()["verdict"] == "ASSERT"


def test_series_rejects_bad_window():
    from prove_or_abstain.panels import make_series
    r = client.post("/investigate/series",
                    files={"series": ("series.csv", _csv(make_series()), "text/csv")},
                    data={"window": "0"})
    assert r.status_code == 400
    assert "window" in r.json()["detail"]


def test_series_requires_period_column():
    from prove_or_abstain.panels import BASELINE
    r = client.post("/investigate/series", files={
        "series": ("series.csv", _csv(BASELINE), "text/csv"),
    })
    assert r.status_code == 400
    assert "period" in r.json()["detail"]


# --------------------------------------------------------------- speculation
def test_speculations_only_on_assert():
    assert_body = client.post("/investigate", json={"panel": "clean"}).json()
    abstain_body = client.post("/investigate", json={"panel": "diffuse"}).json()
    assert len(assert_body["speculations"]) >= 1
    assert abstain_body["speculations"] == []


# --------------------------------------------------------------- llm visibility
def test_llm_mode_reported_in_mock():
    body = client.post("/investigate", json={"panel": "clean"}).json()
    assert body["llm"] == {"model": "qwen-plus", "mode": "mock"}
    assert any("[mock]" in t for t in body["trace"])


# ----------------------------------------------------------- natural-language
def test_query_routes_to_matching_panel():
    body = client.post("/investigate/query",
                       json={"query": "every segment dropped by the same amount, "
                                     "uniform decline"}).json()
    assert body["panel"] == "diffuse"
    assert body["verdict"] == "ABSTAIN"
    assert body["routing"]["panel"] == "diffuse"


def test_query_rejects_empty():
    r = client.post("/investigate/query", json={"query": "   "})
    assert r.status_code == 400


def test_query_never_picks_outside_panels():
    # gibberish query: mock router must still land on a valid panel, never
    # crash or invent a name outside the supplied options.
    body = client.post("/investigate/query", json={"query": "asdkjhaskjdh"}).json()
    assert body["panel"] in {"clean", "diffuse", "mixshift", "deep"}


# ------------------------------------------------------- conversational follow-up
def test_query_followup_filters_to_named_segment():
    unfiltered = client.post("/investigate", json={"panel": "clean"}).json()
    filtered = client.post("/investigate/query", json={
        "query": "and what about just mobile",
        "previous_panel": "clean",
    }).json()
    assert filtered["panel"] == "clean"
    assert filtered["routing"]["filter"] == {"dim": "device", "segment": "mobile"}
    # the pipeline actually ran on half the population once filtered
    assert (filtered["gates"]["segment"]["leading_sample_n"]
            < unfiltered["gates"]["segment"]["leading_sample_n"])


def test_query_followup_keeps_previous_panel_without_filter():
    body = client.post("/investigate/query", json={
        "query": "and what about that",
        "previous_panel": "diffuse",
    }).json()
    assert body["panel"] == "diffuse"
    assert body["routing"]["filter"] is None


def test_guard_filter_rejects_values_outside_the_supplied_dims():
    # This is the safety net for the REAL LLM path (route_query's non-mock
    # branch): a filter is only accepted if both the dim and the segment
    # value were in what the caller supplied, never a hallucinated pair.
    from prove_or_abstain.llm import _guard_filter
    dims = {"segment": ["organic", "paid"], "device": ["mobile", "desktop"]}
    assert _guard_filter({"dim": "segment", "segment": "paid"}, dims) == \
        {"dim": "segment", "segment": "paid"}
    assert _guard_filter({"dim": "segment", "segment": "moon"}, dims) is None
    assert _guard_filter({"dim": "country", "segment": "france"}, dims) is None
    assert _guard_filter(None, dims) is None
    assert _guard_filter({"dim": "segment", "segment": "paid"}, None) is None


# --------------------------------------------- "ask" against a watched source
def test_query_source_needs_two_observations():
    from prove_or_abstain import memory
    memory.reset()
    r = client.post("/investigate/query", json={"query": "why?", "source_id": "onlyone"})
    assert r.status_code == 400   # no observations at all yet

    df = pd.DataFrame({"metric": ["conversion"], "segment": ["paid"], "n": [100], "c": [10]})
    client.post("/sources/onlyone/observe", files={"panel": ("p.csv", _csv(df), "text/csv")})
    r = client.post("/investigate/query", json={"query": "why?", "source_id": "onlyone"})
    assert r.status_code == 400   # cold start only — nothing to compare against yet


def test_query_source_filters_to_named_segment_via_examples():
    """Reuses the real STEM/gender-majority finding (see the real-data tests
    below) to prove "ask" can narrow an already-watched source to a segment
    Qwen extracts from free text — same guarded mechanism as previous_panel,
    just against one persisted dataset instead of 4 named ones."""
    from pathlib import Path
    from prove_or_abstain import memory
    root = Path(__file__).resolve().parent.parent / "examples"
    memory.reset()
    client.post("/sources/majors_ask/observe",
               files={"panel": ("nonstem.csv", (root / "real_majors_nonstem.csv").read_bytes(), "text/csv")})
    unfiltered = client.post("/sources/majors_ask/observe",
               files={"panel": ("stem.csv", (root / "real_majors_stem.csv").read_bytes(), "text/csv")}).json()

    filtered = client.post("/investigate/query", json={
        "query": "what about majority_women?", "source_id": "majors_ask",
    }).json()
    assert filtered["routing"]["filter"] == {"dim": "gender_majority", "segment": "majority_women"}
    # only one segment left in the filtered data, so it trivially IS 100% of
    # the move — unlike the unfiltered comparison's genuinely uncertain 0.01
    assert filtered["gates"]["gender_majority"]["concentration"] == 1.0
    assert unfiltered["gates"]["gender_majority"]["confidence"] < 0.1


# ------------------------------------------------------------- setup suggestion
def test_suggest_setup_flags_revenue_as_sum():
    from prove_or_abstain.llm import template_suggest_setup
    out = template_suggest_setup(["conversion", "revenue", "activation"])
    assert out["sum_metrics"] == ["revenue"]


def test_suggest_setup_never_invents_a_metric_name():
    from prove_or_abstain.llm import get_client
    out = get_client().suggest_setup(["conversion", "revenue"])
    assert set(out["sum_metrics"]) <= {"conversion", "revenue"}


def test_investigate_suggest_endpoint():
    from prove_or_abstain.panels import BASELINE
    r = client.post("/investigate/suggest",
                    files={"baseline": ("baseline.csv", _csv(BASELINE), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["dims"]) == {"segment", "device"}
    assert body["metrics"] == ["activation", "conversion"]
    assert body["sum_metrics"] == []          # neither metric name hints at a sum


def test_investigate_suggest_flags_revenue_column():
    from prove_or_abstain.panels import BASELINE
    revenue_panel = BASELINE.copy()
    revenue_panel["metric"] = "revenue"
    r = client.post("/investigate/suggest",
                    files={"baseline": ("baseline.csv", _csv(revenue_panel), "text/csv")})
    assert r.json()["sum_metrics"] == ["revenue"]


# -------------------------------------------------- wide dimension space (3+ dims)
def _plan_examples():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "examples"
    return (root / "plan_baseline.csv").read_bytes(), (root / "plan_current.csv").read_bytes()


def test_upload_localizes_on_a_third_dimension():
    """Neither 'segment' nor 'device' localizes this drop (concentration
    0.25 and 0.50, both < 0.55) — only 'plan' does (concentration 1.0).
    Proves the loop isn't hardcoded to 2 dimensions: it exhausts whatever
    candidate list it's given, in CSV column order (segment, device, plan)."""
    base_csv, curr_csv = _plan_examples()
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", base_csv, "text/csv"),
        "current": ("current.csv", curr_csv, "text/csv"),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "plan", "segment": "free"}
    assert body["gates"]["segment"]["verdict"] == "ABSTAIN"
    assert body["gates"]["device"]["verdict"] == "ABSTAIN"


def test_dimension_order_changes_speed_not_verdict():
    """Same data, same verdict, but testing 'plan' first instead of last
    finds it in 1 iteration instead of 3 — this is exactly what Qwen's
    plan_dimensions() ordering buys in agent/graph mode: the math tests
    every dimension either way, so the outcome can't change, but a good
    order gets there faster (fewer investigator/verifier passes)."""
    import io
    from prove_or_abstain.graph import APP as GRAPH
    base_csv, curr_csv = _plan_examples()
    base = pd.read_csv(io.BytesIO(base_csv))
    curr = pd.read_csv(io.BytesIO(curr_csv))

    def run(dims):
        state = {"baseline": base, "current": curr, "metrics": ["conversion"],
                 "dims": dims, "autopilot_enabled": False, "trace": []}
        return GRAPH.invoke(state)

    worst = run(["segment", "device", "plan"])   # plan tried last
    best = run(["plan", "segment", "device"])     # plan tried first

    assert worst["verdict"] == best["verdict"] == "ASSERT"
    assert worst["winning_dim"] == best["winning_dim"] == "plan"
    assert best["iteration"] < worst["iteration"]
    assert best["iteration"] == 1
    assert worst["iteration"] == 3


# --------------------------------------------------- evidence-grounded speculation
def test_find_events_matches_exact_segment():
    from prove_or_abstain.evidence import find_events
    events = find_events("segment", "paid")
    assert events and events[0]["event"] == "campaign_budget_cut"
    assert find_events("segment", "organic") != events
    assert find_events("segment", "unknown_segment") == []
    assert find_events(None, "paid") == []


def test_assert_speculation_grounds_in_logged_event():
    # CLEAN localizes to segment=paid, which has a matching synthetic event.
    body = client.post("/investigate", json={"panel": "clean"}).json()
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}
    assert any("campaign_budget_cut" in s for s in body["speculations"])


def test_abstain_speculation_has_no_events_to_ground_in():
    body = client.post("/investigate", json={"panel": "diffuse"}).json()
    assert body["speculations"] == []   # unchanged: ABSTAIN never speculates


# --------------------------------------------------------------- SQL connector
def _sqlite_dsn(tmp_path, tables: dict) -> str:
    from sqlalchemy import create_engine
    dsn = f"sqlite:///{tmp_path / 'panel.db'}"
    engine = create_engine(dsn)
    for table, df in tables.items():
        df.to_sql(table, engine, index=False, if_exists="replace")
    engine.dispose()
    return dsn


def test_sql_connector_matches_upload(tmp_path):
    from prove_or_abstain.panels import BASELINE, CLEAN
    dsn = _sqlite_dsn(tmp_path, {"baseline": BASELINE, "current": CLEAN})
    body = client.post("/investigate/sql", json={
        "dsn": dsn,
        "baseline_query": "SELECT * FROM baseline",
        "current_query": "SELECT * FROM current",
    }).json()
    assert body["panel"] == "sql"
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_sql_connector_with_where_clause(tmp_path):
    # queries can reshape/filter as long as the result stays long-panel
    from prove_or_abstain.panels import BASELINE, DIFFUSE
    dsn = _sqlite_dsn(tmp_path, {"baseline": BASELINE, "current": DIFFUSE})
    body = client.post("/investigate/sql", json={
        "dsn": dsn,
        "baseline_query": "SELECT * FROM baseline WHERE metric = 'conversion'",
        "current_query": "SELECT * FROM current WHERE metric = 'conversion'",
    }).json()
    assert body["panel"] == "sql"
    assert body["verdict"] == "ABSTAIN"


def test_sql_connector_rejects_non_select(tmp_path):
    dsn = _sqlite_dsn(tmp_path, {})
    r = client.post("/investigate/sql", json={
        "dsn": dsn, "baseline_query": "DROP TABLE sqlite_master", "current_query": "SELECT 1",
    })
    assert r.status_code == 400
    assert "SELECT" in r.json()["detail"]


def test_sql_connector_rejects_stacked_statements(tmp_path):
    dsn = _sqlite_dsn(tmp_path, {})
    r = client.post("/investigate/sql", json={
        "dsn": dsn, "baseline_query": "SELECT 1; DROP TABLE sqlite_master", "current_query": "SELECT 1",
    })
    assert r.status_code == 400


def test_sql_guard_allows_semicolon_inside_a_string_literal():
    """A bare "';' in query" check would reject a single, legitimate
    statement that happens to contain a semicolon inside a WHERE literal —
    a false positive, not a safety win. The guard strips string literals
    before counting ';', so this must pass."""
    from prove_or_abstain.connectors.sql import _guard_single_select
    _guard_single_select("SELECT metric, n, c FROM t WHERE note = 'a; b'")  # no raise
    _guard_single_select('SELECT metric, n, c FROM t WHERE note = "a; b"')  # no raise
    _guard_single_select("SELECT metric, n, c FROM t WHERE note = 'it''s; ok'")  # escaped quote


def test_sql_guard_still_rejects_a_real_second_statement():
    from prove_or_abstain.connectors.sql import SqlQueryError, _guard_single_select
    with pytest.raises(SqlQueryError):
        _guard_single_select("SELECT 1; DROP TABLE t")
    with pytest.raises(SqlQueryError):
        # a semicolon after a closed literal is a real second statement
        _guard_single_select("SELECT * FROM t WHERE note = 'a'; DROP TABLE t")


# --------------------------------------------------------------- rate limit
def test_ratelimit_allows_under_limit_and_blocks_over(monkeypatch):
    from prove_or_abstain import ratelimit
    monkeypatch.setattr(ratelimit, "LIMIT_PER_MINUTE", 3)
    ratelimit.reset()
    assert ratelimit.allow("client-a") is True
    assert ratelimit.allow("client-a") is True
    assert ratelimit.allow("client-a") is True
    assert ratelimit.allow("client-a") is False             # 4th within the window: blocked
    assert ratelimit.allow("client-b") is True               # separate client, separate bucket


def test_ratelimit_middleware_429s_and_exempts_health(monkeypatch):
    from prove_or_abstain import ratelimit
    monkeypatch.setattr(ratelimit, "LIMIT_PER_MINUTE", 2)
    ratelimit.reset()
    for _ in range(5):
        assert client.get("/health").status_code == 200      # /health is exempt, never blocked

    assert client.get("/panels/clean").status_code == 200    # 1st normal call
    assert client.get("/panels/clean").status_code == 200    # 2nd
    r = client.get("/panels/clean")                          # 3rd: over the limit of 2
    assert r.status_code == 429
    assert "rate limit" in r.json()["detail"]


# ---------------------------------------------------------------- webhook
def test_webhook_notify_without_url_logs_and_returns_false(monkeypatch, caplog):
    from prove_or_abstain import webhook
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    with caplog.at_level("INFO", logger="prove_or_abstain.webhook"):
        sent = webhook.notify("conversion", "ASSERT", 0.79, "segment=paid",
                              "EXECUTE", "pause the paid campaign")
    assert sent is False
    assert "no WEBHOOK_URL set" in caplog.text
    assert "segment=paid" in caplog.text          # the payload itself was logged


def test_webhook_notify_posts_to_slack_shaped_url(monkeypatch):
    from prove_or_abstain import webhook
    monkeypatch.setenv("WEBHOOK_URL", "https://hooks.slack.com/services/x/y/z")
    posted = {}

    class _FakeResp:
        status_code = 200

    class _FakeRequests:
        RequestException = Exception
        @staticmethod
        def post(url, json, timeout):
            posted["url"], posted["json"] = url, json
            return _FakeResp()

    monkeypatch.setattr(webhook, "requests", _FakeRequests)
    sent = webhook.notify("conversion", "ASSERT", 0.79, "segment=paid",
                          "EXECUTE", "pause the paid campaign")
    assert sent is True
    assert posted["url"] == "https://hooks.slack.com/services/x/y/z"
    assert "attachments" in posted["json"]          # Slack-shaped payload


# ----------------------------------------------------------- Google Sheets
def test_gsheets_url_normalization():
    from prove_or_abstain.connectors.gsheets import _to_csv_url

    edit = "https://docs.google.com/spreadsheets/d/ABC123/edit#gid=42"
    assert _to_csv_url(edit) == "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=42"

    share = "https://docs.google.com/spreadsheets/d/ABC123/edit?usp=sharing"
    assert _to_csv_url(share) == "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=0"

    already_csv = "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv&gid=7"
    assert _to_csv_url(already_csv) == already_csv


def test_gsheets_rejects_non_google_host():
    from prove_or_abstain.connectors.gsheets import SheetError, _to_csv_url
    with pytest.raises(SheetError):
        _to_csv_url("https://evil.example.com/spreadsheets/d/ABC123/edit")


def test_gsheets_fetch_panel_mocked(monkeypatch):
    from prove_or_abstain.connectors import gsheets

    class FakeResp:
        text = "metric,segment,n,c\nconversion,paid,100,7\n"
        def raise_for_status(self): pass

    monkeypatch.setattr(gsheets.requests, "get", lambda url, timeout=15: FakeResp())
    df = gsheets.fetch_panel("https://docs.google.com/spreadsheets/d/ABC123/edit")
    assert list(df.columns) == ["metric", "segment", "n", "c"]
    assert df.iloc[0]["c"] == 7


def test_gsheets_fetch_panel_rejects_private_sheet(monkeypatch):
    from prove_or_abstain.connectors import gsheets

    class FakeResp:
        text = "<html>sign in</html>"
        def raise_for_status(self): pass

    monkeypatch.setattr(gsheets.requests, "get", lambda url, timeout=15: FakeResp())
    with pytest.raises(gsheets.SheetError):
        gsheets.fetch_panel("https://docs.google.com/spreadsheets/d/ABC123/edit")


def test_sheets_endpoint_matches_upload(monkeypatch):
    from prove_or_abstain.panels import BASELINE, CLEAN

    def fake_fetch(url):
        return BASELINE if "baseline" in url else CLEAN

    monkeypatch.setattr("api.app.fetch_sheet_panel", fake_fetch)
    body = client.post("/investigate/sheets", json={
        "baseline_url": "https://docs.google.com/spreadsheets/d/baseline/edit",
        "current_url": "https://docs.google.com/spreadsheets/d/current-clean/edit",
    }).json()
    assert body["panel"] == "sheets"
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_sheets_endpoint_rejects_non_google_url():
    r = client.post("/investigate/sheets", json={
        "baseline_url": "https://evil.example.com/x",
        "current_url": "https://evil.example.com/y",
    })
    assert r.status_code == 400


# --------------------------------------------------------------- panel data
def test_panel_data_matches_source():
    from prove_or_abstain.panels import BASELINE, CLEAN
    body = client.get("/panels/clean").json()
    assert body["panel"] == "clean"
    assert len(body["baseline"]) == len(BASELINE)
    assert len(body["current"]) == len(CLEAN)
    assert set(body["baseline"][0].keys()) == {"metric", "segment", "device", "n", "c"}


def test_panel_data_rejects_unknown_panel():
    assert client.get("/panels/bogus").status_code == 422


def test_upload_response_echoes_dataset():
    from prove_or_abstain.panels import BASELINE, CLEAN
    r = client.post("/investigate/upload", files={
        "baseline": ("baseline.csv", _csv(BASELINE), "text/csv"),
        "current": ("current.csv", _csv(CLEAN), "text/csv"),
    })
    body = r.json()
    assert len(body["dataset"]["baseline"]) == len(BASELINE)
    assert len(body["dataset"]["current"]) == len(CLEAN)


# ------------------------------------------------- agent mode (Qwen loop)
# The agent mode lets Qwen orchestrate the investigation via tool calls. Its
# whole promise is that the VERDICT is unchanged — Qwen drives the path, the
# gates decide the outcome. These tests are the safety net for that claim.

def test_agent_mode_matches_graph_on_every_panel():
    # Same verdict, root cause, confidence and action as the fixed pipeline.
    for panel in ("clean", "diffuse", "mixshift", "deep"):
        g = client.post("/investigate", json={"panel": panel, "autopilot": True,
                                              "mode": "graph"}).json()
        a = client.post("/investigate", json={"panel": panel, "autopilot": True,
                                              "mode": "agent"}).json()
        assert a["verdict"] == g["verdict"], panel
        assert a["root_cause"] == g["root_cause"], panel
        assert a["action"]["kind"] == g["action"]["kind"], panel
        assert a["confidence"] == pytest.approx(g["confidence"]), panel


def test_agent_trace_present_only_in_agent_mode():
    g = client.post("/investigate", json={"panel": "clean", "mode": "graph"}).json()
    a = client.post("/investigate", json={"panel": "clean", "mode": "agent"}).json()
    assert g["agent_trace"] == []
    tools = [s["tool"] for s in a["agent_trace"]]
    assert "test_dimension" in tools  # Qwen (mock driver) actually called a tool


def test_agent_verdict_independent_of_llm_path():
    # THE guarantee: whatever tool sequence Qwen chooses — reordering, skipping
    # dimensions, or finalizing immediately without testing anything — the
    # determinism guard yields the exact same verdict as the graph. A false
    # ABSTAIN from a lazy LLM is impossible.
    import json as _json
    from prove_or_abstain.panels import BASELINE as PB, CLEAN as PC
    from prove_or_abstain.llm import template_report, template_speculations
    import prove_or_abstain.agent_loop as al

    base = dict(baseline=PB, current=PC, metrics=["conversion", "activation"],
                metric_kinds={}, dims=["device", "segment"],
                autopilot_enabled=True, trace=[])
    ref = client.post("/investigate",
                      json={"panel": "clean", "autopilot": True}).json()

    class _Stub:
        def __init__(self, script):
            self.mock = False; self.model = "stub"
            self.last_mode = "real"; self.last_error = None
            self._script = script; self._i = 0

        def chat_with_tools(self, messages, tools, **kw):
            step = self._script[self._i]; self._i += 1
            calls = [{"id": f"c{i}", "name": n, "arguments": a}
                     for i, (n, a) in enumerate(step)]
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": _json.dumps(c["arguments"])}}
                for c in calls]}
            return {"message": msg, "tool_calls": calls, "content": ""}

        def speculate_causes(self, p): return template_speculations(p)
        def write_report(self, p): return template_report(p)

    scripts = {
        "reverse": [[("test_dimension", {"dimension": "segment"})],
                    [("test_dimension", {"dimension": "device"})], [("finalize", {})]],
        "skip": [[("test_dimension", {"dimension": "segment"})], [("finalize", {})]],
        "lazy": [[("finalize", {})]],
        "garbage": [[("test_dimension", {"dimension": "nope"})], [("finalize", {})]],
    }
    orig = al.get_client
    try:
        for label, script in scripts.items():
            al.get_client = lambda s=_Stub(script): s
            out = al.investigate_agentic(dict(base))
            assert out["verdict"] == ref["verdict"], label
            assert out["winning_dim"] == ref["root_cause"]["dimension"], label
            assert out["confidence"] == pytest.approx(ref["confidence"]), label
    finally:
        al.get_client = orig


# ---------------------------------------------------- memory (SQLite) layer

def test_memory_records_and_dedupes_alerts():
    from prove_or_abstain import memory
    memory.reset()
    iid = memory.record_investigation("conversion", ["segment"], "ASSERT", 0.8,
                                      "segment=paid", "r", ["t"], "agent", "qwen-plus")
    assert memory.get_investigation(iid)["verdict"] == "ASSERT"
    a1 = memory.create_alert(iid, "conversion", "segment=paid", "EXECUTE", "d")
    a2 = memory.create_alert(iid, "conversion", "segment=paid", "EXECUTE", "d")
    assert a1 == a2                                   # deduped: same metric×cause
    assert len(memory.get_active_alerts()) == 1
    assert memory.resolve_alert(a1) is True
    assert memory.resolve_alert(a1) is False          # already resolved
    assert len(memory.get_active_alerts()) == 0
    stats = memory.get_stats()
    assert stats["total_investigations"] == 1 and stats["asserts"] == 1


def test_autopilot_adapter_survives_on_memory():
    # The SQLite-backed adapter still produces the legacy Execution/Dashboard.
    from prove_or_abstain import memory, autopilot
    memory.reset()
    autopilot.record_execution("conversion", "segment", "paid", 0.79,
                               "EXECUTE", "pause", "report", ["t1"])
    execs = autopilot.get_executions()
    assert "conversion:segment=paid" in execs
    dash = autopilot.get_dashboard()
    assert dash.total_executions == 1
    assert dash.active_alerts[0]["detail"] == "pause"


def test_record_check_thread_safe_counter(monkeypatch):
    """/investigate/check runs record_check() on a sync endpoint, so
    concurrent calls are genuinely possible under FastAPI's thread pool.
    _TOTAL_CHECKS += 1 is a read-modify-write; without a lock, concurrent
    increments can race and undercount."""
    import threading
    from prove_or_abstain import autopilot
    monkeypatch.setattr(autopilot, "_TOTAL_CHECKS", 0)
    monkeypatch.setattr(autopilot, "_LAST_CHECK", None)
    monkeypatch.setattr(autopilot, "_LAST_VERDICT", None)

    n = 50
    barrier = threading.Barrier(n)

    def hit():
        barrier.wait()
        autopilot.record_check("ASSERT")

    threads = [threading.Thread(target=hit) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert autopilot.get_dashboard().total_checks == n   # no increment lost


# --------------------------------------------------------- benchmark harness

def test_benchmark_high_accuracy_offline():
    from prove_or_abstain.benchmark import build_scenarios, run_benchmark
    assert len(build_scenarios()) == 30
    for mode in ("graph", "agent"):
        m = run_benchmark(mode, verbose=False)
        assert m["n"] == 30
        assert m["accuracy"] >= 0.9, [r for r in m["records"] if not r["correct"]]
        assert m["false_abstain_rate"] == 0.0     # never miss a real cause
        assert all("confidence" in r for r in m["records"])  # feeds calibration


def test_benchmark_writes_inspectable_json(tmp_path):
    from prove_or_abstain.benchmark import _write_results_json, run_benchmark
    graph_m = run_benchmark("graph", verbose=False)
    agent_m = run_benchmark("agent", verbose=False)
    live_evals = {"skipped": "needs DASHSCOPE_API_KEY (and QWEN_MOCK unset)"}
    out = _write_results_json(graph_m, agent_m, live_evals, path=tmp_path / "benchmark_results.json")
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["graph"]["n"] == 30 and payload["agent"]["n"] == 30
    assert payload["live_evals"] == live_evals
    assert "generated_at" in payload


# ----------------------------------------------------- autonomous monitor

def _run(coro):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(coro)


def test_monitor_detects_and_records():
    from prove_or_abstain import memory
    from prove_or_abstain.monitor import MetricMonitor
    from prove_or_abstain.panels import BASELINE, CLEAN
    memory.reset()
    source = {
        "type": "inline", "config": {"current": BASELINE},
        "metrics": ["conversion", "activation"], "dims": ["device", "segment"]}
    mon = MetricMonitor(sources=[source])
    first = _run(mon.check_once())               # cold start: nothing to compare yet
    assert first[0]["verdict"] == "BASELINE_SET"

    source["config"]["current"] = CLEAN
    out = _run(mon.check_once())                 # compares against the persisted reference
    assert out[0]["verdict"] == "ASSERT" and out[0]["cause"] == "segment=paid"
    assert len(memory.get_history()) == 1
    assert len(memory.get_active_alerts()) == 1   # confident ASSERT -> alert


def test_monitor_seeds_baseline_then_no_anomaly():
    from prove_or_abstain import memory
    from prove_or_abstain.monitor import MetricMonitor
    from prove_or_abstain.panels import BASELINE
    memory.reset()
    # no inline baseline -> first cycle only seeds it, second sees no movement
    mon = MetricMonitor(sources=[{
        "type": "inline", "config": {"current": BASELINE},
        "metrics": ["conversion"], "dims": ["device", "segment"]}])
    first = _run(mon.check_once())
    assert first[0]["verdict"] == "BASELINE_SET"
    second = _run(mon.check_once())
    assert second[0]["verdict"] == "NO_ANOMALY"


def test_monitor_survives_bad_source():
    from prove_or_abstain.monitor import MetricMonitor
    mon = MetricMonitor(sources=[{
        "type": "csv", "config": {"path": "/nonexistent/nope.csv"},
        "metrics": ["conversion"], "dims": ["device"]}])
    out = _run(mon.check_once())          # must not raise
    assert "error" in out[0]


def test_monitor_broken_source_does_not_corrupt_the_healthy_ones_history():
    """One source fails every cycle; the other must keep accumulating its
    own persisted observation history unaffected (proves 'Watch a source'
    persistence, not just an in-memory dict, survives partial failures)."""
    from prove_or_abstain import memory
    from prove_or_abstain.monitor import MetricMonitor
    from prove_or_abstain.panels import BASELINE, CLEAN
    memory.reset()
    good = {"type": "inline", "config": {"current": BASELINE},
            "metrics": ["conversion"], "dims": ["device", "segment"],
            "source_id": "good"}
    bad = {"type": "csv", "config": {"path": "/nonexistent/nope.csv"},
          "metrics": ["conversion"], "dims": ["device"], "source_id": "bad"}
    mon = MetricMonitor(sources=[good, bad])

    first = _run(mon.check_once())
    assert first[0]["verdict"] == "BASELINE_SET" and "error" in first[1]
    assert memory.count_observations("good") == 1

    good["config"]["current"] = CLEAN
    second = _run(mon.check_once())
    assert second[0]["verdict"] == "ASSERT" and "error" in second[1]
    assert memory.count_observations("good") == 2   # unaffected by "bad"'s failures


# --------------------------------------------------- watch a source (ingestion)

def test_source_cold_start_seeds_without_a_verdict():
    from prove_or_abstain import memory
    from prove_or_abstain.panels import BASELINE
    memory.reset()
    r = client.post("/sources/coldtest/observe",
                    files={"panel": ("baseline.csv", _csv(BASELINE), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["cold_start"] is True
    assert body["verdict"] == "BASELINE_SET"
    assert memory.count_observations("coldtest") == 1


def test_reference_window_pools_prior_observations_like_split_series():
    """Pooling oracle: N observations of the same shape sum their raw counts
    cell-by-cell, exactly like panels.py::split_series's window pooling —
    the algebra this function deliberately duplicates (see reference.py)."""
    from prove_or_abstain import memory, reference
    memory.reset()
    obs = pd.DataFrame([{"metric": "conversion", "segment": "paid", "n": 1000, "c": 70},
                        {"metric": "conversion", "segment": "organic", "n": 2000, "c": 100}])
    for _ in range(3):
        memory.record_observation("pooltest", obs, ["segment"], ["conversion"])

    pooled = reference.build_reference_window("pooltest", ["segment"])
    pooled = pooled.set_index("segment")
    assert pooled.loc["paid", "n"] == 3000 and pooled.loc["paid", "c"] == 210
    assert pooled.loc["organic", "n"] == 6000 and pooled.loc["organic", "c"] == 300


def test_reference_window_respects_window_size():
    from prove_or_abstain import memory, reference
    memory.reset()
    for n in (1000, 2000, 3000):
        obs = pd.DataFrame([{"metric": "conversion", "segment": "paid", "n": n, "c": n // 10}])
        memory.record_observation("windowtest", obs, ["segment"], ["conversion"])

    all_pooled = reference.build_reference_window("windowtest", ["segment"])
    last_one = reference.build_reference_window("windowtest", ["segment"], window=1)
    assert all_pooled["n"].iloc[0] == 6000          # 1000+2000+3000
    assert last_one["n"].iloc[0] == 3000             # only the most recent


def test_reference_window_raises_with_no_prior_observations():
    from prove_or_abstain import memory, reference
    memory.reset()
    with pytest.raises(ValueError):
        reference.build_reference_window("never-seen", ["segment"])


def test_watch_a_source_end_to_end_localizes_paid():
    from prove_or_abstain import memory
    from prove_or_abstain.panels import BASELINE, CLEAN
    memory.reset()
    client.post("/sources/e2e/observe",
               files={"panel": ("b.csv", _csv(BASELINE), "text/csv")})
    r = client.post("/sources/e2e/observe",
                    files={"panel": ("c.csv", _csv(CLEAN), "text/csv")})
    body = r.json()
    assert body["cold_start"] is False
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "segment", "segment": "paid"}


def test_memory_reset_wipes_observations():
    from prove_or_abstain import memory
    memory.reset()
    df = pd.DataFrame([{"metric": "conversion", "segment": "paid", "n": 100, "c": 10}])
    memory.record_observation("wipeme", df, ["segment"], ["conversion"])
    assert memory.count_observations("wipeme") == 1
    memory.reset()
    assert memory.count_observations("wipeme") == 0


# --------------------------------------------------- schema mapping (Qwen)

def test_map_schema_never_invents_a_column_name():
    from prove_or_abstain.llm import QwenClient
    cols = ["category", "platform", "total_users", "total_conversions"]
    bad = {"metric_column": "nonexistent", "n_column": "total_users",
          "c_column": "made_up", "dim_columns": ["category", "invented_dim"]}
    out = QwenClient._guarded_schema_mapping(bad, cols)
    assert out["metric_column"] is None
    assert out["c_column"] is None
    assert out["n_column"] == "total_users"
    assert set(out["dim_columns"]) == {"category"}


def test_get_client_is_thread_safe(monkeypatch):
    """FastAPI runs sync endpoints in a thread pool, so concurrent requests
    can hit get_client()'s check-then-set on the very first call at once.
    Without a lock, two threads can each see _CLIENT is None and both
    construct a QwenClient, silently discarding one's cost tracker. A
    Barrier maximizes the chance every thread reads _CLIENT as None at the
    same instant, which is exactly the window the lock closes."""
    import threading
    from prove_or_abstain import llm
    monkeypatch.setattr(llm, "_CLIENT", None)

    n = 25
    barrier = threading.Barrier(n)
    results = [None] * n

    def grab(i):
        barrier.wait()
        results[i] = llm.get_client()

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(r) for r in results}) == 1               # every thread got the SAME instance


def test_map_schema_mock_identifies_standard_columns():
    from prove_or_abstain.llm import template_map_schema
    cols = ["metric", "category", "platform", "total_users", "total_conversions"]
    out = template_map_schema(cols, [])
    assert out["metric_column"] == "metric"
    assert out["n_column"] == "total_users"
    assert out["c_column"] == "total_conversions"
    assert set(out["dim_columns"]) == {"category", "platform"}
    assert out["self_verified"] is True


def test_map_schema_self_verification_corrects_first_pass(monkeypatch):
    """Real-mode map_schema is 2 passes: propose, then self-verify. This
    simulates a first pass that swaps n/c by mistake, and a second pass
    that catches and corrects it — proving map_schema() returns the
    CORRECTED mapping, not the first guess."""
    import json as _json
    from prove_or_abstain.llm import QwenClient

    responses = [
        _json.dumps({"dim_columns": ["category", "platform"], "metric_column": "metric",
                     "n_column": "total_conversions", "c_column": "total_users",
                     "reason": "first guess"}),
        _json.dumps({"dim_columns": ["category", "platform"], "metric_column": "metric",
                     "n_column": "total_users", "c_column": "total_conversions",
                     "self_verified": False, "reason": "corrected an n/c swap"}),
    ]
    calls = {"i": 0}

    def fake_complete(self, system, user, **kw):
        i = calls["i"]
        calls["i"] += 1
        return responses[i]

    monkeypatch.setattr(QwenClient, "complete", fake_complete)
    qc = QwenClient(mock=False, model="stub")
    out = qc.map_schema(
        ["metric", "category", "platform", "total_users", "total_conversions"],
        [{"metric": "conversion", "category": "paid", "platform": "mobile",
          "total_users": 3000, "total_conversions": 210}])
    assert calls["i"] == 2                    # both passes actually ran
    assert out["n_column"] == "total_users"    # the CORRECTED mapping, not the first
    assert out["c_column"] == "total_conversions"
    assert out["self_verified"] is False


def test_schema_mapping_rejects_incoherent_mapping_before_the_math():
    """Data map_schema can't make sense of (no metric/n/c hints at all) is
    rejected by _validate_panel before it ever reaches gates.py — mock or
    real, the deterministic backstop is the same."""
    df = pd.DataFrame({"foo": ["a", "b"], "bar": ["c", "d"], "baz": [1, 2]})
    r = client.post("/sources/badmapping/observe",
                    files={"panel": ("bad.csv", _csv(df), "text/csv")})
    assert r.status_code == 400


def test_schema_mapping_end_to_end_via_examples(tmp_path):
    """examples/schema_mapping_seed.csv + _next.csv: non-standard column
    names (category/platform/total_users/total_conversions), non-round
    sample sizes (jittered, not a flat 5000/3000/1500/500), mapped via
    map_schema() (mock mode) onto the long-panel contract, localizing to a
    referral-segment collapse — deliberately a different segment and a
    different drop magnitude than panels.CLEAN, so this isn't just the same
    scenario relabeled."""
    from prove_or_abstain import memory
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "examples"
    memory.reset()
    seed = (root / "schema_mapping_seed.csv").read_bytes()
    nxt = (root / "schema_mapping_next.csv").read_bytes()
    client.post("/sources/rawschema/observe", files={"panel": ("seed.csv", seed, "text/csv")})
    r = client.post("/sources/rawschema/observe", files={"panel": ("next.csv", nxt, "text/csv")})
    body = r.json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "category", "segment": "referral"}


# ----------------------------------------------- real, external data (not synthetic)
# Every other example in examples/ is hand-constructed with a known, planted
# cause — necessary to prove the math against a ground truth, but every
# scenario ends up suspiciously round (n in the thousands, a 7%->5% drop
# reused verbatim in more than one place). These three are real public
# datasets (seaborn-data's flights.csv and titanic.csv, MIT-licensed;
# fivethirtyeight/data's recent-grads.csv, CC BY 4.0), reshaped into the
# long-panel contract but with none of the numbers invented: real group
# sizes (7 to 3.5M), real noise, no planted answer.

def test_real_flights_1960_growth_is_systemic_not_seasonal():
    """seaborn-data/flights.csv: real monthly airline passenger counts,
    1949-1960. 1960 grew +11.2% over 1959 (a real trend — postwar air
    travel boom) but the growth is NOT concentrated in any one month
    (concentration well under 0.55) — a genuine ABSTAIN, not a planted one."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "examples"
    r = client.post("/investigate/series",
                    files={"series": ("flights.csv",
                          (root / "real_flights_series.csv").read_bytes(), "text/csv")},
                    data={"window": "1", "sum_metrics": "passengers"})
    body = r.json()
    assert body["verdict"] == "ABSTAIN"
    assert body["gates"]["month"]["concentration"] < 0.3


def test_real_titanic_survival_gap_localizes_to_sex_not_class():
    """seaborn-data/titanic.csv: real passenger manifest. Comparing
    Southampton (n=644) against Cherbourg (n=168) passengers, overall
    survival jumps from 34% to 55%. Historically this is usually attributed
    to Cherbourg carrying more 1st-class passengers — but decomposed
    honestly, 'pclass' alone does NOT clear the significance gate (p=0.10,
    small samples once split further), while 'sex' does (p=0.0018): the
    well-documented 'women and children first' effect dominates. Confidence
    is genuinely low (real data, not a clean planted scenario) — an honest
    RECOMMEND, not a confident EXECUTE."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "examples"
    r = client.post("/investigate/upload", files={
        "baseline": ("southampton.csv",
                    (root / "real_titanic_southampton.csv").read_bytes(), "text/csv"),
        "current": ("cherbourg.csv",
                   (root / "real_titanic_cherbourg.csv").read_bytes(), "text/csv"),
    })
    body = r.json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "sex", "segment": "female"}
    assert body["gates"]["pclass"]["verdict"] == "ABSTAIN"     # the popular guess doesn't hold up
    assert body["confidence"] < 0.3                            # real data: honestly uncertain
    assert body["action"]["kind"] == "RECOMMEND"               # too low-confidence to auto-execute


def test_real_college_majors_stem_gap_localizes_to_gender_majority():
    """fivethirtyeight/college-majors' recent-grads.csv: 173 real US majors,
    each with real (not invented) total/employed counts. Splitting on the
    dataset's own STEM classification (Engineering, Computers & Mathematics,
    Biology & Life Science, Physical Sciences, Agriculture & Natural
    Resources vs. everything else — a standard categorization, not one we
    picked to force a result) drops the employment rate 81.0% -> 75.0%. That
    move is NOT uniform: splitting further by the majority gender of each
    major (also a real, pre-existing field: ShareWomen) shows the drop
    concentrating in majority-women majors (81.5% -> 68.8%) far more than
    majority-men ones (79.9% -> 79.2%) — a real, un-planted finding, with
    genuinely low confidence to show for it, same as the Titanic case above."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "examples"
    r = client.post("/investigate/upload", files={
        "baseline": ("nonstem.csv",
                    (root / "real_majors_nonstem.csv").read_bytes(), "text/csv"),
        "current": ("stem.csv",
                   (root / "real_majors_stem.csv").read_bytes(), "text/csv"),
    })
    body = r.json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "gender_majority", "segment": "majority_women"}
    assert body["confidence"] < 0.1                              # real data: honestly uncertain
    assert body["action"]["kind"] == "RECOMMEND"                 # too low-confidence to auto-execute


def test_real_college_majors_raw_columns_need_qwens_schema_mapping(monkeypatch):
    """Same real dataset, but with its columns left as a plausible raw export
    would actually look (Field/Group/Total/Employed) instead of hand-renamed
    to metric/dim/n/c. `template_map_schema()` (the deterministic mock/
    fallback heuristic) genuinely cannot solve this: 'Employed' isn't in its
    keyword list (success/conversion/converted/purchase/click), so the
    mapping comes back incomplete and /sources/.../observe correctly 400s
    rather than guessing. This is the honest gap the README points at —
    "a real source with truly unusual column names" — not a synthetic one
    engineered to already contain a matching keyword. Once given the mapping
    a competent read of the columns/sample-rows would produce (which is
    exactly what map_schema()'s real-mode Qwen call, prompted the same way,
    is asked to return), the downstream math recovers the identical verdict
    proven above with the pre-renamed CSVs — the ONLY thing standing between
    'unreadable' and 'correct' here is that one mapping decision."""
    import prove_or_abstain.llm as llm
    from pathlib import Path
    from prove_or_abstain import memory
    root = Path(__file__).resolve().parent.parent / "examples"
    raw_nonstem = (root / "real_majors_nonstem_raw.csv").read_bytes()
    raw_stem = (root / "real_majors_stem_raw.csv").read_bytes()

    mapping = llm.template_map_schema(
        ["Field", "Group", "Total", "Employed"],
        [{"Field": "Employment", "Group": "majority_men", "Total": 1660894, "Employed": 1327119}])
    assert mapping["metric_column"] is None and mapping["c_column"] is None   # genuinely stuck

    memory.reset()
    r = client.post("/sources/majors_raw_mock/observe",
                    files={"panel": ("nonstem.csv", raw_nonstem, "text/csv")})
    assert r.status_code == 400                              # honest failure, not a guess

    correct_mapping = {"dim_columns": ["Group"], "metric_column": "Field",
                       "n_column": "Total", "c_column": "Employed",
                       "self_verified": True, "reason": "test: the mapping a correct read produces"}
    memory.reset()
    monkeypatch.setattr(llm.QwenClient, "map_schema", lambda self, columns, sample_rows: correct_mapping)
    client.post("/sources/majors_raw_mapped/observe",
                files={"panel": ("nonstem.csv", raw_nonstem, "text/csv")})
    r = client.post("/sources/majors_raw_mapped/observe",
                    files={"panel": ("stem.csv", raw_stem, "text/csv")})
    body = r.json()
    assert body["verdict"] == "ASSERT"
    assert body["root_cause"] == {"dimension": "Group", "segment": "majority_women"}


# --------------------------------------------------- SSE streaming + fallback

def test_sse_streams_events_in_order():
    events = []
    with client.stream("GET", "/investigate/stream?panel=clean&autopilot=true") as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
    assert events[0] == "detector" and events[-1] == "done"
    assert "gate_result" in events and "verdict" in events


def test_agent_recovers_when_every_tool_errors(monkeypatch):
    # item 8: even if every test_dimension tool call raises, the determinism
    # sweep (which uses the math directly) still reaches the correct verdict.
    import json as _json
    import prove_or_abstain.agent_loop as al
    from prove_or_abstain.panels import BASELINE as PB, CLEAN as PC
    from prove_or_abstain.llm import template_report, template_speculations

    ref = client.post("/investigate", json={"panel": "clean", "autopilot": True}).json()
    monkeypatch.setattr(al, "_test_dimension",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    class _Stub:
        mock = False; model = "stub"; last_mode = "real"; last_error = None
        _script = [[("test_dimension", {"dimension": "device"})],
                   [("test_dimension", {"dimension": "segment"})], [("finalize", {})]]
        _i = 0

        def chat_with_tools(self, messages, tools, **kw):
            step = self._script[self._i]; self._i += 1
            calls = [{"id": f"c{j}", "name": n, "arguments": a}
                     for j, (n, a) in enumerate(step)]
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": _json.dumps(c["arguments"])}}
                for c in calls]}
            return {"message": msg, "tool_calls": calls, "content": ""}

        def speculate_causes(self, p): return template_speculations(p)
        def write_report(self, p): return template_report(p)

    monkeypatch.setattr(al, "get_client", lambda: _Stub())
    out = al.investigate_agentic(dict(baseline=PB, current=PC,
        metrics=["conversion", "activation"], metric_kinds={},
        dims=["device", "segment"], autopilot_enabled=True, trace=[]))
    assert out["verdict"] == ref["verdict"]
    assert out["winning_dim"] == ref["root_cause"]["dimension"]


# ----------------------------------------------------------- cost tracking

def test_cost_tracker_pricing():
    from prove_or_abstain.cost_tracker import CostTracker
    t = CostTracker("qwen-plus")
    t.add_usage(1_000_000, 1_000_000)
    assert t.cost_usd == pytest.approx(0.80 + 2.40)   # $/1M in + out
    assert t.to_dict()["total_tokens"] == 2_000_000


def test_api_reports_cost_zero_in_mock():
    body = client.post("/investigate", json={"panel": "clean", "mode": "agent"}).json()
    assert body["cost"]["usd"] == 0.0 and body["cost"]["tokens"] == 0


# ------------------------------------------------------------- calibration

def test_calibration_perfect_set_has_low_ece():
    from prove_or_abstain.calibrate import calibrate_confidence
    # 9/10 correct at confidence 0.9 => accuracy == confidence => ECE ~ 0
    recs = [{"got": "ASSERT", "confidence": 0.9, "correct": i < 9} for i in range(10)]
    cal = calibrate_confidence(recs)
    assert cal["n"] == 10
    assert cal["ece"] < 0.02


def test_calibration_over_benchmark():
    from prove_or_abstain.benchmark import run_benchmark
    from prove_or_abstain.calibrate import calibrate_confidence
    cal = calibrate_confidence(run_benchmark("agent", verbose=False)["records"])
    assert cal["n"] > 0
    assert 0.0 <= cal["ece"] <= 1.0
    assert sum(b["count"] for b in cal["buckets"]) == cal["n"]


# ---------------------------------------------------------- audit trail

def _clean_state():
    from prove_or_abstain.panels import BASELINE, CLEAN
    return {"baseline": BASELINE, "current": CLEAN,
            "metrics": ["conversion", "activation"], "metric_kinds": {},
            "dims": ["device", "segment"], "autopilot_enabled": True, "trace": []}


def test_audit_trail_has_gates_and_replays():
    from prove_or_abstain.agent_loop import investigate_agentic
    from prove_or_abstain.audit import create_audit_trail, verify_replay
    st = _clean_state()
    final = investigate_agentic(dict(st))
    trail = create_audit_trail(final, final["reports_by_dim"],
                               final["agent_trace"], final["llm"]["model"], "agent")
    assert set(trail["gates"]) == {"material", "localized", "significant", "clean"}
    assert trail["verdict"] == "ASSERT" and trail["cause"] == "segment=paid"
    assert len(trail["input_hash"]) == 64                 # SHA256 hex
    # a fresh run reproduces it exactly
    assert verify_replay(trail, investigate_agentic(dict(st))) is True


def test_audit_replay_detects_tampering():
    from prove_or_abstain.agent_loop import investigate_agentic
    from prove_or_abstain.audit import create_audit_trail, verify_replay
    st = _clean_state()
    final = investigate_agentic(dict(st))
    trail = create_audit_trail(final, final["reports_by_dim"],
                               final["agent_trace"], final["llm"]["model"], "agent")
    trail["confidence"] = 0.123                            # tamper
    assert verify_replay(trail, investigate_agentic(dict(st))) is False


def test_audit_trail_significant_flag_correct_for_sum_metrics():
    """A "sum" metric's significance gate is a sample-floor check
    (gates.py:evaluate_gates), not a z-test, so leading_p stays NaN even
    when the gate genuinely passed. The audit trail must reflect the real
    gate outcome (GateReport.significant), not silently report False for
    every sum-kind ASSERT just because leading_p is unusable here."""
    from prove_or_abstain.agent_loop import investigate_agentic
    from prove_or_abstain.audit import create_audit_trail
    from prove_or_abstain.benchmark import _revenue_panel
    base, curr = _revenue_panel("paid")
    st = {"baseline": base, "current": curr, "metrics": ["revenue"],
          "metric_kinds": {"revenue": "sum"}, "dims": ["segment", "device"],
          "autopilot_enabled": False, "trace": []}
    final = investigate_agentic(dict(st))
    assert final["verdict"] == "ASSERT"                    # sanity: the scenario asserts
    win = final["winning_report"]
    assert math.isnan(win.leading_p)                       # no z-test ran for "sum"
    assert win.significant is True                         # but the real gate passed

    trail = create_audit_trail(final, final["reports_by_dim"],
                               final["agent_trace"], final["llm"]["model"], "agent")
    assert trail["gates"]["significant"] is True            # must match, not silently False


# ------------------------------------------------ robust JSON parsing (llm)

def test_robust_parse_json_recovers_or_falls_back():
    from prove_or_abstain.llm import QwenClient
    c = QwenClient(mock=True)
    assert c._robust_parse_json('["device","segment"]', None) == ["device", "segment"]
    assert c._robust_parse_json("```json\n[\"a\",\"b\"]\n```", None) == ["a", "b"]
    # JSON embedded in prose -> outermost-brace recovery
    assert c._robust_parse_json('Here: {"panel":"clean"} done', None) == {"panel": "clean"}
    # no valid JSON, but known fields -> regex recovery
    got = c._robust_parse_json("verdict=ASSERT confidence: 0.83", "FB")
    assert got == {"verdict": "ASSERT", "confidence": 0.83}
    # total garbage -> fallback
    assert c._robust_parse_json("no json here", "FB") == "FB"
