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
