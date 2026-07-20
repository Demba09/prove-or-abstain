"""
benchmark.py — 30 synthetic scenarios with known ground truth.

The ground truth comes from how each panel is GENERATED (a paid-only collapse
=> ASSERT segment=paid; a uniform drop => ABSTAIN systemic), never from the
pipeline's own output — otherwise accuracy would be circular. run_benchmark()
then measures whether the deterministic pipeline agrees.

Offline (QWEN_MOCK=1) everything here runs without a network call:
  - run_benchmark()   : accuracy / precision / recall / false-ASSERT /
                        false-ABSTAIN of the pipeline vs. ground truth.
Live (needs DASHSCOPE_API_KEY), and skipped cleanly without a key:
  - compare_llm_raw() : gives a raw model ONLY a text summary (no data) and
                        counts invented causes — the hallucinations the gated
                        pipeline is built to avoid.
  - cross_model_eval(): the same 30 scenarios on qwen-plus/max/turbo, reporting
                        correctness, latency, tokens and cost. Accuracy is
                        model-independent by design (the math decides), so the
                        table is really about cost/latency.

CLI: python -m prove_or_abstain.benchmark
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prove_or_abstain import panels as P
from prove_or_abstain.agent_loop import investigate_agentic
from prove_or_abstain.graph import APP as GRAPH


# --------------------------------------------------------------- scenario data
@dataclass
class Scenario:
    name: str
    category: str
    baseline: pd.DataFrame
    current: pd.DataFrame
    dims: list[str]
    metrics: list[str]
    metric_kinds: dict
    expected_verdict: str            # "ASSERT" | "ABSTAIN"
    expected_cause: str | None       # "dim=segment" on ASSERT, else None


def _conv(rate_fn, n_map=None) -> pd.DataFrame:
    """A full (conversion + stable activation decoy) panel from a rate fn."""
    return P._rows(rate_fn, lambda s, d: P._ACT0[s], n_map)


def _seg_drop(segment: str, new_rate: float) -> pd.DataFrame:
    return _conv(lambda s, d: new_rate if s == segment else P._RATE0[s])


def _dev_drop(device: str, factor: float) -> pd.DataFrame:
    return _conv(lambda s, d: P._RATE0[s] * (factor if d == device else 1.0))


def _two_seg_drop(a: str, ra: float, b: str, rb: float) -> pd.DataFrame:
    drops = {a: ra, b: rb}
    return _conv(lambda s, d: drops.get(s, P._RATE0[s]))


def _diffuse(delta: float) -> pd.DataFrame:
    return _conv(lambda s, d: P._RATE0[s] - delta)


def _deep(segment: str, device: str, new_rate: float) -> pd.DataFrame:
    return _conv(lambda s, d: new_rate if (s, d) == (segment, device) else P._RATE0[s])


def _mixshift(rates: dict, n_map: dict) -> pd.DataFrame:
    return _conv(lambda s, d: rates[s], n_map)


def _noisy_seg_drop(segment: str, new_rate: float, seed: int) -> pd.DataFrame:
    """A clear segment collapse with small deterministic jitter on the others,
    so confidence lands near — but above — the threshold."""
    import random
    rng = random.Random(seed)
    jitter = {s: 1.0 + rng.uniform(-0.04, 0.04) for s in P.SEGMENTS}
    return _conv(lambda s, d: new_rate if s == segment
                 else P._RATE0[s] * jitter[s])


def _revenue_panel(drop_segment: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A revenue (sum-kind) panel: per (segment, device) totals. If
    drop_segment is set, that segment's revenue halves in current."""
    def rows(scale):
        out = []
        for s in P.SEGMENTS:
            for d in P.DEVICES:
                base = P._N[(s, d)] * P._RATE0[s] * 10        # arbitrary $ scale
                amt = base * (0.5 if s == drop_segment else 1.0) if scale else base
                out.append({"metric": "revenue", "segment": s, "device": d,
                            "n": P._N[(s, d)], "c": round(amt)})
        return pd.DataFrame(out)
    return rows(False), rows(True)


def _tiny_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A real paid drop but on a tiny sample — evidence too weak to ASSERT."""
    n_map = {(s, d): 30 for s in P.SEGMENTS for d in P.DEVICES}
    base = _conv(lambda s, d: P._RATE0[s], n_map)
    curr = _conv(lambda s, d: 0.03 if s == "paid" else P._RATE0[s], n_map)
    return base, curr


def build_scenarios() -> list[Scenario]:
    B = P.BASELINE
    sc: list[Scenario] = []

    def add(name, cat, current, verdict, cause, *, baseline=B,
            dims=("device", "segment"), metrics=("conversion", "activation"),
            kinds=None):
        sc.append(Scenario(name, cat, baseline, current, list(dims),
                           list(metrics), kinds or {}, verdict, cause))

    # 5 clean -> ASSERT segment
    add("seg_paid_hard", "clean_segment", _seg_drop("paid", 0.045), "ASSERT", "segment=paid")
    add("seg_paid_soft", "clean_segment", _seg_drop("paid", 0.052), "ASSERT", "segment=paid")
    add("seg_organic",   "clean_segment", _seg_drop("organic", 0.032), "ASSERT", "segment=organic")
    add("seg_referral",  "clean_segment", _seg_drop("referral", 0.055), "ASSERT", "segment=referral")
    add("seg_email",     "clean_segment", _seg_drop("email", 0.06), "ASSERT", "segment=email")

    # 5 clean -> ASSERT device
    add("dev_mobile_hard",  "clean_device", _dev_drop("mobile", 0.6), "ASSERT", "device=mobile")
    add("dev_mobile_soft",  "clean_device", _dev_drop("mobile", 0.75), "ASSERT", "device=mobile")
    add("dev_desktop_hard", "clean_device", _dev_drop("desktop", 0.6), "ASSERT", "device=desktop")
    add("dev_desktop_soft", "clean_device", _dev_drop("desktop", 0.7), "ASSERT", "device=desktop")
    add("dev_mobile_mid",   "clean_device", _dev_drop("mobile", 0.68), "ASSERT", "device=mobile")

    # 3 clean -> ABSTAIN (no single cause dominates)
    add("split_paid_organic", "clean_abstain",
        _two_seg_drop("paid", 0.056, "organic", 0.042), "ABSTAIN", None)
    add("split_paid_referral", "clean_abstain",
        _two_seg_drop("paid", 0.06, "referral", 0.062), "ABSTAIN", None)
    add("split_three", "clean_abstain",
        _conv(lambda s, d: {"paid": 0.06, "organic": 0.043,
                            "referral": 0.066}.get(s, P._RATE0[s])), "ABSTAIN", None)

    # 5 diffuse -> ABSTAIN systemic
    for i, delta in enumerate([0.006, 0.005, 0.008, 0.004, 0.007]):
        add(f"diffuse_{i}", "diffuse", _diffuse(delta), "ABSTAIN", None)

    # 3 mixshift -> ABSTAIN interaction
    add("mixshift_a", "mixshift", P.MIXSHIFT, "ABSTAIN", None)
    add("mixshift_b", "mixshift",
        _mixshift({"organic": 0.048, "paid": 0.078, "referral": 0.08, "email": 0.10},
                  P._N_MIXSHIFT), "ABSTAIN", None)
    add("mixshift_c", "mixshift",
        _mixshift({"organic": 0.042, "paid": 0.082, "referral": 0.079, "email": 0.11},
                  P._N_MIXSHIFT), "ABSTAIN", None)

    # 3 deep -> ASSERT with drill-down (winning dim is the cell's device)
    add("deep_paid_mobile",    "deep", _deep("paid", "mobile", 0.03), "ASSERT", "device=mobile")
    add("deep_organic_mobile", "deep", _deep("organic", "mobile", 0.02), "ASSERT", "device=mobile")
    add("deep_paid_desktop",   "deep", _deep("paid", "desktop", 0.03), "ASSERT", "device=desktop")

    # 3 edge cases
    rev_b, rev_c = _revenue_panel("paid")
    add("edge_revenue_sum", "edge", rev_c, "ASSERT", "segment=paid",
        baseline=rev_b, metrics=("revenue",), kinds={"revenue": "sum"})
    tiny_b, tiny_c = _tiny_panel()
    add("edge_small_sample", "edge", tiny_c, "ABSTAIN", None, baseline=tiny_b)
    add("edge_single_dim", "edge", _seg_drop("paid", 0.045), "ASSERT", "segment=paid",
        dims=("segment",))

    # 3 noisy -> borderline confidence (still a real paid collapse)
    add("noisy_1", "noisy", _noisy_seg_drop("paid", 0.05, 1), "ASSERT", "segment=paid")
    add("noisy_2", "noisy", _noisy_seg_drop("paid", 0.048, 7), "ASSERT", "segment=paid")
    add("noisy_3", "noisy", _noisy_seg_drop("organic", 0.033, 13), "ASSERT", "segment=organic")

    return sc


# --------------------------------------------------------------- the harness
def _run_one(s: Scenario, mode: str) -> dict:
    state = {"baseline": s.baseline, "current": s.current, "metrics": s.metrics,
             "metric_kinds": s.metric_kinds, "dims": s.dims,
             "autopilot_enabled": False, "trace": []}
    final = investigate_agentic(state) if mode == "agent" else GRAPH.invoke(state)
    win = final.get("winning_report")
    cause = None
    if win is not None:
        cause = f"{final.get('winning_dim')}={win.leading_segment}"
    # A single-cell anomaly straddles two dimensions that BOTH concentrate
    # 100% on it (mathematically inevitable, not a calibration artifact —
    # see the "deep" category). Whichever axis is tested first becomes the
    # top-level cause; the driller always re-decomposes within it and finds
    # the other axis as `refined` — so the full diagnosis is recovered
    # either way, just with the two axes swapped between "cause" and
    # "refinement". Surfaced here so callers can credit either.
    refined = None
    drill = final.get("drilldown") or {}
    if drill.get("refined"):
        r = drill["refined"]
        refined = f"{r['dim']}={r['segment']}"
    return {"verdict": final.get("verdict"), "cause": cause, "refined_cause": refined,
            "confidence": final.get("confidence", 0.0)}


def run_benchmark(mode: str = "graph", verbose: bool = True) -> dict:
    scenarios = build_scenarios()
    records, tp = [], 0
    fp = fn = tn = 0
    correct = 0
    for s in scenarios:
        got = _run_one(s, mode)
        v_ok = got["verdict"] == s.expected_verdict
        # Credit a match on the top-level cause OR the drill-down's refined
        # cause: for a single-cell "deep" anomaly, both of its two defining
        # dimensions concentrate 100% on it (see _run_one) -- whichever is
        # tested first becomes the top-level cause, and the driller always
        # finds the other as `refined`, so either one is the complete,
        # correct diagnosis. Order (hence Qwen's choice) picks which label
        # goes on top; it never loses or invents a cause.
        cause_ok = (s.expected_verdict == "ABSTAIN") or (got["cause"] == s.expected_cause) \
            or (got["refined_cause"] == s.expected_cause)
        ok = v_ok and cause_ok
        correct += ok
        # confusion matrix on the ASSERT/ABSTAIN axis (positive = ASSERT)
        exp_pos = s.expected_verdict == "ASSERT"
        got_pos = got["verdict"] == "ASSERT"
        tp += exp_pos and got_pos
        fp += (not exp_pos) and got_pos
        fn += exp_pos and (not got_pos)
        tn += (not exp_pos) and (not got_pos)
        got_cause_display = got["cause"]
        if got["refined_cause"] and got["cause"] != s.expected_cause:
            got_cause_display = f"{got['cause']} (→ {got['refined_cause']})"
        records.append({"name": s.name, "category": s.category,
                        "expected": s.expected_verdict, "got": got["verdict"],
                        "expected_cause": s.expected_cause, "got_cause": got_cause_display,
                        "confidence": round(got["confidence"], 3), "correct": ok})

    n = len(scenarios)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    cal = calibrate_confidence(records)
    metrics = {
        "mode": mode, "n": n, "accuracy": correct / n,
        "precision": precision, "recall": recall,
        "false_assert_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_abstain_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "ece": cal["ece"], "calibration": cal,
        "records": records,
    }
    if verbose:
        _print_table(records, metrics)
    return metrics


def _print_table(records: list[dict], metrics: dict) -> None:
    print(f"\n### Benchmark — mode={metrics['mode']}, {metrics['n']} scenarios\n")
    print("| scenario | category | expected | got | conf | ✓ |")
    print("|---|---|---|---|---|---|")
    for r in records:
        mark = "✅" if r["correct"] else "❌"
        exp = r["expected"] + (f" {r['expected_cause']}" if r["expected_cause"] else "")
        got = r["got"] + (f" {r['got_cause']}" if r["got_cause"] else "")
        print(f"| {r['name']} | {r['category']} | {exp} | {got} | {r['confidence']} | {mark} |")
    print(f"\n**accuracy={metrics['accuracy']:.1%}  precision={metrics['precision']:.2f}  "
          f"recall={metrics['recall']:.2f}  false-ASSERT={metrics['false_assert_rate']:.1%}  "
          f"false-ABSTAIN={metrics['false_abstain_rate']:.1%}**")


# ------------------------------------------------ live-only (needs a key)
def _has_key() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY")) and os.environ.get("QWEN_MOCK") != "1"


def compare_llm_raw() -> dict:
    """Ask a raw Qwen (no data, only a summary) for the cause, and count how
    often it invents a specific cause where the truth is 'no single cause'
    (systemic) or names the wrong segment. Requires a live key."""
    if not _has_key():
        return {"skipped": "needs DASHSCOPE_API_KEY (and QWEN_MOCK unset)"}
    from prove_or_abstain.llm import QwenClient
    client = QwenClient(mock=False)
    scenarios = build_scenarios()
    hallucinations = 0
    considered = 0
    for s in scenarios:
        n0, c0 = P.metric_totals(s.baseline, s.metrics[0])
        n1, c1 = P.metric_totals(s.current, s.metrics[0])
        r0 = c0 / n0 if n0 else 0
        r1 = c1 / n1 if n1 else 0
        summary = (f"The metric '{s.metrics[0]}' moved from {r0:.3f} to {r1:.3f}. "
                   f"Segments: {P.SEGMENTS}. Devices: {P.DEVICES}. "
                   f"You have NO access to the underlying data. In one line, name the "
                   f"single most likely cause as 'dimension=value', or 'systemic' if "
                   f"none can be isolated.")
        try:
            ans = client.complete("You are a data analyst.", summary,
                                  max_tokens=40).strip().lower()
        except Exception:
            continue
        considered += 1
        names_cause = "=" in ans and "systemic" not in ans
        if s.expected_verdict == "ABSTAIN" and names_cause:
            hallucinations += 1                     # invented a cause where none exists
        elif s.expected_verdict == "ASSERT" and names_cause and \
                s.expected_cause.lower() not in ans:
            hallucinations += 1                     # named the wrong cause
    return {"considered": considered, "hallucinations": hallucinations,
            "hallucination_rate": hallucinations / considered if considered else 0.0}


def cross_model_eval(models=("qwen-turbo", "qwen-plus", "qwen-max")) -> dict:
    """Run the 30 scenarios in agent mode on each model. Verdict accuracy is
    model-independent (the gates decide), so this is really a cost/latency
    comparison. Requires a live key."""
    if not _has_key():
        return {"skipped": "needs DASHSCOPE_API_KEY (and QWEN_MOCK unset)"}
    from prove_or_abstain import llm, cost_tracker
    rows = []
    for model in models:
        client = llm.QwenClient(mock=False, model=model)
        client.tracker = cost_tracker.CostTracker(model)
        llm._CLIENT = client                        # agent_loop uses get_client()
        t0 = time.time()
        m = run_benchmark(mode="agent", verbose=False)
        dt = (time.time() - t0) / m["n"]
        usage = client.tracker.to_dict()
        rows.append({"model": model, "accuracy": m["accuracy"],
                     "avg_latency_s": round(dt, 2),
                     "avg_tokens": usage["total_tokens"] // max(m["n"], 1),
                     "cost_usd": usage["cost_usd"]})
    llm._CLIENT = None
    return {"models": rows}


# ------------------------------------------------------------ calibration
def _bench_positive(rec: dict) -> bool:
    """Calibrate only on ASSERT predictions (ABSTAIN carries confidence 0)."""
    got = rec.get("got")
    return (got == "ASSERT") if got is not None else rec.get("confidence", 0.0) > 0.0


def calibrate_confidence(benchmark_results: list[dict], n_buckets: int = 10) -> dict:
    """Expected Calibration Error on the pipeline's ASSERT predictions."""
    preds = [r for r in benchmark_results if _bench_positive(r)]
    buckets = []
    ece = 0.0
    total = len(preds)
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        in_b = [r for r in preds
                if lo <= r["confidence"] < hi or (b == n_buckets - 1 and r["confidence"] == 1.0)]
        if not in_b:
            buckets.append({"range": (round(lo, 1), round(hi, 1)), "mid": round((lo + hi) / 2, 2),
                            "count": 0, "accuracy": None, "avg_confidence": None})
            continue
        acc = sum(bool(r["correct"]) for r in in_b) / len(in_b)
        avg_conf = sum(r["confidence"] for r in in_b) / len(in_b)
        ece += (len(in_b) / total) * abs(acc - avg_conf) if total else 0.0
        buckets.append({"range": (round(lo, 1), round(hi, 1)), "mid": round((lo + hi) / 2, 2),
                        "count": len(in_b), "accuracy": round(acc, 3),
                        "avg_confidence": round(avg_conf, 3)})
    return {"n": total, "ece": round(ece, 4), "buckets": buckets}


def plot_calibration_curve(benchmark_results: list[dict]) -> str:
    """ASCII reliability diagram: accuracy (#) vs the perfect diagonal (·)."""
    cal = calibrate_confidence(benchmark_results)
    lines = [f"Calibration (ECE={cal['ece']:.3f}, n={cal['n']})",
             "conf │ accuracy (# = actual, · = ideal)"]
    for bkt in cal["buckets"]:
        mid = bkt["mid"]
        ideal = int(round(mid * 20))
        if bkt["count"] == 0:
            bar = " " * ideal + "·"
        else:
            acc = bkt["accuracy"]
            filled = int(round(acc * 20))
            cells = ["#" if i < filled else " " for i in range(21)]
            cells[min(ideal, 20)] = "·" if cells[min(ideal, 20)] == " " else "#"
            bar = "".join(cells)
        n = bkt["count"]
        lines.append(f"{mid:.2f} │ {bar}  n={n}")
    return "\n".join(lines)


def _write_results_json(graph_m: dict, agent_m: dict, live_evals: dict,
                        path: Path = Path("benchmark_results.json")) -> Path:
    """Dump this run to a committed, inspectable file — real output from
    the deterministic 30-scenario harness, not a hand-written claim.
    Reproduce with: QWEN_MOCK=1 python -m prove_or_abstain.benchmark"""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "python -m prove_or_abstain.benchmark",
        "note": "graph/agent are the 30 ground-truth scenarios (QWEN_MOCK=1 "
                "unless DASHSCOPE_API_KEY is set — reproduce with the command "
                "above). live_evals only populates with a real key.",
        "graph": graph_m,
        "agent": agent_m,
        "live_evals": live_evals,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


if __name__ == "__main__":
    # Keep the main 30-scenario table cheap/fast/mock by default (that's
    # the point of QWEN_MOCK=1 reproducibility) -- but only when no key was
    # given. Forcing it unconditionally would make _has_key() always False
    # below, so compare_llm_raw()/cross_model_eval() could never run even
    # with a real DASHSCOPE_API_KEY on the command line.
    if not os.environ.get("DASHSCOPE_API_KEY"):
        os.environ.setdefault("QWEN_MOCK", "1")
    os.environ.setdefault("PROBATIO_DB", ":memory:")
    graph_m = run_benchmark("graph")
    agent_m = run_benchmark("agent")
    print("\nagent vs graph accuracy:",
          f"{agent_m['accuracy']:.1%} / {graph_m['accuracy']:.1%}")
    if _has_key():
        live_evals = {"raw_llm_hallucinations": compare_llm_raw(),
                     "cross_model": cross_model_eval()}
        print("\nraw-LLM hallucinations:", live_evals["raw_llm_hallucinations"])
        print("\ncross-model:", live_evals["cross_model"])
    else:
        live_evals = {"skipped": "needs DASHSCOPE_API_KEY (and QWEN_MOCK unset)"}
        print("\n(live evals skipped — set DASHSCOPE_API_KEY to run "
              "compare_llm_raw + cross_model_eval)")
    out_path = _write_results_json(graph_m, agent_m, live_evals)
    print(f"\nWrote {out_path} — commit it if this run should stand as recorded evidence.")
