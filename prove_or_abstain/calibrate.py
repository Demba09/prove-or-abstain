"""
calibrate.py — is the confidence score calibrated?

A confidence of 0.7 should mean "right about 70% of the time". calibrate_confidence
buckets the benchmark's ASSERT predictions by confidence, measures the actual
accuracy per bucket, and reports the Expected Calibration Error (ECE) — the
sample-weighted gap between confidence and accuracy. plot_calibration_curve
draws it in ASCII (no matplotlib dependency).

Feed it the `records` from prove_or_abstain.benchmark.run_benchmark().
"""
from __future__ import annotations


def _positive(rec: dict) -> bool:
    """Calibrate the confidence score where it is meaningful — i.e. on the
    pipeline's ASSERT predictions (ABSTAIN carries confidence 0 by construction)."""
    got = rec.get("got")
    return (got == "ASSERT") if got is not None else rec.get("confidence", 0.0) > 0.0


def calibrate_confidence(benchmark_results: list[dict], n_buckets: int = 10) -> dict:
    preds = [r for r in benchmark_results if _positive(r)]
    buckets = []
    ece = 0.0
    total = len(preds)
    for b in range(n_buckets):
        lo, hi = b / n_buckets, (b + 1) / n_buckets
        # last bucket is closed on the right so confidence == 1.0 lands in it
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


if __name__ == "__main__":
    import os
    os.environ.setdefault("QWEN_MOCK", "1")
    os.environ.setdefault("PROBATIO_DB", ":memory:")
    from prove_or_abstain.benchmark import run_benchmark
    recs = run_benchmark("agent", verbose=False)["records"]
    print(plot_calibration_curve(recs))
    print("\n", calibrate_confidence(recs))
