"""
bench/run.py — run the benchmark and write results + metrics to disk.

    python -m bench.run

Always runs our agent (offline, deterministic). Also runs the naive Qwen
baseline IF a DashScope key is configured (DASHSCOPE_API_KEY set and QWEN_MOCK
unset) — otherwise it is skipped and its column is left for a later run:

    DASHSCOPE_API_KEY=sk-... QWEN_MOCK= python -m bench.run

Writes bench/results.json (raw verdicts) and bench/metrics.json (the four
numbers per agent), and prints a summary table. Reproducible: same seed in,
same numbers out.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from scenarios import generate_suite
from bench.agents import our_agent, NaiveAgent
from bench.metrics import compute_metrics

HERE = Path(__file__).parent


def _naive_client():
    """A live Qwen client, or None when no key / mock mode."""
    if os.environ.get("QWEN_MOCK") == "1" or not os.environ.get("DASHSCOPE_API_KEY"):
        return None
    from llm import QwenClient
    return QwenClient(mock=False)


def run(seed: int = 0) -> dict:
    scenarios = generate_suite(seed)
    agents = {"prove_or_abstain": our_agent}
    client = _naive_client()
    if client is not None:
        agents["naive_qwen"] = NaiveAgent(client)

    results = {name: [] for name in agents}
    metrics = {}
    for name, agent in agents.items():
        pairs = [(sc, agent(sc)) for sc in scenarios]
        results[name] = [
            {"scenario": sc.name, "label": sc.label, "cause": sc.cause,
             "verdict": v} for sc, v in pairs]
        metrics[name] = compute_metrics(pairs)

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _fmt(x):
    return "—" if x is None else f"{x:.0%}"


def main():
    metrics = run()
    naive = "naive_qwen" in metrics
    print(f"\nBenchmark over {metrics['prove_or_abstain']['n_scenarios']} scenarios "
          f"({metrics['prove_or_abstain']['n_no_cause']} with no single cause)\n")
    rows = [
        ("cause-invention rate (lower better)", "cause_invention_rate"),
        ("precision when it claims a cause", "precision"),
        ("recall on real causes", "recall"),
        ("abstention-reason accuracy", "abstention_reason_accuracy"),
    ]
    head = f"{'metric':38s} {'prove-or-abstain':>18s}"
    if naive:
        head += f" {'naive Qwen':>12s}"
    print(head)
    for label, key in rows:
        line = f"{label:38s} {_fmt(metrics['prove_or_abstain'][key]):>18s}"
        if naive:
            line += f" {_fmt(metrics['naive_qwen'][key]):>12s}"
        print(line)
    if not naive:
        print("\n(naive Qwen skipped — set DASHSCOPE_API_KEY and unset QWEN_MOCK "
              "to fill its column)")


if __name__ == "__main__":
    main()
