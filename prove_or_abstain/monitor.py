"""
monitor.py — continuous autonomous surveillance (the Track-4 autopilot).

A MetricMonitor watches a set of data sources. Each cycle it fetches the
current panel per source, compares it to the last successful fetch (the rolling
baseline), and when a metric has moved materially it runs the deterministic
investigation, persists the verdict to memory.py, and — on a confident ASSERT —
fires the webhook. One bad source never kills the loop.

Sources are dicts:
    {"type": "sql"|"sheets"|"csv"|"inline",
     "config": {...},                     # per-type (see _fetch_current)
     "metrics": ["conversion"], "dims": ["device", "segment"],
     "metric_kinds": {"revenue": "sum"}}  # optional

CLI: python -m prove_or_abstain.monitor   (one demo cycle on a built-in panel)
"""
from __future__ import annotations

import asyncio
import os

import pandas as pd

from prove_or_abstain import memory, webhook
from prove_or_abstain.agent_loop import investigate_agentic
from prove_or_abstain.autopilot import record_check
from prove_or_abstain.gates import MATERIAL_REL
from prove_or_abstain.panels import metric_totals

EXECUTE_CONFIDENCE = 0.70


def _material_move(baseline: pd.DataFrame, current: pd.DataFrame,
                   metrics: list[str], kinds: dict) -> bool:
    """Cheap pre-check (same rule as the detector) before spending an
    investigation: any metric whose aggregate moved >= 2%."""
    for m in metrics:
        n0, c0 = metric_totals(baseline, m)
        n1, c1 = metric_totals(current, m)
        if kinds.get(m) == "sum":
            r0, r1 = c0, c1
        else:
            r0 = c0 / n0 if n0 else 0.0
            r1 = c1 / n1 if n1 else 0.0
        if abs(r1 - r0) / (abs(r0) + 1e-12) >= MATERIAL_REL:
            return True
    return False


class MetricMonitor:
    def __init__(self, sources: list[dict], check_interval_s: int = 900):
        self.sources = sources
        self.check_interval_s = check_interval_s
        self._baselines: dict[int, pd.DataFrame] = {}
        # Seed baselines from any source that supplied one inline.
        for i, s in enumerate(sources):
            seed = s.get("config", {}).get("baseline")
            if isinstance(seed, pd.DataFrame):
                self._baselines[i] = seed

    def _fetch_current(self, source: dict) -> pd.DataFrame:
        kind, cfg = source["type"], source.get("config", {})
        if kind == "inline":
            return cfg["current"]
        if kind == "csv":
            return pd.read_csv(cfg["path"])
        if kind == "sql":
            from prove_or_abstain.connectors.sql import fetch_panel
            return fetch_panel(cfg["dsn"], cfg["query"])
        if kind == "sheets":
            from prove_or_abstain.connectors.gsheets import fetch_panel
            return fetch_panel(cfg["url"])
        raise ValueError(f"unknown source type '{kind}'")

    def _investigate(self, source: dict, baseline, current) -> dict:
        metrics = source["metrics"]
        kinds = source.get("metric_kinds", {})
        if not _material_move(baseline, current, metrics, kinds):
            return {"verdict": "NO_ANOMALY"}

        state = {"baseline": baseline, "current": current, "metrics": metrics,
                 "metric_kinds": kinds, "dims": source["dims"],
                 "autopilot_enabled": True, "trace": []}
        final = investigate_agentic(state)

        verdict = final.get("verdict", "NO_ANOMALY")
        conf = final.get("confidence", 0.0)
        win = final.get("winning_report")
        cause = f"{final.get('winning_dim')}={win.leading_segment}" if win else None
        llm = final.get("llm", {})
        inv_id = memory.record_investigation(
            metrics[0], source["dims"], verdict, conf, cause,
            final.get("report", ""), final.get("trace", []),
            mode="agent", model=llm.get("model", ""))

        if verdict == "ASSERT" and conf >= EXECUTE_CONFIDENCE:
            action = final["actions"][0]
            memory.create_alert(inv_id, metrics[0], cause, action.kind, action.detail)
            webhook.notify(metrics[0], verdict, conf, cause, action.kind, action.detail)
        return {"verdict": verdict, "confidence": conf, "cause": cause,
                "investigation_id": inv_id}

    async def check_once(self) -> list[dict]:
        """One surveillance cycle over all sources. Errors are per-source and
        never propagate — a broken feed is logged and skipped."""
        results = []
        for i, source in enumerate(self.sources):
            try:
                current = self._fetch_current(source)
            except Exception as exc:  # a bad fetch must not kill the loop
                results.append({"source": i, "error": str(exc)})
                continue
            baseline = self._baselines.get(i)
            if baseline is None:                 # first sighting -> seed only
                self._baselines[i] = current
                results.append({"source": i, "verdict": "BASELINE_SET"})
                continue
            try:
                out = self._investigate(source, baseline, current)
                out["source"] = i
                results.append(out)
                self._baselines[i] = current     # roll the baseline forward
            except Exception as exc:
                results.append({"source": i, "error": f"investigate: {exc}"})

        verdicts = [r.get("verdict") for r in results]
        record_check("ASSERT" if "ASSERT" in verdicts else "NO_ANOMALY")
        return results

    async def run_forever(self) -> None:
        while True:
            try:
                await self.check_once()
            except Exception as exc:             # last-resort guard
                print(f"[monitor] cycle error: {exc}")
            await asyncio.sleep(self.check_interval_s)


if __name__ == "__main__":
    os.environ.setdefault("QWEN_MOCK", "1")
    os.environ.setdefault("PROBATIO_DB", ":memory:")
    from prove_or_abstain.panels import BASELINE, CLEAN
    mon = MetricMonitor(sources=[{
        "type": "inline",
        "config": {"baseline": BASELINE, "current": CLEAN},
        "metrics": ["conversion", "activation"],
        "dims": ["device", "segment"],
    }])
    print(asyncio.run(mon.check_once()))
    print("history:", len(memory.get_history()), "alerts:", len(memory.get_active_alerts()))
