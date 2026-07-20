"""
monitor.py — continuous autonomous surveillance (the Track-4 autopilot).

A MetricMonitor watches a set of data sources. Each cycle it fetches the
current panel per source and hands it to ingest.ingest_and_investigate(),
which persists it (prove_or_abstain/memory.py::observations) and compares it
to the pooled reference window of every PRIOR observation for that source —
durable, surviving a restart, unlike the old in-process
`self._baselines` snapshot dict this module used to keep. When a metric has
moved materially, the deterministic investigation runs and, on a confident
ASSERT, this module — not ingest_and_investigate — fires the webhook and
records the alert (see the autopilot=False note on _investigate below for
why). One bad source never kills the loop.

Sources are dicts:
    {"type": "sql"|"sheets"|"csv"|"inline",
     "config": {...},                     # per-type (see _fetch_current)
     "metrics": ["conversion"], "dims": ["device", "segment"],
     "metric_kinds": {"revenue": "sum"},  # optional
     "source_id": "my-source"}            # optional, defaults to "monitor-{i}"

CLI: python -m prove_or_abstain.monitor   (one demo cycle on a built-in panel)
"""
from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd

from prove_or_abstain import ingest, memory, webhook
from prove_or_abstain.autopilot import record_check

logger = logging.getLogger(__name__)

EXECUTE_CONFIDENCE = 0.70


class MetricMonitor:
    def __init__(self, sources: list[dict], check_interval_s: int = 900):
        self.sources = sources
        self.check_interval_s = check_interval_s

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

    def _investigate(self, source_id: str, source: dict, current: pd.DataFrame) -> dict:
        """ingest_and_investigate(..., autopilot=False): the graph's own
        actuator can then never emit an EXECUTE action (that requires
        autopilot_enabled=True), so _run_investigation never fires its own
        record_execution()/webhook.notify() on this call — avoiding a
        double-fire with the alert this method fires itself below, on the
        exact same condition (verdict==ASSERT, confidence>=0.70) the
        pre-refactor version of this module already used. Cold start
        (ingest_and_investigate's own first-sighting seed) and NO_ANOMALY
        both skip recording an investigation row, matching prior behaviour
        (nothing was persisted for those cases before this refactor either)."""
        out = ingest.ingest_and_investigate(
            source_id, current, metrics=source["metrics"], dims=source["dims"],
            metric_kinds=source.get("metric_kinds", {}), autopilot=False,
            mode="agent")

        if out.get("cold_start"):
            return {"verdict": "BASELINE_SET"}

        verdict = out.get("verdict", "NO_ANOMALY")
        if verdict == "NO_ANOMALY":
            return {"verdict": "NO_ANOMALY"}

        conf = out.get("confidence", 0.0)
        rc = out.get("root_cause")
        cause = f"{rc['dimension']}={rc['segment']}" if rc else None
        llm = out.get("llm") or {}
        inv_id = memory.record_investigation(
            source["metrics"][0], source["dims"], verdict, conf, cause,
            out.get("report", ""), out.get("trace", []),
            mode="agent", model=llm.get("model", ""))

        if verdict == "ASSERT" and conf >= EXECUTE_CONFIDENCE:
            detail = f"Autopilot: scoped action on {cause} (conf {conf:.2f})."
            memory.create_alert(inv_id, source["metrics"][0], cause, "EXECUTE", detail)
            webhook.notify(source["metrics"][0], verdict, conf, cause, "EXECUTE", detail)
        return {"verdict": verdict, "confidence": conf, "cause": cause,
                "investigation_id": inv_id}

    async def check_once(self) -> list[dict]:
        """One surveillance cycle over all sources. Errors are per-source and
        never propagate — a broken feed is logged and skipped."""
        results = []
        for i, source in enumerate(self.sources):
            source_id = source.get("source_id", f"monitor-{i}")
            try:
                current = self._fetch_current(source)
            except Exception as exc:  # a bad fetch must not kill the loop
                results.append({"source": i, "error": str(exc)})
                continue
            try:
                out = self._investigate(source_id, source, current)
                out["source"] = i
                results.append(out)
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
                logger.error("cycle error: %s", exc, exc_info=True)
            await asyncio.sleep(self.check_interval_s)


if __name__ == "__main__":
    # Standalone entry point (not via api/app.py) — configure logging here
    # too, or run_forever()'s logger.error() calls go nowhere.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("QWEN_MOCK", "1")
    os.environ.setdefault("PROBATIO_DB", ":memory:")
    from prove_or_abstain.panels import BASELINE, CLEAN

    source = {
        "type": "inline",
        "config": {"current": BASELINE},   # 1st cycle: nothing to compare yet
        "metrics": ["conversion", "activation"],
        "dims": ["device", "segment"],
        "source_id": "demo",
    }
    mon = MetricMonitor(sources=[source])
    print("cycle 1 (cold start — persists BASELINE, no verdict yet):")
    print(asyncio.run(mon.check_once()))

    source["config"]["current"] = CLEAN    # 2nd cycle: something has changed
    print("cycle 2 (compares CLEAN against the persisted reference window):")
    print(asyncio.run(mon.check_once()))

    print("history:", len(memory.get_history()), "alerts:", len(memory.get_active_alerts()))
