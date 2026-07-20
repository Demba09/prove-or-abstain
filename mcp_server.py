"""mcp_server.py — Model Context Protocol server for prove-or-abstain.

Exposes the investigation agent as MCP tools that Qwen Cloud agents can call
directly. This makes Qwen the primary orchestrator — it decides WHEN to
investigate, interprets the results, and generates human-readable responses.

Run:  python mcp_server.py           (stdio transport, for Qwen Cloud MCP)
 or:  python mcp_server.py --port 8080  (SSE transport)

The MCP server wraps the same LangGraph pipeline used by the REST API, so the
verdict is always deterministic — Qwen only orchestrates, never computes.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import asdict

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise SystemExit(
        "mcp package not found. Install: pip install mcp\n"
        "See: https://modelcontextprotocol.io"
    )

# --- import the prove-or-abstain pipeline (same as the REST API) ---
sys.path.insert(0, os.path.dirname(__file__))
from api.app import _METRICS, _PANELS                               # panel data lives in api/app
from prove_or_abstain.investigate import _run_investigation, check_all_panels
from prove_or_abstain.autopilot import get_dashboard as autopilot_dashboard, resolve_execution, record_check
from prove_or_abstain.panels import BASELINE

mcp = FastMCP(
    name="prove-or-abstain",
    instructions="Causal investigation agent for product metrics. "
                 "Investigates metric changes and returns ASSERT (cause proven) "
                 "or ABSTAIN (cannot localize — escalates to human). "
                 "Use investigate_scenario to check built-in demo scenarios, "
                 "investigate_sql to query live databases, "
                 "autonomous_check to run all panels at once with autopilot ON, "
                 "and get_dashboard to see current status and active alerts.",
)

# -- built-in scenario descriptions (for the LLM to understand what's available) --
_SCENARIOS_INFO = {
    "clean":      "one segment's rate collapses while others are stable — localizes",
    "diffuse":    "every segment drops equally — does NOT localize (systemic)",
    "mixshift":   "both population mix AND rates shift — mechanism entangled, ABSTAIN",
    "deep":       "a single cell (paid×mobile) collapses — ASSERTs, then drills down",
}


@mcp.tool(name="investigate_scenario")
def investigate_scenario(scenario: str = "clean") -> str:
    """Run the causal investigation on one of the built-in demo scenarios.

    Args:
        scenario: One of 'clean', 'diffuse', 'mixshift', or 'deep'.

    Returns a JSON string with verdict, confidence, root cause, gates,
    drill-down, action, report, speculations, and full trace.
    """
    if scenario not in _PANELS:
        return json.dumps({
            "error": f"unknown scenario: '{scenario}'. Available: {sorted(_PANELS.keys())}"
        })
    result = _run_investigation(
        BASELINE, _PANELS[scenario],
        metrics=_METRICS,
        dims=["device", "segment"],
        autopilot=False,
    )
    result["panel"] = scenario
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(name="investigate_sql")
def investigate_sql(dsn: str, baseline_query: str, current_query: str) -> str:
    """Run the investigation from a live SQL database query.

    Args:
        dsn: SQLAlchemy connection string ('postgresql://user:pass@host/db', 'mysql://...', 'sqlite:///file.db')
        baseline_query: Single SELECT returning [metric, dims..., n, c] for the baseline period
        current_query: Same shape for the current period

    Returns verdict + decomposition gates + report as JSON.
    """
    from prove_or_abstain.connectors.sql import fetch_panel as sql_panel, SqlQueryError
    try:
        base = sql_panel(dsn, baseline_query)
        curr = sql_panel(dsn, current_query)
    except SqlQueryError as e:
        return json.dumps({"error": f"SQL error: {e}"})

    from api.app import _RESERVED
    dims = [c for c in base.columns if c not in _RESERVED]
    metrics = sorted(base["metric"].unique())
    result = _run_investigation(base, curr, metrics=metrics, dims=dims,
                                autopilot=False, metric_kinds={})
    result["panel"] = "sql"
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(name="autonomous_check")
def autonomous_check() -> str:
    """Run the autonomous monitor: check ALL four built-in panels with
    autopilot ON. If any panel ASSERTs with high confidence, an execution
    record is created (visible via get_dashboard).

    Returns summary of all 4 investigations as JSON."""
    summary = check_all_panels(_PANELS, BASELINE, _METRICS, ["device", "segment"])
    record_check(summary["verdict"])
    return json.dumps(summary, ensure_ascii=False, default=str)


@mcp.tool(name="get_dashboard")
def get_dashboard() -> str:
    """Get the current autopilot dashboard status — active alerts,
    total checks, total executions, uptime."""
    return json.dumps(asdict(autopilot_dashboard()), ensure_ascii=False, default=str)


@mcp.tool(name="resolve_alert")
def resolve_alert(alert_id: str) -> str:
    """Mark an active autopilot alert as resolved (human-in-the-loop).

    Args:
        alert_id: The alert ID (e.g., 'conversion:segment=paid')
    """
    entry = resolve_execution(alert_id)
    if entry is None:
        return json.dumps({"error": f"alert '{alert_id}' not found or already resolved"})
    return json.dumps({"resolved": alert_id})


@mcp.tool(name="describe_panels")
def describe_panels() -> str:
    """List available built-in scenarios with descriptions — useful so the
    Qwen agent knows which scenario matches the user's question."""
    return json.dumps(_SCENARIOS_INFO, ensure_ascii=False)


@mcp.tool(name="describe_gates")
def describe_gates() -> str:
    """Explain the 4 verification gates — what the agent requires before
    it will ASSERT a cause instead of ABSTAINing."""
    return json.dumps({
        "gates": {
            "material":    "|ΔR|/R₀ ≥ 2% — the move is large enough to matter",
            "localized":   "top contribution share ≥ 0.55 — one segment dominates",
            "significant": "two-proportion z-test, p ≤ 0.01 — not sampling noise",
            "clean":       "interaction share ≤ 0.50 — rate and mix effects separable",
        },
        "verdicts": {
            "ASSERT":  "all 4 gates pass → cause proven → recommend or execute action",
            "ABSTAIN": "any gate fails → cannot localize → escalate to human",
        }
    }, ensure_ascii=False)


# --- runner ---
if __name__ == "__main__":
    os.environ.setdefault("QWEN_MOCK", "1")

    parser = argparse.ArgumentParser(description="prove-or-abstain MCP server")
    parser.add_argument("--port", type=int, default=None,
                        help="Run with SSE transport on this port (default: stdio for Qwen Cloud)")
    args = parser.parse_args()

    if args.port:
        print(f"MCP server starting on http://0.0.0.0:{args.port}/sse")
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
