"""Central configuration: verifier thresholds, loop bound, default metric config.

Thresholds here are placeholders to be hardened on J8-J9 against the two demo
scenarios. They are deliberately in one place so tuning the verifier never means
hunting through node code.
"""

from __future__ import annotations

# --- Verifier gating thresholds (deterministic; the LLM only articulates) ---
# explained_share of the top hypothesis required to ASSERT a cause with high confidence
TAU_HIGH: float = 0.60
# below this, the evidence is too weak -> loop back or abstain
TAU_LOW: float = 0.35
# minimum gap between top-1 and top-2 explained share to call the cause "separated"
MIN_SEPARATION: float = 0.20

# --- Loop control ---
MAX_ITERATIONS: int = 3      # bounded back-tracking; never an unbounded cycle

# --- Autonomy ---
# When False (default), the agent only PROPOSES actions. It auto-executes an action
# only when AUTOPILOT_ENABLED is True AND the verifier confidence is "high".
# An autopilot that can act must be able to refuse to act: abstention => no execution.
AUTOPILOT_ENABLED: bool = False

# --- Anomaly detection ---
# relative deviation vs baseline that flags an anomaly (e.g. 0.25 = +/-25%)
ANOMALY_REL_THRESHOLD: float = 0.25

# --- Default metric config for the SaaS churn demo ---
DEFAULT_METRIC_CONFIG: dict = {
    "name": "churn_30d",
    "kind": "rate",
    "grain": "week",
    "baseline_window": 1,
    "dimensions": ["plan", "channel", "region", "cohort"],
    "churn_col": "churned",
    "period_col": "period",
}

# --- Qwen / DashScope ---
DASHSCOPE_BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL: str = "qwen-plus"   # qwen-max for the final run; plus is cheaper while building
