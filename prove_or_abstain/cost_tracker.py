"""
cost_tracker.py — token counting and cost estimation for Qwen calls.

Pricing is DashScope international, per 1M tokens (June 2026). A CostTracker
accumulates usage from every real API round-trip; QwenClient owns one and
updates it whenever a response carries a `usage` block. In mock mode nothing is
added, so the cost is exactly 0.
"""
from __future__ import annotations

# (input $/1M, output $/1M)
PRICING = {
    "qwen-plus":  (0.80, 2.40),
    "qwen-max":   (1.40, 5.60),
    "qwen-turbo": (0.40, 1.20),
}
_DEFAULT = PRICING["qwen-plus"]


class CostTracker:
    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.input_tokens += int(prompt_tokens or 0)
        self.output_tokens += int(completion_tokens or 0)
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        p_in, p_out = PRICING.get(self.model, _DEFAULT)
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
        }


if __name__ == "__main__":
    t = CostTracker("qwen-plus")
    t.add_usage(1000, 300)
    t.add_usage(500, 120)
    print(t.to_dict())
