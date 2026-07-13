"""
check_qwen.py — verifies that the DASHSCOPE key and the Qwen endpoint respond.

Run:  python check_qwen.py
Prerequisites: pip install openai python-dotenv  +  DASHSCOPE_API_KEY set.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

from llm import QwenClient

key = os.environ.get("DASHSCOPE_API_KEY")
if not key:
    raise SystemExit("DASHSCOPE_API_KEY is missing. Run: export DASHSCOPE_API_KEY=... "
                     "or put it in .env (python-dotenv).")

client = QwenClient(mock=False)
print(f"model    : {client.model}")
print(f"base_url : {client.base_url}")
print("calling Qwen...")
reply = client.complete(
    system="Answer in one word.",
    user="Say 'ready' if you receive this message.",
    max_tokens=10,
)
print(f"Qwen reply: {reply!r}")

print("\ntesting the function-calling path (rank_dimensions)...")
import time
dims = ["device", "segment", "region"]
t0 = time.perf_counter()
order = client.plan_dimensions("conversion", -0.15, dims)
print(f"ranked in  : {(time.perf_counter() - t0) * 1000:.0f} ms")
print(f"order      : {order}")
assert sorted(order) == sorted(dims), "function call must return a permutation"

print("\ntesting the natural-language entry point (parse_question)...")
spec = client.parse_question("why did our sales drop last weekend?",
                             ["conversion", "activation"], dims)
print(f"parsed     : {spec}")
assert spec["metric"] in ("conversion", "activation")
print("\nOK — key, endpoint, function-calling and NL parsing all working.")
