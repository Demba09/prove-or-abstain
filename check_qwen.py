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
print("\nOK — key and endpoint working.")
