"""
check_qwen.py — vérifie que ta clé DASHSCOPE et l'endpoint Qwen répondent.

Lance :  python3.12 check_qwen.py
Prérequis : pip install openai python-dotenv  +  DASHSCOPE_API_KEY configurée.
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
    raise SystemExit("DASHSCOPE_API_KEY absente. Fais : export DASHSCOPE_API_KEY=... "
                     "ou mets-la dans .env (python-dotenv).")

client = QwenClient(mock=False)
print(f"modèle   : {client.model}")
print(f"base_url : {client.base_url}")
print("appel Qwen en cours...")
reply = client.complete(
    system="Réponds en un mot.",
    user="Dis 'pret' si tu reçois ce message.",
    max_tokens=10,
)
print(f"réponse Qwen : {reply!r}")
print("\nOK — clé et endpoint fonctionnels. Gate Qwen franchissable.")
