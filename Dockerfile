# prove-or-abstain — image de démo (FastAPI + LangGraph, panels en mémoire).
FROM python:3.12-slim

WORKDIR /app

# requirements d'abord : la layer pip n'est invalidée que si les deps changent.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le code ensuite (.dockerignore exclut .env, .venv, tests, data, caches).
COPY . .

# Secrets injectés au runtime uniquement :
#   docker run -e DASHSCOPE_API_KEY=...   (ou variables d'env du panel Alibaba)
# Sans clé, llm.py bascule en mode mock : l'image tourne aussi offline.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# PORT surchargeable (ex. CAPort de Function Compute) ; 8000 par défaut.
# exec : uvicorn prend le PID 1 et reçoit bien SIGTERM au docker stop.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
