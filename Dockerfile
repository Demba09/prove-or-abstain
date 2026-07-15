# prove-or-abstain — demo image (FastAPI + LangGraph, in-memory panels).
FROM python:3.12-slim

WORKDIR /app

# requirements first: the pip layer is only invalidated when deps change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code next (.dockerignore excludes .env, .venv, tests, data, caches).
COPY . .

# Secrets are injected at runtime only:
#   docker run -e DASHSCOPE_API_KEY=...   (or the Alibaba Cloud env panel)
# Without a key, prove_or_abstain/llm.py falls back to mock mode: the image also runs offline.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# PORT is overridable (e.g. Function Compute's CAPort); defaults to 8000.
# exec: uvicorn becomes PID 1 and properly receives SIGTERM on docker stop.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
