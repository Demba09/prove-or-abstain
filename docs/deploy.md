# Deploying prove-or-abstain to Alibaba Cloud

The service is a single container (FastAPI on port 8000, `/health` probe,
secrets injected at runtime). This runs it on **Function Compute** with a
custom-container image pushed to **Container Registry (ACR)**.

## Prerequisites

- An Alibaba Cloud account with Container Registry and Function Compute enabled.
- Docker with `buildx` locally.
- A DashScope API key (`DASHSCOPE_API_KEY`) — without it the service still runs
  in mock mode, so you can deploy first and add the key later.

## 1. Create an ACR namespace and repository

In the ACR console, create a namespace (e.g. `my-namespace`) and note your
registry host for the region, e.g. `registry.cn-hangzhou.aliyuncs.com`.

## 2. Log in to the registry

```bash
docker login registry.cn-hangzhou.aliyuncs.com
# username: your Aliyun account; password: the ACR access credential
```

## 3. Build and push the image (amd64)

Function Compute runs `linux/amd64`, so build for that platform explicitly.
Use the helper script:

```bash
REGISTRY=registry.cn-hangzhou.aliyuncs.com \
NAMESPACE=my-namespace \
IMAGE=prove-or-abstain \
TAG=v1 \
./deploy/aliyun.sh
```

`DRY_RUN=1 ./deploy/aliyun.sh` prints the plan without building. The script
pushes `registry.<region>.aliyuncs.com/<namespace>/prove-or-abstain:v1`.

## 4. Create the Function Compute function

Create a function with a **custom container** runtime:

- **Image**: the pushed reference from step 3.
- **Port**: `8000` (the container listens there; `PORT` is overridable if your
  FC configuration injects a different `CAPort`).
- **Trigger**: HTTP.
- **Health check**: path `/health` (returns `{"status":"ok"}`).
- **Environment variables** (runtime only, never baked into the image):
  - `DASHSCOPE_API_KEY` — enables real Qwen; omit for mock mode.
  - `ACTION_WEBHOOK_URL` — optional; where autopilot `EXECUTE` actions are POSTed.
  - `QWEN_MODEL` / `QWEN_BASE_URL` — optional overrides.

## 5. Verify

Once deployed, hit the function's HTTP endpoint:

```bash
curl https://<your-fc-endpoint>/health
curl -X POST https://<your-fc-endpoint>/investigate \
  -H 'content-type: application/json' -d '{"panel":"clean"}'
```

The root path (`/`) serves the interactive demo page.

## Notes

- The image runs as an unprivileged user and writes nothing to disk at runtime.
- `.env` and tests are excluded from the image (`.dockerignore`); configuration
  is entirely through runtime environment variables.
