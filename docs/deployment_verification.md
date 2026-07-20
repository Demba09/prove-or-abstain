# Deployment verification

Honest status, not a claim: this file exists because "the README explains
how to deploy" and "deployment is proven" are two different things. Here's
exactly which one is true today, and what closes the gap.

## What's verified, right now, with real output

The FastAPI app itself — the thing that would run inside the container —
was started and exercised directly (`uvicorn api.app:app`), not simulated:

```
$ curl -s http://127.0.0.1:8040/health
{"status":"ok"}

$ curl -s -X POST http://127.0.0.1:8040/investigate -d '{"panel":"clean"}' -H 'content-type: application/json'
verdict: ASSERT   root_cause: {'dimension': 'segment', 'segment': 'paid'}

$ curl -s -X POST http://127.0.0.1:8040/investigate -d '{"panel":"diffuse"}' -H 'content-type: application/json'
verdict: ABSTAIN
```

Both verdict paths work, `/health` responds — the same checks the
Dockerfile's `HEALTHCHECK` and Function Compute's readiness probe would run.

The `.dockerignore` rules were also verified against `git ls-files` (which
files would actually enter the build context via `COPY . .`): `examples/`
(needed at runtime by `benchmark.py` and `POST /investigate/example`) is
included; `.env`, `tests/`, `scripts/`, caches, and git metadata are
correctly excluded — no secrets, no dev-only files leak into the image.

## What's NOT verified from this environment, and why

Two steps could not be executed here, for concrete, checkable reasons —
not skipped, blocked:

1. **Building the actual container image.** `dockerd` *does* start in this
   sandbox (an earlier version of this note said otherwise — that was
   specific to a prior sandbox instance, not a permanent limitation). The
   build gets further now: it fails pulling the `python:3.12-slim` base
   image from Docker Hub through this environment's outbound proxy
   (`403 Forbidden` on the registry CDN) — a network policy on *this*
   sandbox, not a Dockerfile problem. So the Dockerfile's non-root
   `USER appuser` step and the `HEALTHCHECK` are reviewed but not
   build-verified; step 1 below covers that.
2. **Pushing to Alibaba Cloud Container Registry / deploying to Function
   Compute.** No `aliyun` CLI and no Alibaba Cloud credentials exist in this
   environment. This is deliberate on my end too: actually deploying would
   create real, billable cloud resources on somebody's account — that needs
   the account owner's explicit action, not an agent doing it unasked.

So: the code that would run in the container is proven to work, and the
`.dockerignore` file manifest is proven correct; the actual image build and
the cloud deployment are not, and can't be from here.

## Closing the gap — run this yourself

```bash
# 1. Build and run the actual container
docker build -t prove-or-abstain .
docker run -p 8000:8000 -e DASHSCOPE_API_KEY=sk-... prove-or-abstain
curl localhost:8000/health
curl -X POST localhost:8000/investigate -d '{"panel":"clean"}' -H 'content-type: application/json'

# 2. Push + deploy (see README's Docker section for the full registry/FC commands)
docker buildx build --platform linux/amd64 -t registry.<region>.aliyuncs.com/<ns>/prove-or-abstain:latest --push .
# then create/update the Function Compute function from that image, and hit its public URL
```

Paste me the output of step 1 (or step 2's live URL + a `curl .../health`)
and I'll fold it into this file and the README as the actual proof, dated
and attributed — replacing this document's "not verified" with a real
transcript instead of a claim.
