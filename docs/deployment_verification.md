# Deployment verification

Honest status, not a claim: this file exists because "the README explains
how to deploy" and "deployment is proven" are two different things. Here's
exactly which one is true today, and what closes the gap.

## What's verified, right now, with real output

The FastAPI app itself — the thing that would run inside the container —
was started and exercised directly (`uvicorn api.app:app`), not simulated:

```
$ curl -s http://127.0.0.1:8030/health
{"status":"ok"}

$ curl -s -X POST http://127.0.0.1:8030/investigate -d '{"panel":"clean"}' -H 'content-type: application/json'
verdict: ASSERT   root_cause: {"dimension": "segment", "segment": "paid"}

$ curl -s -X POST http://127.0.0.1:8030/investigate -d '{"panel":"diffuse"}' -H 'content-type: application/json'
verdict: ABSTAIN
```

Both verdict paths work, `/health` responds — the same checks the
Dockerfile's `HEALTHCHECK` and Function Compute's readiness probe would run.

## What's NOT verified from this environment, and why

Two steps could not be executed here, for concrete, checkable reasons —
not skipped, blocked:

1. **Building/running the actual container.** `docker build` fails in this
   sandbox: `dockerd` cannot start (`ulimit: error setting limit: Operation
   not permitted`) — the daemon itself isn't available, by sandbox design.
   This also means the Dockerfile's non-root `USER appuser` step (added in
   response to a code review flagging the image running as root) is
   reviewed but not build-verified — step 1 below covers it too; watch for
   permission errors if anything under `/app` needs to be writable beyond
   what the `chown -R` already covers.
2. **Pushing to Alibaba Cloud Container Registry / deploying to Function
   Compute.** No `aliyun` CLI and no Alibaba Cloud credentials exist in this
   environment. This is deliberate on my end too: actually deploying would
   create real, billable cloud resources on somebody's account — that needs
   the account owner's explicit action, not an agent doing it unasked.

So: the code that would run in the container is proven to work; the
container build and the cloud deployment itself are not, and can't be from
here.

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
