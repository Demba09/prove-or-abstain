#!/usr/bin/env bash
# deploy/aliyun.sh — build a linux/amd64 image and push it to Alibaba Cloud
# Container Registry (ACR), ready to run on Function Compute (custom-container).
#
# No secrets live here. Configure via environment variables, then run:
#
#   REGISTRY=registry.cn-hangzhou.aliyuncs.com \
#   NAMESPACE=my-namespace \
#   IMAGE=prove-or-abstain \
#   TAG=v1 \
#   ./deploy/aliyun.sh
#
# Prerequisites: docker with buildx, and `docker login <REGISTRY>` already done
# (see docs/deploy.md). Pass DRY_RUN=1 to print the plan without building.
set -euo pipefail

REGISTRY="${REGISTRY:?set REGISTRY, e.g. registry.cn-hangzhou.aliyuncs.com}"
NAMESPACE="${NAMESPACE:?set NAMESPACE (your ACR namespace)}"
IMAGE="${IMAGE:-prove-or-abstain}"
TAG="${TAG:-v1}"
PLATFORM="${PLATFORM:-linux/amd64}"   # Function Compute runs amd64

REF="${REGISTRY}/${NAMESPACE}/${IMAGE}:${TAG}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

echo "Building ${REF}"
echo "  platform : ${PLATFORM}"
echo "  context  : ${HERE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — not building. Command would be:"
  echo "  docker buildx build --platform ${PLATFORM} -t ${REF} --push ${HERE}"
  exit 0
fi

docker buildx build --platform "${PLATFORM}" -t "${REF}" --push "${HERE}"

echo
echo "Pushed ${REF}"
echo "Next: create/redeploy a Function Compute custom-container function with"
echo "  image=${REF}, port=8000, HTTP trigger, health check path=/health,"
echo "  and runtime env DASHSCOPE_API_KEY (and optional ACTION_WEBHOOK_URL)."
echo "See docs/deploy.md for the click-by-click."
