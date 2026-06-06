#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DOCKERHUB_USER:-}" ]]; then
  echo "DOCKERHUB_USER is required" >&2
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-protonet-sr}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/arm64}"
EXTRA_TAG="${EXTRA_TAG:-rpi5}"

FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}"

docker buildx inspect protonet-builder >/dev/null 2>&1 || docker buildx create --use --name protonet-builder

docker buildx build \
  --platform "${PLATFORMS}" \
  -t "${FULL_IMAGE}:${IMAGE_TAG}" \
  -t "${FULL_IMAGE}:${EXTRA_TAG}" \
  --push \
  .

docker buildx imagetools inspect "${FULL_IMAGE}:${IMAGE_TAG}"
