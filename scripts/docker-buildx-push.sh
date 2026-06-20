#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DOCKERHUB_IMAGE="${1:-${DOCKERHUB_IMAGE:-}}"
IMAGE_TAG="${2:-${IMAGE_TAG:-rpi5}}"

if [[ -z "${DOCKERHUB_IMAGE}" && -n "${DOCKERHUB_USER:-}" ]]; then
  DOCKERHUB_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME:-protonet-sr-app}"
fi

if [[ -z "${DOCKERHUB_IMAGE}" ]]; then
  echo "Usage: $0 DOCKERHUB_USER/IMAGE [TAG]" >&2
  exit 2
fi

PLATFORM="${PLATFORMS:-${PLATFORM:-linux/arm64}}" \
  exec "${REPO_ROOT}/speaker_app/scripts/docker-buildx-push.sh" \
  "${DOCKERHUB_IMAGE}" "${IMAGE_TAG}"
