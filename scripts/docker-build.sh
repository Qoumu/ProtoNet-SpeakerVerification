#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-local}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE="${REPO_ROOT}/speaker_app/Dockerfile"

IMAGE_NAME="${IMAGE_NAME:-protonet-sr-app}"
LOCAL_TAG="${LOCAL_TAG:-local}"
RPI5_TAG="${RPI5_TAG:-rpi5}"
BUILDER_NAME="${BUILDER_NAME:-protonet-builder}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/docker-build.sh [local|rpi5]

Modes:
  local   Build a native image for the current machine with docker build
  rpi5    Cross-build a linux/arm64 image for Raspberry Pi 5 with docker buildx

Optional environment variables:
  IMAGE_NAME     Docker image name (default: protonet-sr-app)
  LOCAL_TAG      Tag used for local builds (default: local)
  RPI5_TAG       Tag used for Raspberry Pi 5 builds (default: rpi5)
  BUILDER_NAME   buildx builder name for cross-builds (default: protonet-builder)

Examples:
  ./scripts/docker-build.sh local
  ./scripts/docker-build.sh rpi5
  IMAGE_NAME=myrepo/protonet-sr ./scripts/docker-build.sh local
EOF
}

ensure_buildx_builder() {
  docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1 || \
    docker buildx create --use --name "${BUILDER_NAME}" >/dev/null
}

require_model() {
  if [[ ! -f "${REPO_ROOT}/output/ecapa_tdnn_protonet_model.pth" ]]; then
    echo "Missing model: ${REPO_ROOT}/output/ecapa_tdnn_protonet_model.pth" >&2
    exit 2
  fi
}

build_local() {
  require_model
  echo "Building native image ${IMAGE_NAME}:${LOCAL_TAG}"
  docker build \
    --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}" \
    --build-arg "INSTALL_PULSE=true" \
    --file "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:${LOCAL_TAG}" \
    "${REPO_ROOT}"
}

build_rpi5() {
  require_model
  ensure_buildx_builder
  echo "Building Raspberry Pi 5 image ${IMAGE_NAME}:${RPI5_TAG} for linux/arm64"
  docker buildx build \
    --platform linux/arm64 \
    --file "${DOCKERFILE}" \
    -t "${IMAGE_NAME}:${RPI5_TAG}" \
    --load \
    "${REPO_ROOT}"
}

case "${MODE}" in
  local)
    build_local
    ;;
  rpi5|cross|arm64)
    build_rpi5
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo >&2
    usage >&2
    exit 1
    ;;
esac
