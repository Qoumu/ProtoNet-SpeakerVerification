#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-local}"

IMAGE_NAME="${IMAGE_NAME:-protonet-sr}"
LOCAL_TAG="${LOCAL_TAG:-local}"
RPI5_TAG="${RPI5_TAG:-rpi5}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
BUILDER_NAME="${BUILDER_NAME:-protonet-builder}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/docker-build.sh [local|rpi5]

Modes:
  local   Build a native image for the current machine with docker build
  rpi5    Cross-build a linux/arm64 image for Raspberry Pi 5 with docker buildx

Optional environment variables:
  IMAGE_NAME     Docker image name (default: protonet-sr)
  LOCAL_TAG      Tag used for local builds (default: local)
  RPI5_TAG       Tag used for Raspberry Pi 5 builds (default: rpi5)
  BUILD_CONTEXT  Docker build context (default: .)
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

build_local() {
  echo "Building native image ${IMAGE_NAME}:${LOCAL_TAG}"
  docker build \
    -t "${IMAGE_NAME}:${LOCAL_TAG}" \
    "${BUILD_CONTEXT}"
}

build_rpi5() {
  ensure_buildx_builder
  echo "Building Raspberry Pi 5 image ${IMAGE_NAME}:latest for linux/arm64"
  docker buildx build \
    --platform linux/arm64 \
    -t "${IMAGE_NAME}:latest" \
    --load \
    "${BUILD_CONTEXT}"
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
