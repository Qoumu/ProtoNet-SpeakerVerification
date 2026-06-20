#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${APP_DIR}/.." && pwd)"

DOCKERHUB_IMAGE="${DOCKERHUB_IMAGE:-${1:-}}"
IMAGE_TAG="${IMAGE_TAG:-${2:-rpi5}}"
PLATFORM="${PLATFORM:-linux/arm64}"
BUILDER_NAME="${BUILDER_NAME:-protonet-speaker-builder}"
PUBLISH_LATEST="${PUBLISH_LATEST:-true}"
PUSH_RETRIES="${PUSH_RETRIES:-3}"

if [[ -z "${DOCKERHUB_IMAGE}" ]]; then
    cat >&2 <<'EOF'
Usage:
  ./scripts/docker-buildx-push.sh DOCKERHUB_USER/IMAGE [TAG]

Example:
  docker login
  ./scripts/docker-buildx-push.sh myuser/protonet-speaker-app rpi5
EOF
    exit 2
fi

if [[ "${DOCKERHUB_IMAGE}" != */* || "${DOCKERHUB_IMAGE}" == *:* ]]; then
    echo "DOCKERHUB_IMAGE must look like USER/IMAGE without a tag: ${DOCKERHUB_IMAGE}" >&2
    exit 2
fi

if [[ ! -f "${REPO_ROOT}/output/ecapa_tdnn_protonet_model.pth" ]]; then
    echo "Missing bundled model: ${REPO_ROOT}/output/ecapa_tdnn_protonet_model.pth" >&2
    exit 2
fi

if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
    docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use >/dev/null
else
    docker buildx use "${BUILDER_NAME}"
fi
docker buildx inspect --bootstrap "${BUILDER_NAME}" >/dev/null

tags=(--tag "${DOCKERHUB_IMAGE}:${IMAGE_TAG}")
if [[ "${PUBLISH_LATEST}" == "true" && "${IMAGE_TAG}" != "latest" ]]; then
    tags+=(--tag "${DOCKERHUB_IMAGE}:latest")
fi

echo "Building ${PLATFORM} image from ${REPO_ROOT}"
echo "Publishing ${DOCKERHUB_IMAGE}:${IMAGE_TAG}"

build_and_push() {
    docker buildx build \
        --platform "${PLATFORM}" \
        --file "${APP_DIR}/Dockerfile" \
        "${tags[@]}" \
        --output "type=registry,compression=gzip,compression-level=9" \
        --provenance=false \
        "${REPO_ROOT}"
}

attempt=1
until build_and_push; do
    if (( attempt >= PUSH_RETRIES )); then
        echo "Push failed after ${attempt} attempts." >&2
        exit 1
    fi
    delay=$((attempt * 15))
    echo "Push attempt ${attempt} failed; retrying in ${delay}s with fresh registry credentials." >&2
    sleep "${delay}"
    ((attempt += 1))
done

docker buildx imagetools inspect "${DOCKERHUB_IMAGE}:${IMAGE_TAG}"

cat <<EOF

Published: ${DOCKERHUB_IMAGE}:${IMAGE_TAG}

Run on the Raspberry Pi (X11/Openbox session):
docker run --pull=always --rm --entrypoint cat ${DOCKERHUB_IMAGE}:${IMAGE_TAG} /opt/protonet/run-rpi-image.sh | bash -s -- ${DOCKERHUB_IMAGE}:${IMAGE_TAG}
EOF
