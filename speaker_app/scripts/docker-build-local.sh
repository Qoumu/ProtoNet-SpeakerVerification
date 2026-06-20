#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${APP_DIR}/.." && pwd)"

IMAGE="${LOCAL_IMAGE:-protonet-speaker-app:local}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

docker build \
    --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL}" \
    --build-arg "INSTALL_PULSE=true" \
    --tag "${IMAGE}" \
    --file "${APP_DIR}/Dockerfile" \
    "${REPO_ROOT}"

docker image inspect "${IMAGE}" \
    --format 'Built {{.RepoTags}} ({{.Architecture}}, {{.Size}} bytes)'
