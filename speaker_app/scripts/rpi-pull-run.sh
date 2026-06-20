#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${APP_DIR}"

if [[ ! -f .env ]]; then
    echo "Missing ${APP_DIR}/.env; copy config/raspberry-pi.env.example and edit it first." >&2
    exit 2
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ -z "${SPEAKER_APP_IMAGE:-}" || "${SPEAKER_APP_IMAGE}" == *YOUR_DOCKERHUB_USER* ]]; then
    echo "Set SPEAKER_APP_IMAGE in .env to the Docker Hub image you pushed." >&2
    exit 2
fi

mkdir -p runtime/data runtime/logs

docker compose --profile raspberry-pi pull speaker-app
docker compose --profile raspberry-pi up -d --no-build speaker-app
docker compose --profile raspberry-pi ps speaker-app
