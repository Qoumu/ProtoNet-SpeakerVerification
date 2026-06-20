#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-${SPEAKER_APP_IMAGE:-qumm296/sr-app:latest}}"
CONTAINER_NAME="${CONTAINER_NAME:-protonet-speaker}"
RUNTIME_DIR="${PROTONET_RUNTIME_DIR:-${HOME}/.local/share/protonet-speaker}"
DISPLAY_VALUE="${DISPLAY:-:0}"

fail() {
    echo "Error: $*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || fail "Docker is not installed"
docker info >/dev/null 2>&1 || fail "Docker is unavailable; check that it is running and your user can access it"
[[ -d /dev/snd ]] || fail "/dev/snd is unavailable; connect the microphone and check the Pi audio configuration"
[[ -d /tmp/.X11-unix ]] || fail "X11 is not running; start the Pi X11/Openbox desktop first"

if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    xauthority_file="${XAUTHORITY}"
elif [[ -f "${HOME}/.Xauthority" ]]; then
    xauthority_file="${HOME}/.Xauthority"
else
    fail "no Xauthority file found; run this command inside the Pi graphical X11 session"
fi

if audio_gid="$(getent group audio | cut -d: -f3)" && [[ -n "${audio_gid}" ]]; then
    :
else
    audio_gid="$(stat -c '%g' /dev/snd)"
fi

mkdir -p "${RUNTIME_DIR}/data" "${RUNTIME_DIR}/logs"

if ! docker pull "${IMAGE}"; then
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || fail "could not pull or find ${IMAGE}"
    echo "Using the existing local image ${IMAGE}" >&2
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker rm --force "${CONTAINER_NAME}" >/dev/null
fi

run_args=(
    --detach
    --name "${CONTAINER_NAME}"
    --restart unless-stopped
    --user "$(id -u):$(id -g)"
    --group-add "${audio_gid}"
    --device /dev/snd:/dev/snd
    --security-opt no-new-privileges:true
    --env APP_PROFILE=raspberry-pi
    --env APP_WINDOWED=false
    --env APP_RECOGNITION_THRESHOLD="${APP_RECOGNITION_THRESHOLD:-0.50}"
    --env DISPLAY="${DISPLAY_VALUE}"
    --env XAUTHORITY=/run/user/host.Xauthority
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw
    --volume "${xauthority_file}:/run/user/host.Xauthority:ro"
    --volume "${RUNTIME_DIR}/data:/app/data"
    --volume "${RUNTIME_DIR}/logs:/app/logs"
)

if [[ -n "${APP_AUDIO_DEVICE:-}" ]]; then
    run_args+=(--env "APP_AUDIO_DEVICE=${APP_AUDIO_DEVICE}")
fi
if [[ -n "${ENROLLMENT_PASSWORD_HASH:-}" ]]; then
    run_args+=(--env "ENROLLMENT_PASSWORD_HASH=${ENROLLMENT_PASSWORD_HASH}")
fi

docker run "${run_args[@]}" "${IMAGE}"
sleep 2
docker ps --filter "name=^/${CONTAINER_NAME}$"
docker logs --tail 30 "${CONTAINER_NAME}"
