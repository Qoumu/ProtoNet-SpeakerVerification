#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$APP_DIR"

load_env_file() {
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ""|\#*)
                continue
                ;;
        esac

        key=${line%%=*}
        value=${line#*=}
        case "$key" in
            ""|*[!A-Za-z0-9_]*)
                continue
                ;;
        esac

        case "$value" in
            \"*\")
                value=${value#\"}
                value=${value%\"}
                ;;
            \'*\')
                value=${value#\'}
                value=${value%\'}
                ;;
        esac

        export "$key=$value"
    done < "$1"
}

if [ -f .env ]; then
    load_env_file .env
fi

PROFILE=${APP_PROFILE:-raspberry-pi}
VENV_DIR=${SPEAKER_APP_VENV:-"$APP_DIR/.venv"}

if [ -n "${SPEAKER_APP_PYTHON:-}" ]; then
    PYTHON=$SPEAKER_APP_PYTHON
elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON=$VENV_DIR/bin/python
else
    PYTHON=python3
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'Speaker app Python is unavailable: %s\n' "$PYTHON" >&2
    printf 'Run ./scripts/setup-rpi.sh or set SPEAKER_APP_PYTHON.\n' >&2
    exit 1
fi

case "${APP_WINDOWED:-}" in
    1|true|TRUE|yes|YES|on|ON)
        set -- --profile "$PROFILE" --windowed "$@"
        ;;
    *)
        set -- --profile "$PROFILE" "$@"
        ;;
esac

exec "$PYTHON" -m speaker_app.main "$@"
