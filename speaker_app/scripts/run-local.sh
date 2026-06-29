#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export SPEAKER_APP_VENV=${SPEAKER_APP_VENV:-"$HOME/Projects/.venv"}
exec "$SCRIPT_DIR/run-app.sh" --profile development --windowed "$@"
