#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

VENV_DIR=${SPEAKER_APP_VENV:-"$HOME/Projects/.venv"}
PYTHON=${SPEAKER_APP_PYTHON:-"$VENV_DIR/bin/python"}

if [ ! -x "$PYTHON" ]; then
    printf 'Speaker app Python is unavailable: %s\n' "$PYTHON" >&2
    printf 'Set SPEAKER_APP_PYTHON or create the venv described in README.md.\n' >&2
    exit 1
fi

exec "$PYTHON" -m speaker_app.main --profile development --windowed "$@"
