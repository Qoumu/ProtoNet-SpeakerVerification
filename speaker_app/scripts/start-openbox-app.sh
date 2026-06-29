#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

openbox-session &
exec "$SCRIPT_DIR/run-app.sh"
