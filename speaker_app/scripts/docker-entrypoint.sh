#!/bin/sh
set -eu

if [ -z "${ENROLLMENT_PASSWORD_HASH:-}" ] && [ -z "${ENROLLMENT_PASSWORD_HASH_FILE:-}" ]; then
    export ENROLLMENT_PASSWORD_HASH_FILE=/app/config/enrollment_password_hash
fi

exec "$@"
