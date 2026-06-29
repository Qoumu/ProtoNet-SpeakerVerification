#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${APP_DIR}"

apt_packages=(
    python3-venv
    python3-pip
    libasound2
    libdbus-1-3
    libegl1
    libfontconfig1
    libgl1
    libglib2.0-0
    libgomp1
    libportaudio2
    libsndfile1
    libx11-xcb1
    libxcb-cursor0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-randr0
    libxcb-render-util0
    libxcb-shape0
    libxcb-xfixes0
    libxkbcommon-x11-0
    openbox
)

if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends "${apt_packages[@]}"
    elif [[ "$(id -u)" == "0" ]]; then
        apt-get update
        apt-get install -y --no-install-recommends "${apt_packages[@]}"
    else
        echo "Skipping apt install because sudo is unavailable and the user is not root." >&2
    fi
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rpi.txt
python -m pip install --no-deps -e .

if [[ ! -f .env ]]; then
    cp config/raspberry-pi.env.example .env
    echo "Created ${APP_DIR}/.env."
fi

echo "Native Raspberry Pi setup complete."
echo "Run python -m speaker_app.password_hash once to create the persistent enrollment password."
echo "Start with: ./scripts/run-app.sh"
