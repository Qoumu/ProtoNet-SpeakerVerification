FROM python:3.11-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000
ARG PYTORCH_INDEX_URL=""
ARG INSTALL_PULSE=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_X11_NO_MITSHM=1 \
    APP_PROFILE=raspberry-pi \
    APP_WINDOWED=false \
    APP_DATA_DIR=/app/data \
    APP_LOG_DIR=/app/logs \
    APP_MODEL_DIR=/app/models \
    APP_MODEL_PATH=/app/models/ecapa_tdnn_protonet_model.pth \
    APP_MODEL_VERSION=ecapa-tdnn-protonet-v1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libasound2 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libportaudio2 \
        libsndfile1 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY speaker_app/requirements-rpi.txt speaker_app/requirements.txt ./
RUN pip install --upgrade pip

# Keep the largest wheels in separate layers. Registry retries can then reuse every
# layer that completed instead of restarting one multi-gigabyte upload.
RUN grep -Ev '^(torch|torchaudio|PySide6-Essentials)' requirements.txt \
        > /tmp/requirements-base.txt \
    && pip install -r /tmp/requirements-base.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install 'PySide6-Essentials==6.7.3' \
    && python -c "from PySide6.QtCore import QThreadPool; from PySide6.QtWidgets import QApplication"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && if [ -n "$PYTORCH_INDEX_URL" ]; then \
        pip install --index-url "$PYTORCH_INDEX_URL" 'torch==2.9.1' 'torchaudio==2.9.1'; \
    else \
        pip install 'torch==2.9.1' 'torchaudio==2.9.1'; \
    fi \
    && python -c \
        "import torch, torchaudio; assert torch.__version__.split('+')[0] == torchaudio.__version__.split('+')[0] == '2.9.1'"

# WSLg needs PulseAudio support; Raspberry Pi uses direct ALSA and omits it.
RUN if [ "$INSTALL_PULSE" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            libasound2-plugins \
            libpulse0; \
        rm -rf /var/lib/apt/lists/*; \
    fi

COPY speaker_app/pyproject.toml ./
COPY speaker_app/speaker_app ./speaker_app
COPY output/ecapa_tdnn_protonet_model.pth /app/models/ecapa_tdnn_protonet_model.pth
COPY speaker_app/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY speaker_app/scripts/run-rpi-image.sh /opt/protonet/run-rpi-image.sh
RUN pip install --no-deps . \
    && groupadd --gid "$APP_GID" speaker \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home speaker \
    && mkdir -p /app/config /app/data /app/logs \
    && python -c \
        "from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService as S; print(S.create_password_hash('protonet'))" \
        > /app/config/enrollment_password_hash \
    && chmod 0444 /app/config/enrollment_password_hash /app/models/ecapa_tdnn_protonet_model.pth \
    && chmod 0755 /usr/local/bin/docker-entrypoint.sh /opt/protonet/run-rpi-image.sh \
    && chown -R speaker:speaker /app/data /app/logs

USER speaker
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "speaker_app.main", "--profile", "raspberry-pi"]
