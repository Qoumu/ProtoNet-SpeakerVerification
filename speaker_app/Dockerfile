FROM python:3.11-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libasound2 \
        libegl1 \
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
COPY requirements-rpi.txt requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements-rpi.txt

COPY pyproject.toml ./
COPY speaker_app ./speaker_app
RUN pip install --no-deps . \
    && groupadd --gid "$APP_GID" speaker \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home speaker \
    && mkdir -p /app/data /app/logs /app/models \
    && chown -R speaker:speaker /app/data /app/logs

USER speaker
CMD ["python", "-m", "speaker_app.main", "--profile", "raspberry-pi"]
