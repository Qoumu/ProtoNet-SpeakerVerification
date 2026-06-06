FROM python:3.11-slim

ARG TARGETPLATFORM
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_PROGRESS_BAR=off \
    NUMBA_DISABLE_JIT=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_runtime.txt ./requirements_runtime.txt

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --prefer-binary \
    --only-binary=:all: \
    --no-compile \
    -r requirements_runtime.txt

COPY model ./model
COPY utils ./utils
COPY enroll_lite.py ./enroll_lite.py
COPY speaker_cli.py ./speaker_cli.py
COPY output/ECAPATDNN_DataAug_protonet_model.pth ./output/ECAPATDNN_DataAug_protonet_model.pth
COPY output/enrolled_speakers.pt ./output/enrolled_speakers.pt

RUN mkdir -p /data

LABEL org.opencontainers.image.title="Protonet Speaker Recognition" \
      org.opencontainers.image.description="Speaker verification and enrollment runtime for ECAPA-TDNN Protonet" \
      org.opencontainers.image.source="https://hub.docker.com" \
      org.opencontainers.image.vendor="OpenAI Codex" \
      org.opencontainers.image.platform="${TARGETPLATFORM}" \
      org.opencontainers.image.architecture="${TARGETARCH}"

ENTRYPOINT ["python", "speaker_cli.py"]
CMD ["verify"]
