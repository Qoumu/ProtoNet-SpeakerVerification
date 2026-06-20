# Raspberry Pi Speaker Recognition App

PySide6 touchscreen application for the ECAPA-TDNN model trained in the parent
repository. The same package runs windowed on a development laptop and full-screen
in Docker on Raspberry Pi OS Lite with host X11/Openbox and ALSA.

## Implemented Flow

- Enrollment: validated speaker ID, scrypt password authorization, six accepted
  clips, one normalized embedding per clip, normalized mean prototype, SQLite save.
- Recognition: validated recording, embedding extraction, compatible-profile lookup,
  cosine comparison, threshold acceptance, speaker or `Unknown Speaker` result.
- Runtime: PyTorch `.pth` and ONNX models, file-backed test audio, live PortAudio
  microphone input, Qt worker threads, persistent model-version-aware profiles.

During enrollment, press `Record Clip`, speak for at least two seconds, then press
`Stop Recording`. The app validates the WAV and enables `Next Clip` only when the
full clip is long enough, not too quiet, and not clipped. Recording also stops
automatically at the configured five-second maximum.

The application does not apply VAD: it does not trim silence, split speech regions,
or concatenate voiced frames. Every captured sample is retained. The full-duration
clip receives non-stationary noise reduction and configurable gain before it is saved
and passed to the model. Configure these with `APP_AUDIO_GAIN_DB` (default `20.0`)
and `APP_AUDIO_DENOISE_ENABLED` (default `true`). Duration, whole-clip RMS, and
clipping checks remain, but they never alter the waveform length.

The default threshold, `0.70`, follows the repository's existing ONNX test default.
It is provisional and must be replaced with a value selected from held-out FAR/FRR
or EER evaluation for the deployed microphone and model.

Accepted enrollment clips are retained by default at:

```text
data/enrollment_audio/<speaker_id>/<speaker_id>_clip_01.wav
data/enrollment_audio/<speaker_id>/<speaker_id>_clip_02.wav
...
```

Set `APP_RETAIN_ENROLLMENT_AUDIO=false` to delete them after a successful enrollment.
Cancelled or rejected clips are always removed.

The home page includes `Remove Enrolled Speaker`. It displays the enrolled speaker IDs
and display names, lets the user select one profile, then requires the enrollment
password and a confirmation dialog. Only the selected SQLite profile is deleted; its
WAV files under `data/enrollment_audio/` are preserved.

## Native Laptop Setup

Run from this directory:

```bash
cd speaker_app
python3 -m venv ~/Projects/.venv  # Skip when this shared venv already exists.
. ~/Projects/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,onnx]'
python -m speaker_app.password_hash
```

Export the printed scrypt value only in the current shell or put it in a protected,
ignored environment file:

```bash
export ENROLLMENT_PASSWORD_HASH='scrypt$...'
python -m pytest
python -m speaker_app.main --profile development --windowed --log-level DEBUG
```

You can launch without activating the environment by using the venv interpreter
directly:

```bash
~/Projects/.venv/bin/python -m speaker_app.main \
  --profile development --windowed --log-level DEBUG
```

Alternatively, `./scripts/run-local.sh --log-level DEBUG` automatically uses
`~/Projects/.venv`. Override it with `SPEAKER_APP_PYTHON=/path/to/python` when needed.

Debug messages are written to both the terminal and `logs/speaker_app.log`. They
cover startup, microphone state, recording and stop events, audio-quality metrics,
worker timing, embedding extraction, enrollment stages, database writes, similarity
scores, recognition decisions, cancellation, and errors. Passwords and raw embedding
vectors are never logged. Use `--log-level INFO` for less output.

The default development model is
`../output/ecapa_tdnn_protonet_model.pth`. Override it with `APP_MODEL_PATH`.

For deterministic testing without a microphone:

```bash
python -m speaker_app.main --profile development --windowed \
  --test-audio /absolute/path/to/speech.wav
```

The same WAV is copied for each recording in this mode. It is intended for GUI and
workflow tests; use distinct service fixtures in automated recognition tests.

## Configuration

All platform values use environment variables. Start from:

- `config/development.env.example`
- `config/raspberry-pi.env.example`

Speaker profiles default to `data/speakers.db`. Enrollment recordings are retained
after a successful save unless `APP_RETAIN_ENROLLMENT_AUDIO=false`. Recognition
recordings are always temporary.

## Local Docker Build

Build a native image for the current laptop:

```bash
cd speaker_app
./scripts/docker-build-local.sh
```

This creates `protonet-speaker-app:local`. The build context is the repository root
because the image includes `output/ecapa_tdnn_protonet_model.pth`. The local build
uses PyTorch's CPU wheel index so it does not download the multi-gigabyte CUDA
runtime. Override the tag with `LOCAL_IMAGE=my-image:test` when needed.

Basic container checks:

```bash
docker run --rm protonet-speaker-app:local \
  python -m speaker_app.main --help

docker run --rm protonet-speaker-app:local \
  python -c "from speaker_app.config import load_config; from speaker_app.model import load_embedding_extractor; load_embedding_extractor(load_config('raspberry-pi')); print('model ready')"
```

The complete GUI additionally needs the X11 socket/Xauthority and microphone mappings
from `compose.yaml`. Use the development profile natively first; container GUI/audio
behavior still depends on host X11 and `/dev/snd` permissions.

### Run Locally on WSLg

This repository's laptop environment uses WSLg, where X11 is exposed through
`/tmp/.X11-unix` and microphone audio through `/mnt/wslg/PulseServer`. After building
the local image, run:

```bash
cd speaker_app
docker compose -f compose.local.yaml up
```

The window opens in WSLg and uses the `pulse` PortAudio device. Application data is
mounted from `speaker_app/data`, logs from `speaker_app/logs`, and the model is loaded
from the image. Stop it with `Ctrl+C`, or use detached mode:

```bash
docker compose -f compose.local.yaml up -d
docker compose -f compose.local.yaml logs -f
docker compose -f compose.local.yaml down
```

## Build and Push for Raspberry Pi 5

The Raspberry Pi 5 normally runs a 64-bit `linux/arm64` image. Create a Docker Hub
repository such as `YOUR_USER/protonet-speaker-app`, then authenticate on the build
laptop. Use a Docker Hub access token instead of putting a password in this repo.

```bash
cd speaker_app
docker login --username YOUR_DOCKERHUB_USER
./scripts/docker-buildx-push.sh \
  qumm296/sr-app rpi5
```

The script creates or reuses a Buildx container builder, cross-builds `linux/arm64`,
pushes both `:rpi5` and `:latest`, and inspects the published manifest. To publish only
the requested tag:

```bash
PUBLISH_LATEST=false ./scripts/docker-buildx-push.sh \
  YOUR_DOCKERHUB_USER/protonet-speaker-app rpi5
```

The image contains the application, Python/native dependencies, trained model, and a
default enrollment-password hash. The default enrollment password is `protonet`.
Speaker profiles and enrollment recordings are not included because they contain
user data; the launcher persists them on the Pi.

## Raspberry Pi 5 Deployment

Required host assumptions: 64-bit Raspberry Pi OS, Docker Engine, X11/Openbox, a
working USB microphone exposed as `/dev/snd`, and an Xauthority file. Run this single
command inside the Pi graphical session; no repository checkout is required:

```bash
docker run --pull=always --rm --entrypoint cat qumm296/sr-app:latest \
  /opt/protonet/run-rpi-image.sh | \
  bash -s -- qumm296/sr-app:latest
```

The first container prints its bundled host launcher, and host `bash` executes it.
The launcher pulls the current image, supplies the X11 and `/dev/snd` mappings,
recreates the application container, and prints startup logs. It persists the SQLite
database, enrollment WAV files, and logs under
`~/.local/share/protonet-speaker/`. A literal bare `docker run IMAGE` cannot provide
host devices or bind mounts; Docker requires those settings at container creation.

When the repository is present on the Pi, the equivalent shorter command is:

```bash
./speaker_app/scripts/run-rpi-image.sh qumm296/sr-app:latest
```

Manage the running application with:

```bash
docker logs -f protonet-speaker
docker restart protonet-speaker
docker stop protonet-speaker
```

Override the built-in password by exporting `ENROLLMENT_PASSWORD_HASH` before running
the launcher. Real Raspberry Pi display, audio permissions, inference latency, and
end-to-end recognition still require hardware-in-the-loop verification.

## Tests

```bash
python -m pytest
```

Tests cover authorization lockout/expiry, embedding math, SQLite persistence and
compatibility filtering, WAV validation, enrollment, duplicate IDs, recognized
speakers, and unknown-speaker rejection.
