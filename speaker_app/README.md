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

## Raspberry Pi 5 Deployment

Required host assumptions: 64-bit Raspberry Pi OS, Docker Engine with Compose,
X11/Openbox, a working USB microphone exposed as `/dev/snd`, and an existing
Xauthority file. Verify these values on the actual Pi; they are host-specific.

```bash
cd speaker_app
cp config/raspberry-pi.env.example .env
mkdir -p runtime/data runtime/logs runtime/secrets
python -m speaker_app.password_hash > runtime/secrets/enrollment_password_hash
chmod 600 runtime/secrets/enrollment_password_hash
```

Edit `.env` with the Pi's `XAUTHORITY_FILE`, `AUDIO_GID`, microphone selector,
UID/GID, model version, and calibrated threshold. The Compose file mounts the parent
repository's `output/` directory read-only as `/app/models`; place the checkpoint
there or change the model mount/path.

Build and start after X11/Openbox is available:

```bash
docker compose --profile raspberry-pi build
docker compose --profile raspberry-pi up -d
docker compose logs -f speaker-app
```

The container maps only `/dev/snd`, the X11 socket/Xauthority, persistent data and
logs, the read-only model directory, and the password-hash secret. It does not use
`privileged` mode. Real Raspberry Pi display, audio permissions, latency, restart,
and end-to-end recognition still require hardware-in-the-loop verification.

## Tests

```bash
python -m pytest
```

Tests cover authorization lockout/expiry, embedding math, SQLite persistence and
compatibility filtering, WAV validation, enrollment, duplicate IDs, recognized
speakers, and unknown-speaker rejection.
