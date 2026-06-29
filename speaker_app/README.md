# Raspberry Pi Speaker Recognition App

PySide6 touchscreen application for the ECAPA-TDNN model trained in the parent
repository. The app now deploys natively on Raspberry Pi OS and can also run
windowed on a development laptop.

## Implemented Flow

- Enrollment: validated speaker ID, scrypt password authorization, five accepted
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
and passed to the model. Configure these with `APP_AUDIO_GAIN_DB` and
`APP_AUDIO_DENOISE_ENABLED`.

Accepted enrollment clips are retained by default at:

```text
data/enrollment_audio/<speaker_id>/<speaker_id>_clip_01.wav
data/enrollment_audio/<speaker_id>/<speaker_id>_clip_02.wav
...
```

Set `APP_RETAIN_ENROLLMENT_AUDIO=false` to delete them after a successful enrollment.
Cancelled or rejected clips are always removed.

## Raspberry Pi Deployment

Expected host setup:

- 64-bit Raspberry Pi OS.
- A graphical X11/Openbox session for the touchscreen.
- A working microphone visible to PortAudio/ALSA.
- The trained model at `../output/ecapa_tdnn_protonet_model.pth` relative to this
  `speaker_app` directory, or `APP_MODEL_PATH` set to another model file.

Install system packages, create `.venv`, install Python dependencies, and create
`.env` from the Raspberry Pi template:

```bash
cd speaker_app
./scripts/setup-rpi.sh
```

Generate the enrollment password hash once:

```bash
. .venv/bin/activate
python -m speaker_app.password_hash
```

This saves the hash to `data/enrollment_password_hash` with owner-only file
permissions. The app reads that file automatically on every restart. To change the
password later, run `python -m speaker_app.password_hash --force`.
If your shell has an old `ENROLLMENT_PASSWORD_HASH_FILE` value, this setup command
still writes to `APP_DATA_DIR/enrollment_password_hash`; pass `--output` only when
you intentionally want a different file.
Old container paths such as `APP_DATA_DIR=/app/data` are ignored by native runs and
fall back to the repo-local `data/` directory.

Run the app full-screen:

```bash
./scripts/run-app.sh
```

For debug runs in a normal desktop window:

```bash
./scripts/run-app.sh --windowed --log-level DEBUG
```

`run-app.sh` loads `speaker_app/.env` when it exists, defaults to the
`raspberry-pi` profile, and uses `.venv/bin/python` unless `SPEAKER_APP_PYTHON` is
set.

To start Openbox and then the app from a lightweight Pi session:

```bash
./scripts/start-openbox-app.sh
```

## Native Laptop Setup

Run from this directory:

```bash
cd speaker_app
python3 -m venv ~/Projects/.venv
. ~/Projects/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,onnx]'
python -m speaker_app.password_hash
```

The command saves `data/enrollment_password_hash`. Then run:

```bash
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

## Configuration

All runtime values use environment variables. Start from:

- `config/development.env.example`
- `config/raspberry-pi.env.example`

Both profiles use repo-local writable paths by default:

- Data and SQLite database: `speaker_app/data`
- Logs: `speaker_app/logs`
- Model directory: `output` in the parent repository

Important values:

- `APP_MODEL_PATH`: model file to load.
- `APP_AUDIO_DEVICE`: ALSA/PortAudio device name or numeric index.
- `APP_RECOGNITION_THRESHOLD`: cosine-similarity acceptance threshold.
- `APP_RETAIN_ENROLLMENT_AUDIO`: keep or remove accepted enrollment WAV files.
- `ENROLLMENT_PASSWORD_HASH_FILE`: optional path override for the password hash file.
- `ENROLLMENT_PASSWORD_HASH`: optional direct hash override for temporary testing.

Debug messages are written to both the terminal and `logs/speaker_app.log`. They
cover startup, microphone state, recording and stop events, audio-quality metrics,
worker timing, embedding extraction, enrollment stages, database writes, similarity
scores, recognition decisions, cancellation, and errors. Passwords and raw embedding
vectors are never logged. Use `--log-level INFO` for less output.

For deterministic testing without a microphone:

```bash
python -m speaker_app.main --profile development --windowed \
  --test-audio /absolute/path/to/speech.wav
```

The same WAV is copied for each recording in this mode. It is intended for GUI and
workflow tests; use distinct service fixtures in automated recognition tests.

## Tests

```bash
python -m pytest
```

Tests cover authorization lockout/expiry, embedding math, SQLite persistence and
compatibility filtering, WAV validation, enrollment, duplicate IDs, recognized
speakers, and unknown-speaker rejection.
