# Speaker Recognition App — AI Agent Context

## 1. Project Overview

Build a touchscreen-friendly speaker recognition application with a Python backend and a PySide6/Qt Widgets GUI.

Development and deployment must occur in two distinct stages:

1. **Local development and testing on the developer's laptop**
   - Run the application directly in a local Python virtual environment.
   - Use a normal resizable application window by default.
   - Test GUI navigation, microphone recording, enrollment, recognition, database persistence, error handling, and model inference locally.
   - Use mocks or test audio files when microphone hardware is unavailable.

2. **Dockerized deployment on Raspberry Pi**
   - Containerize the verified application only after local native testing succeeds.
   - Run the container on Raspberry Pi OS Lite with Openbox/X11.
   - Connect the container to the Raspberry Pi display, microphone, persistent data directories, and model files.
   - Run the GUI full-screen on the Raspberry Pi touchscreen or attached display.

The system has two main user functions:

1. **Enroll Speaker**
   - Ask the user for a speaker ID and optional display name.
   - Require an enrollment authorization password before recording begins.
   - Record approximately 5–6 audio clips.
   - Validate the recorded audio.
   - Extract one embedding from each clip.
   - Combine the embeddings into a representative speaker voiceprint.
   - Store the speaker ID, metadata, and embedding.

2. **Recognize Speaker**
   - Record a new speech sample.
   - Extract its speaker embedding.
   - Compare it with all stored speaker embeddings.
   - Display the recognized speaker ID.
   - Display `Unknown Speaker` when no stored speaker passes the recognition threshold.

---

## 2. Development and Deployment Targets

### 2.1 Stage 1 — Local Laptop Development

The first implementation must be developed, run, and tested directly on the developer's laptop without Docker.

- Runtime: Native Python virtual environment
- Backend language: Python
- GUI framework: PySide6 using Qt Widgets
- Display mode: Normal resizable window during development
- Database: Local SQLite file
- Audio input: Laptop microphone or attached USB microphone
- Model execution: Local CPU or available laptop accelerator
- Configuration profile: `development`
- Launch command:

```bash
python -m speaker_app.main --profile development --windowed
```

The local application must support:

- GUI-only testing with dummy services
- Test WAV files instead of live recording
- Live microphone testing
- Temporary local database paths
- Debug logging
- Unit and integration tests

The agent must not require Docker to run the first working version.

### 2.2 Stage 2 — Dockerized Raspberry Pi Deployment

After local testing passes, package the application into a Docker image for Raspberry Pi.

- Device: Raspberry Pi
- Operating system: Raspberry Pi OS Lite / CLI-only environment
- Host window manager: Openbox
- Host display server: X11
- Container runtime: Docker Engine with Docker Compose
- Display mode: Full-screen application
- Audio input: Raspberry Pi-compatible USB microphone
- Database: SQLite stored on a persistent host-mounted volume
- Configuration profile: `raspberry-pi`
- Container architecture: Must match the Raspberry Pi architecture, normally ARM64 on a 64-bit OS

The Docker container must integrate with host resources rather than attempting to run a full desktop environment inside the container.

Required host integrations may include:

- X11 display socket
- Xauthority credentials
- ALSA audio devices such as `/dev/snd`
- Persistent application data directory
- Persistent log directory
- Read-only model directory when appropriate

Expected host-side launch flow:

```text
Raspberry Pi boots
→ X11 and Openbox start
→ Docker Compose starts the application container
→ Qt connects to the host X11 display
→ The application opens full-screen
```

Example deployment command:

```bash
docker compose --profile raspberry-pi up -d
```

### 2.3 Environment Differences

The application must not assume that the laptop and Raspberry Pi use the same:

- Operating system
- CPU architecture
- Microphone name or device index
- Display resolution
- File paths
- User ID or group ID
- Audio device permissions
- Model runtime or acceleration provider

All platform-specific values must come from configuration or environment variables.

---

## 3. Main Design Principles

The application must follow these principles:

- Keep GUI code separate from speaker-recognition logic.
- Do not place model inference or audio recording directly inside button event handlers.
- Do not block the Qt GUI thread.
- Run recording, preprocessing, embedding extraction, enrollment, and recognition in worker threads.
- Use Qt signals to report status, progress, results, and errors.
- Use a single main window with page navigation through `QStackedWidget`.
- Use large buttons and large text suitable for a Raspberry Pi touchscreen.
- Keep the interface simple and readable.
- Only allow one audio or inference operation at a time.
- Disable incompatible buttons while a task is running.
- Handle microphone, recording, model, database, and audio-quality errors explicitly.
- Use structured return values instead of relying on terminal `print()` output.
- Store the embedding model version with each enrolled speaker.
- Keep local-native execution working independently of Docker.
- Use environment-specific configuration profiles for laptop and Raspberry Pi.
- Do not hardcode microphone indices, display identifiers, absolute host paths, user IDs, or group IDs.
- Isolate host hardware access behind service interfaces so it can be mocked during local tests.
- Keep application data outside the container filesystem through persistent volumes.
- Do not use Docker `--privileged` unless a documented hardware requirement makes it unavoidable.

---

## 4. Application Architecture

```text
┌──────────────────────────────────────────────┐
│                  PySide6 GUI                 │
│ Home / Enrollment / Recognition / Result UI │
└──────────────────────┬───────────────────────┘
                       │ Qt signals and slots
┌──────────────────────▼───────────────────────┐
│                App Controller                │
│ Navigation, state transitions, task control │
└───────────────┬─────────────────┬────────────┘
                │                 │
┌───────────────▼──────────┐ ┌────▼─────────────────────┐
│      Audio Service       │ │ Speaker Recognition Core │
│ Record / validate / VAD  │ │ Embedding / comparison   │
└───────────────┬──────────┘ └────┬─────────────────────┘
                │                 │
                └──────────┬──────┘
                           │
                 ┌─────────▼─────────┐
                 │ Speaker Repository │
                 │ SQLite persistence │
                 └───────────────────┘
```

---

## 5. Recommended Project Structure

```text
speaker_app/
├── speaker_app/
│   ├── __init__.py
│   ├── main.py
│   ├── app_controller.py
│   ├── config.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── home_page.py
│   │   ├── enroll_info_page.py
│   │   ├── enroll_authorization_page.py
│   │   ├── enroll_recording_page.py
│   │   ├── enroll_processing_page.py
│   │   ├── recognition_page.py
│   │   ├── recognition_result_page.py
│   │   ├── message_dialog.py
│   │   └── styles.qss
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_service.py
│   │   ├── enrollment_authorization_service.py
│   │   ├── enrollment_service.py
│   │   ├── recognition_service.py
│   │   └── speaker_repository.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── embedding_extractor.py
│   │   └── model_loader.py
│   │
│   ├── workers/
│   │   ├── __init__.py
│   │   └── backend_worker.py
│   │
│   └── domain/
│       ├── __init__.py
│       └── results.py
│
├── config/
│   ├── development.env.example
│   └── raspberry-pi.env.example
│
├── data/
│   ├── speakers.db
│   ├── temporary_audio/
│   └── enrollment_audio/
│
├── models/
├── logs/
│
├── scripts/
│   ├── run-local.sh
│   ├── run-local.ps1
│   └── start-openbox-app.sh
│
├── tests/
│   ├── unit/
│   │   ├── test_repository.py
│   │   ├── test_authorization.py
│   │   └── test_embedding_comparison.py
│   ├── integration/
│   │   ├── test_enrollment_flow.py
│   │   └── test_recognition_flow.py
│   └── fixtures/
│       └── audio/
│
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-rpi.txt
└── README.md
```

Rules:

- The Python package must run locally without Docker.
- `Dockerfile` and `compose.yaml` are deployment layers around the same application package.
- Do not create separate, diverging laptop and Raspberry Pi codebases.
- Environment files must not contain committed secrets.
- The SQLite database, recordings, logs, and downloaded model assets must not be baked into the Docker image.

---

## 6. User Interface Pages

### 6.1 Home Page

The first screen must show two main buttons:

- `Enroll Speaker`
- `Recognize Speaker`

Additional status text:

- Microphone status
- Model status
- Database status

Example:

```text
┌────────────────────────────────────────┐
│       SPEAKER RECOGNITION SYSTEM       │
│                                        │
│      [      ENROLL SPEAKER      ]      │
│                                        │
│      [    RECOGNIZE SPEAKER     ]      │
│                                        │
│         Microphone: Ready              │
└────────────────────────────────────────┘
```

### 6.2 Enrollment Information Page

Inputs:

- Speaker ID: required
- Display name: optional

Buttons:

- `Continue`
- `Back`

Validation rules:

- Speaker ID cannot be empty.
- Speaker ID must be unique unless overwrite behavior is explicitly selected.
- Trim leading and trailing whitespace.
- Restrict the speaker ID to safe characters where possible.

Recommended allowed format:

```text
[A-Za-z0-9_-]
```

After the information passes validation, navigate to the **Enrollment Authorization Page**. Do not begin recording yet.

### 6.3 Enrollment Authorization Page

Before recording begins, require an enrollment authorization password.

Display:

- Speaker ID being enrolled
- Masked password input field
- Optional password visibility toggle
- Instruction: `Enter the enrollment password to continue`

Buttons:

- `Confirm`
- `Back`

Required behavior:

- A correct password navigates to the Enrollment Recording Page.
- An incorrect password keeps the user on the page and displays a short error.
- Clear the password field after every failed attempt.
- Preserve the entered speaker information when navigating back.
- Clear the password and all pending enrollment state when enrollment is cancelled.
- Never print, log, persist, or include the plaintext password in an exception.
- Limit repeated failed attempts according to configuration.
- The password authorizes access to enrollment only and must not be associated with a speaker profile.

Security requirements:

- Never hardcode a plaintext password in source code.
- Store only a secure password hash outside version control.
- Use a password-hashing algorithm such as Argon2id, scrypt, or bcrypt.
- Verify passwords through a dedicated authorization service.
- Load the hash from a secret environment variable, Docker secret, or protected host file.

### 6.4 Enrollment Recording Page

Display:

- Current clip number, such as `Clip 2 of 6`
- Recording state
- Countdown before recording
- Audio duration
- Audio-quality status
- Overall enrollment progress

Buttons:

- `Record Clip`
- `Record Again`
- `Next Clip`
- `Cancel Enrollment`

Initial implementation should use manual confirmation between clips because it is easier to test and debug.

### 6.5 Enrollment Processing Page

Display processing stages such as:

```text
Validating clips...
Extracting embeddings...
Combining embeddings...
Saving speaker profile...
```

Display progress when meaningful.

### 6.6 Enrollment Success Page

Display:

- `Speaker enrolled successfully`
- Speaker ID
- Number of accepted clips

Button:

- `Return Home`

### 6.7 Recognition Page

Initial state:

- `Start Recording`
- `Back`
- Microphone status

Processing states:

```text
Recording...
Validating audio...
Extracting embedding...
Comparing speakers...
```

### 6.8 Recognition Result Page

For a recognized speaker:

```text
Recognized Speaker

USER_001

Similarity: 0.87
```

Buttons:

- `Recognize Again`
- `Return Home`

For a rejected match:

```text
Unknown Speaker

No enrolled speaker passed
the acceptance threshold.
```

Use the label **Similarity Score**, not **Confidence**, unless the score has been statistically calibrated as a probability.

---

## 7. Application State Flow

```text
HOME
├── ENROLL_INFO
│   └── ENROLL_AUTHORIZATION
│       └── ENROLL_RECORDING
│           └── ENROLL_PROCESSING
│               └── ENROLL_SUCCESS
│
└── RECOGNITION_READY
    └── RECOGNITION_RECORDING
        └── RECOGNITION_PROCESSING
            └── RECOGNITION_RESULT
```

The controller must manage these states and prevent invalid transitions.

Examples:

- The user cannot enter `ENROLL_RECORDING` until authorization succeeds.
- The user cannot process enrollment before all required clips are accepted.
- The user cannot start recognition while enrollment is running.
- The user cannot navigate away during a database write unless cancellation is safely supported.
- Going back from authorization preserves the pending speaker ID and display name.
- Cancelling enrollment clears pending information, password data, and temporary recordings.

---

## 8. Backend Interfaces

Existing standalone Python scripts should gradually be refactored into importable functions.

Do not make the final GUI depend on parsing human-readable terminal output.

### 8.1 Result Data Classes

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordingResult:
    success: bool
    audio_path: Path | None
    duration_seconds: float
    sample_rate: int
    message: str


@dataclass(frozen=True)
class AudioValidationResult:
    accepted: bool
    duration_seconds: float
    speech_duration_seconds: float
    peak_level: float
    message: str


@dataclass(frozen=True)
class AuthorizationResult:
    accepted: bool
    remaining_attempts: int
    message: str


@dataclass(frozen=True)
class EnrollmentResult:
    success: bool
    speaker_id: str
    accepted_clip_count: int
    message: str


@dataclass(frozen=True)
class RecognitionResult:
    accepted: bool
    speaker_id: str | None
    similarity: float
    message: str
```

### 8.2 Enrollment Authorization Service

```python
class EnrollmentAuthorizationService:
    def verify_password(
        self,
        password: str,
    ) -> AuthorizationResult:
        ...
```

Responsibilities:

- Verify the entered password against a securely stored hash.
- Track failed attempts for the current authorization session.
- Reset failed-attempt state after successful authorization.
- Return a user-safe result without exposing the stored hash.
- Never save the plaintext password.
- Work in both local and container environments through injected configuration.

### 8.3 Audio Service

```python
from pathlib import Path


class AudioService:
    def record_clip(
        self,
        output_path: Path,
        duration_seconds: float,
    ) -> RecordingResult:
        ...

    def validate_audio(
        self,
        audio_path: Path,
    ) -> AudioValidationResult:
        ...
```

Responsibilities:

- Open and validate the configured microphone.
- Record PCM audio.
- Save audio as WAV.
- Check minimum duration.
- Check whether speech is present.
- Detect silence or extremely low energy.
- Detect clipping where practical.
- Return structured errors.
- Support a mock or file-backed implementation for automated local tests.

### 8.4 Embedding Extractor

```python
from pathlib import Path
import numpy as np


class EmbeddingExtractor:
    @property
    def model_version(self) -> str:
        ...

    @property
    def embedding_dimension(self) -> int:
        ...

    def extract(self, audio_path: Path) -> np.ndarray:
        ...
```

Requirements:

- Load the speaker-recognition model once at application startup.
- Reuse the loaded model for all enrollment and recognition operations.
- Return a one-dimensional `float32` NumPy array.
- Apply L2 normalization before returning, or clearly document where normalization occurs.
- Raise meaningful exceptions for unreadable or invalid audio.
- Allow the runtime provider to differ between laptop and Raspberry Pi.

### 8.5 Enrollment Service

```python
from pathlib import Path


class EnrollmentService:
    def enroll(
        self,
        speaker_id: str,
        display_name: str | None,
        audio_paths: list[Path],
    ) -> EnrollmentResult:
        ...
```

Responsibilities:

1. Validate the speaker ID.
2. Validate the number of clips.
3. Extract one embedding per accepted clip.
4. Combine embeddings into a speaker prototype.
5. Normalize the final prototype.
6. Save the speaker record.
7. Return a structured result.

Authorization must be completed before this service is called. Do not pass the password into the enrollment service.

### 8.6 Recognition Service

```python
from pathlib import Path


class RecognitionService:
    def recognize(
        self,
        audio_path: Path,
    ) -> RecognitionResult:
        ...
```

Responsibilities:

1. Validate the recognition audio.
2. Extract the query embedding.
3. Load all compatible speaker profiles.
4. Compute similarity with each speaker.
5. Select the highest score.
6. Apply the recognition threshold.
7. Return either the speaker ID or `Unknown Speaker`.

---

## 9. Embedding Processing

For `N` enrollment clips, extract embeddings:

```text
E1, E2, ..., EN
```

Normalize each embedding:

\[
\hat{E_i} = \frac{E_i}{\|E_i\|_2}
\]

Calculate the mean prototype:

\[
P = \frac{1}{N}\sum_{i=1}^{N}\hat{E_i}
\]

Normalize the final prototype:

\[
\hat{P} = \frac{P}{\|P\|_2}
\]

Recommended implementation:

```python
import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length embedding.")

    return vector / norm


def create_speaker_prototype(
    embeddings: list[np.ndarray],
) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one embedding is required.")

    normalized = np.stack(
        [l2_normalize(item) for item in embeddings],
        axis=0,
    )

    prototype = normalized.mean(axis=0)
    return l2_normalize(prototype)
```

Store the final prototype.

Optionally store individual clip embeddings for later experiments or profile-quality checks.

---

## 10. Speaker Comparison

Use cosine similarity for normalized embeddings.

For two normalized embeddings, cosine similarity is equivalent to their dot product.

```python
def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = l2_normalize(first)
    second = l2_normalize(second)
    return float(np.dot(first, second))
```

Recognition logic:

```python
best_speaker_id = None
best_score = -1.0

for speaker in speakers:
    score = cosine_similarity(
        query_embedding,
        speaker.embedding,
    )

    if score > best_score:
        best_score = score
        best_speaker_id = speaker.speaker_id

if best_score < recognition_threshold:
    return RecognitionResult(
        accepted=False,
        speaker_id=None,
        similarity=best_score,
        message="Unknown speaker",
    )

return RecognitionResult(
    accepted=True,
    speaker_id=best_speaker_id,
    similarity=best_score,
    message="Speaker recognized",
)
```

The threshold must come from validation experiments, such as:

- Equal Error Rate
- Detection Error Tradeoff curve
- False Acceptance Rate
- False Rejection Rate

Do not invent a threshold without documenting that it is temporary.

---

## 11. SQLite Database Design

Use SQLite because the application is local and does not require a separate database server.

Recommended table:

```sql
CREATE TABLE IF NOT EXISTS speakers (
    speaker_id TEXT PRIMARY KEY,
    display_name TEXT,
    embedding BLOB NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    number_of_samples INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Recommended Python conversion:

```python
embedding_blob = embedding.astype("float32").tobytes()
```

Loading:

```python
embedding = np.frombuffer(
    embedding_blob,
    dtype=np.float32,
).copy()
```

Validation when loading:

- Stored model version matches the current model.
- Stored embedding dimension matches the current model output.
- Embedding data is not empty or corrupted.

### Repository Interface

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    display_name: str | None
    embedding: np.ndarray
    model_version: str
    number_of_samples: int


class SpeakerRepository:
    def initialize(self) -> None:
        ...

    def exists(self, speaker_id: str) -> bool:
        ...

    def save(self, profile: SpeakerProfile) -> None:
        ...

    def get(self, speaker_id: str) -> SpeakerProfile | None:
        ...

    def get_all_compatible(
        self,
        model_version: str,
        embedding_dimension: int,
    ) -> list[SpeakerProfile]:
        ...

    def delete(self, speaker_id: str) -> bool:
        ...
```

---

## 12. Threading Requirements

The Qt main thread must only handle:

- Drawing widgets
- Responding to user input
- Updating labels and progress bars
- Navigating between pages

The following must run outside the main GUI thread:

- Audio recording
- Audio validation
- Voice activity detection
- Embedding extraction
- Enrollment processing
- Database writes
- Speaker comparison

Use `QThreadPool` and `QRunnable`, or a dedicated `QObject` moved to a `QThread`.

Recommended generic worker:

```python
from collections.abc import Callable
from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class BackendWorker(QRunnable):
    def __init__(
        self,
        function: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(
                *self.args,
                **self.kwargs,
            )
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
```

Requirements:

- Every worker operation must restore the GUI to a usable state after success or failure.
- Disable task buttons while a worker is running.
- Show a visible processing status.
- Log full technical error details.
- Show a short, user-friendly error message in the GUI.

---

## 13. Error Handling

The application must explicitly handle:

### Hardware and Platform Errors

- Microphone not detected
- Microphone is busy
- Recording device cannot be opened
- Audio stream stops unexpectedly
- Configured microphone exists on the laptop but not on Raspberry Pi
- X11 display is unavailable inside the container
- Container cannot access `/dev/snd`
- Persistent data volume is not writable
- Model package is incompatible with the target CPU architecture

### Recording Errors

- Clip is too short
- No speech is detected
- Audio is too quiet
- Audio is heavily clipped
- File cannot be written
- User cancels recording

### Enrollment Authorization Errors

- Empty password
- Incorrect password
- Maximum failed attempts reached
- Password hash is missing or invalid
- Authorization service fails
- Authorization expires before recording begins

### Enrollment Errors

- Empty speaker ID
- Invalid speaker ID format
- Speaker ID already exists
- Fewer than the required number of accepted clips
- Embedding extraction fails
- Embeddings have inconsistent dimensions
- Database save fails

### Recognition Errors

- No speakers are enrolled
- Recognition recording is invalid
- Model version mismatch
- Embedding dimension mismatch
- Similarity is below the threshold
- Recognition service crashes

### GUI Behavior

Never display a Python traceback to the user.

Show messages such as:

```text
Microphone unavailable.
Please check the selected microphone and try again.
```

Write technical details to logs.

---

## 14. Logging

Use Python's `logging` module.

Recommended format:

```text
2026-06-19 09:30:12 | INFO | audio_service | Recording started
2026-06-19 09:30:17 | INFO | audio_service | Recording completed
2026-06-19 09:30:19 | INFO | recognition_service | Best match USER_001, score=0.873
```

Recommended logical log location:

```text
logs/speaker_app.log
```

During Raspberry Pi deployment, mount `logs/` to a persistent host directory so logs survive container replacement.

Do not log raw embedding vectors unless explicitly required for debugging.

Do not log sensitive audio paths unnecessarily.

---

## 15. Configuration

All adjustable and platform-dependent values must come from typed configuration and environment variables.

Example:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    profile: str = "development"
    windowed: bool = True
    enrollment_clip_count: int = 6
    enrollment_clip_duration_seconds: float = 5.0
    recognition_clip_duration_seconds: float = 5.0
    recognition_threshold: float = 0.75
    enrollment_password_hash_env: str = "ENROLLMENT_PASSWORD_HASH"
    max_enrollment_password_attempts: int = 3
    enrollment_authorization_timeout_seconds: int = 120
    sample_rate: int = 16000
    audio_channels: int = 1
    audio_dtype: str = "int16"
    audio_device: str | None = None
    database_path: Path = Path("data/speakers.db")
    temporary_audio_dir: Path = Path("data/temporary_audio")
    enrollment_audio_dir: Path = Path("data/enrollment_audio")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
```

The recognition threshold is only an example placeholder and must be replaced with a value selected from model evaluation.

The `enrollment_password_hash_env` value is the name of an environment variable, not the hash itself. The actual hash must be provisioned outside the source repository.

### Development Profile

Typical behavior:

```text
APP_PROFILE=development
APP_WINDOWED=true
APP_AUDIO_DEVICE=<local microphone or empty>
APP_DATA_DIR=./data
APP_LOG_LEVEL=DEBUG
```

### Raspberry Pi Profile

Typical behavior:

```text
APP_PROFILE=raspberry-pi
APP_WINDOWED=false
APP_AUDIO_DEVICE=<Raspberry Pi microphone selector>
APP_DATA_DIR=/app/data
APP_MODEL_DIR=/app/models
APP_LOG_LEVEL=INFO
DISPLAY=:0
```

Never commit real passwords, hashes, tokens, or machine-specific secrets in `.env` files. Commit only `.env.example` templates.

---

## 16. Audio File Requirements

Recommended recording format:

- Container: WAV
- Encoding: PCM
- Channels: Mono
- Sample rate: Must match model requirements, typically 16 kHz
- Sample type: 16-bit signed integer

The backend must resample or reject files that do not meet model requirements.

Use temporary filenames that avoid collisions:

```text
recognition_20260619_093012.wav
USER_001_clip_01.wav
```

Recognition audio should normally be deleted after inference.

Enrollment recordings may be deleted after successful enrollment unless they are required for testing, auditing, or retraining.

---

## 17. GUI Styling Requirements

The final Raspberry Pi UI is intended for a touchscreen, while the local development UI must remain convenient on a laptop.

Shared requirements:

- Large touch targets
- Minimum button height around 60–80 pixels
- Clear status text
- High contrast
- Avoid small icons without labels
- Avoid nested menus
- Keep important actions centered
- Always provide a clear `Back`, `Cancel`, or `Return Home` action
- Prevent repeated clicks from launching duplicate tasks
- Use layouts rather than absolute coordinates

Suggested design resolution:

```text
800 × 480
```

Local development behavior:

- Start in a normal resizable window.
- Allow developer resizing and debugging.
- Support `--windowed` explicitly.

Raspberry Pi behavior:

- Start full-screen.
- Adapt to the actual display resolution.
- Hide desktop decorations where practical.
- Do not assume the laptop and Raspberry Pi use the same DPI or font scaling.

---

## 18. Application Startup

At application startup:

1. Load the selected configuration profile.
2. Configure logging.
3. Create required directories.
4. Initialize the SQLite database.
5. Initialize the enrollment authorization service and verify that a password hash is configured.
6. Detect the configured microphone.
7. Load the embedding model once.
8. Build the main window.
9. Display hardware, authorization, database, and model status.
10. Choose windowed or full-screen display from configuration.

Example entry point:

```python
import argparse
import sys
from PySide6.QtWidgets import QApplication

from speaker_app.config import load_config
from speaker_app.ui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("development", "raspberry-pi"),
        default="development",
    )
    parser.add_argument("--windowed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.profile)

    app = QApplication(sys.argv)
    window = MainWindow(config=config)

    if args.windowed or config.windowed:
        window.resize(800, 480)
        window.show()
    else:
        window.showFullScreen()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

### Local Launch

```bash
python -m speaker_app.main --profile development --windowed
```

### Raspberry Pi Container Launch

The container command should run the same entry point with a different profile:

```bash
python -m speaker_app.main --profile raspberry-pi
```

---

## 19. Development, Dockerization, and Backend Integration

### 19.1 Existing Backend Integration

The current project already has separate Python programs for:

- Recording audio
- Speaker enrollment
- Speaker recognition

The AI agent should first inspect these programs and identify:

- Current function names
- Required arguments
- Produced files
- Model-loading behavior
- Embedding shape
- Audio format
- Printed outputs
- Failure behavior
- Database or file-storage behavior

Preferred integration order:

1. Reuse existing functions where possible.
2. Refactor script entry points into reusable services.
3. Keep CLI wrappers for standalone testing.
4. Import services directly from the GUI.
5. Avoid launching subprocesses in the final architecture unless necessary.

Temporary fallback:

- Use `QProcess` to run existing scripts.
- Require machine-readable JSON output.
- Parse only JSON, never free-form terminal text.

### 19.2 Local Native Development Workflow

Required order:

```text
Create virtual environment
→ Install Python dependencies
→ Run unit tests
→ Run GUI with dummy services
→ Test local microphone recording
→ Test enrollment
→ Test recognition
→ Test database persistence
→ Fix platform-independent defects
→ Freeze dependency versions
```

Do not begin Raspberry Pi Docker debugging while the same feature is still failing in native local execution.

Local testing must cover:

- Fresh database startup
- Duplicate speaker IDs
- Correct and incorrect enrollment passwords
- Successful and failed recordings
- Unknown-speaker rejection
- Application restart with persisted speakers
- Model loading only once
- GUI responsiveness during inference

### 19.3 Docker Image Requirements

The Docker image must:

- Use a Linux base image compatible with the Raspberry Pi architecture.
- Install only required runtime libraries.
- Copy application source and dependency manifests.
- Run as a non-root user where practical.
- Avoid embedding the database, recordings, logs, passwords, or environment-specific configuration.
- Define a clear application command.
- Include a health check only when it can test a meaningful process state.
- Pin major dependency versions for reproducibility.

The image must not require a full desktop environment inside the container. Qt must connect to the host X11 server.

### 19.4 Docker Compose Host Integration

A Raspberry Pi Compose configuration will typically require settings similar to:

```yaml
services:
  speaker-app:
    build:
      context: .
    environment:
      APP_PROFILE: raspberry-pi
      DISPLAY: ${DISPLAY:-:0}
      XAUTHORITY: /run/user/host.Xauthority
    devices:
      - /dev/snd:/dev/snd
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
      - ${XAUTHORITY_FILE}:/run/user/host.Xauthority:ro
      - ./runtime/data:/app/data
      - ./runtime/logs:/app/logs
      - ./models:/app/models:ro
    restart: unless-stopped
```

This is a design template, not a guaranteed final Compose file. The actual device paths, Xauthority path, user IDs, group IDs, and permissions must be verified on the Raspberry Pi.

Security and permission rules:

- Prefer Xauthority-based access over globally disabling X11 access control.
- Do not use `xhost +`.
- Map only required devices.
- Do not use `privileged: true` by default.
- Add only required supplemental groups, such as the host audio group.
- Keep password hashes outside the image and inject them at runtime.

### 19.5 Persistent Data

The following must survive container replacement:

- SQLite speaker database
- Enrollment audio when retention is enabled
- Logs
- Optional model cache
- Application configuration that belongs to the deployment

Use bind mounts or named volumes. Treat the container filesystem as disposable.

### 19.6 Architecture and Dependency Compatibility

Before deployment, verify:

- Raspberry Pi OS is 32-bit or 64-bit.
- Docker image architecture matches the OS and CPU.
- PySide6 or the chosen Qt Python package is available for the target architecture.
- Speaker-model runtime supports the Raspberry Pi architecture.
- Native audio and numerical dependencies have compatible builds.
- Model inference latency and memory use are acceptable.

Build options:

1. Build the image directly on the Raspberry Pi.
2. Build an ARM image on the laptop using Docker Buildx.
3. Publish a multi-architecture image if both laptop-Linux and Raspberry Pi execution are needed.

Regardless of build method, run the final image on the real Raspberry Pi before declaring deployment complete.

---

## 20. Functional Acceptance Criteria

### Local Development Acceptance

Before Dockerization begins:

- The application starts natively on the laptop.
- The GUI opens in windowed mode.
- All pages and state transitions work.
- Correct authorization allows enrollment.
- Incorrect authorization blocks enrollment.
- Recording works with the configured local microphone or a test-audio adapter.
- Enrollment creates and persists a speaker profile.
- Recognition returns a speaker ID or `Unknown Speaker`.
- The interface does not freeze during recording or inference.
- Unit and integration tests pass.

### Enrollment Acceptance

Enrollment is considered complete when:

- The user enters a valid unique speaker ID.
- The correct enrollment password is accepted.
- Recording cannot begin before authorization succeeds.
- The system records the configured number of accepted clips.
- Each clip passes audio validation.
- The model extracts a valid embedding from every accepted clip.
- The system creates a normalized prototype embedding.
- The database stores the speaker profile successfully.
- The GUI displays enrollment success.

### Recognition Acceptance

Recognition is considered complete when:

- The system records a valid speech clip.
- The model extracts a valid query embedding.
- Compatible speaker profiles are available.
- Similarity is computed against all compatible profiles.
- The best match is selected.
- The recognition threshold is applied.
- The GUI displays either the speaker ID or `Unknown Speaker`.

### Raspberry Pi Container Acceptance

Deployment is considered complete when:

- The image runs on the target Raspberry Pi architecture.
- Qt renders on the host Openbox/X11 display.
- The container can record from the Raspberry Pi microphone.
- Database and logs persist after container recreation.
- The application starts full-screen with the Raspberry Pi profile.
- Container restart does not erase enrolled speakers.
- The container runs without unnecessary privileged access.
- End-to-end enrollment and recognition succeed on the real device.

### Responsiveness

- The interface must not freeze during recording or inference.
- Buttons must remain visually responsive.
- The user must see the current operation state.
- Errors must return the application to a recoverable state.

---

## 21. Non-Functional Requirements

- The application must run natively on the laptop before Dockerization.
- The application should start without internet access after dependencies and model assets are installed.
- The deployed system should work entirely on-device.
- The model should be loaded only once per application session.
- Database operations should be transactional.
- Enrollment authorization must use secure password hashing.
- Plaintext passwords must never be persisted or logged.
- The application should recover cleanly from failed recordings.
- Temporary files should be cleaned up safely.
- Code should include type hints.
- Public classes and functions should include concise docstrings.
- Avoid global mutable state.
- Use dependency injection where practical.
- Add unit tests for authorization, repository operations, and embedding comparison.
- Keep hardware-specific audio logic isolated from GUI code.
- Keep local and Raspberry Pi configuration separate without duplicating application logic.
- Docker builds should be reproducible.
- Persistent data must not depend on the container writable layer.
- The Raspberry Pi container should run with the minimum required devices and permissions.

---

## 22. Initial Implementation Milestones

### Milestone 1 — Native Local GUI Prototype

Create and run on the laptop:

- Main window
- Home page
- Enrollment information page
- Enrollment authorization page
- Enrollment recording page
- Recognition page
- Result page
- Navigation using `QStackedWidget`

Use dummy backend results.

### Milestone 2 — Native Local Audio Integration

Connect:

- Local microphone detection
- WAV recording
- Test-audio adapter
- Audio validation
- Worker-thread execution

### Milestone 3 — Native Local Enrollment Integration

Connect:

- Password authorization
- Embedding extraction
- Multi-clip prototype generation
- SQLite storage
- Enrollment status handling

### Milestone 4 — Native Local Recognition Integration

Connect:

- Query embedding extraction
- Database loading
- Cosine similarity
- Threshold decision
- Result display

At the end of Milestone 4, the complete application must run successfully on the laptop without Docker.

### Milestone 5 — Dockerization

Create:

- `Dockerfile`
- `.dockerignore`
- `compose.yaml`
- Raspberry Pi environment template
- Persistent volume mapping
- X11 integration
- Audio-device mapping
- Non-root runtime configuration where practical

### Milestone 6 — Raspberry Pi Deployment

Verify:

- ARM architecture compatibility
- Openbox/X11 display access
- USB microphone access
- Full-screen startup
- Persistent database and logs
- Container restart policy
- End-to-end enrollment and recognition
- Startup after Raspberry Pi reboot

---

## 23. AI Agent Instructions

When modifying this project, the AI agent must:

1. Read this file before making architectural changes.
2. Preserve separation between GUI and backend logic.
3. Implement and test the application natively on the laptop before requiring Docker.
4. Treat Docker as a deployment layer, not as a separate application implementation.
5. Never run long backend tasks in the Qt main thread.
6. Avoid replacing working backend algorithms without a clear reason.
7. Inspect existing code before inventing new interfaces.
8. Keep public interfaces typed and documented.
9. Return structured result objects from backend services.
10. Keep speaker-model loading centralized and reusable.
11. Preserve model-version and embedding-dimension validation.
12. Avoid hardcoding paths outside the configuration layer.
13. Avoid hardcoding microphone indices, `DISPLAY`, UID, GID, or device paths.
14. Avoid arbitrary recognition thresholds.
15. Add error handling for every hardware or file operation.
16. Keep the Raspberry Pi CLI-only and Docker deployment environment in mind.
17. Prefer incremental, testable changes.
18. Explain any database migration or incompatible change.
19. Do not expose raw exceptions directly in the GUI.
20. Do not store, print, or log the plaintext enrollment password.
21. Do not hardcode the enrollment password or password hash in committed source files.
22. Do not associate the enrollment password with individual speaker profiles.
23. Do not allow access to enrollment recording without successful authorization.
24. Do not store persistent data only inside the container writable layer.
25. Do not use a privileged container unless the need is documented and narrower alternatives fail.
26. Do not assume that successful laptop microphone access proves Raspberry Pi microphone access.
27. Do not delete enrollment audio unless the configured retention policy allows it.
28. Do not claim recognition accuracy without evaluation evidence.
29. Keep development and Raspberry Pi configuration examples free of real secrets.
30. Run final hardware-in-the-loop tests on the actual Raspberry Pi.

---

## 24. Information Still Needed from the Existing Code and Environment

Before final backend integration, inspect and document:

### Existing Python Backend

- Recording library currently used
- Microphone device name or index
- Audio sample rate
- Audio duration per clip
- VAD implementation
- Speaker embedding model name
- Embedding dimension
- Input preprocessing pipeline
- Current enrollment strategy
- Current recognition metric
- Current recognition threshold
- Current speaker-storage format
- Current model runtime: CPU, CUDA, ONNX, Torch, TensorFlow, or another provider

### Authorization

- Enrollment password provisioning method
- Password-hashing library and algorithm
- Maximum failed authorization attempts
- Authorization timeout behavior

### Laptop Development Environment

- Laptop operating system
- Python version
- Local microphone identifier
- Available inference acceleration
- Expected model path
- Whether local Docker Desktop is available for optional image building

### Raspberry Pi Deployment Environment

- Raspberry Pi model
- Raspberry Pi OS version and architecture
- Available RAM
- Display resolution
- X11 display identifier
- Xauthority file location
- Openbox startup method
- USB microphone ALSA identifier
- Host audio group ID
- Docker and Docker Compose versions
- Whether the image will be built on the Pi or cross-built on the laptop
- Expected Raspberry Pi inference latency
- Exact behavior when multiple speakers have similar scores

These values should replace placeholders in this document, environment templates, and configuration code.

---

## 25. Final Expected User and Deployment Flow

### Local Development Flow

```text
Clone project
→ Create local Python virtual environment
→ Install dependencies
→ Configure development profile
→ Run unit tests
→ Launch windowed GUI
→ Test microphone and test-audio modes
→ Test authorization
→ Test enrollment
→ Test recognition
→ Verify database persistence
→ Freeze verified dependencies
```

### Dockerization and Raspberry Pi Deployment Flow

```text
Start from locally verified application
→ Create Dockerfile and Compose configuration
→ Build ARM-compatible image
→ Copy or pull image on Raspberry Pi
→ Configure X11, Xauthority, audio device, volumes, and secrets
→ Start Openbox/X11 on host
→ Start container
→ Verify full-screen Qt display
→ Verify microphone access
→ Run end-to-end enrollment and recognition
→ Configure automatic startup after reboot
```

### Enroll Speaker

```text
Launch application
→ Home
→ Enroll Speaker
→ Enter speaker ID and optional display name
→ Continue
→ Enter enrollment password
→ Verify authorization
→ Record clip 1
→ Validate clip
→ Repeat until 6 accepted clips
→ Extract embeddings
→ Build voiceprint
→ Save speaker profile
→ Show success
→ Return Home
```

### Recognize Speaker

```text
Launch application
→ Home
→ Recognize Speaker
→ Record speech
→ Validate audio
→ Extract embedding
→ Compare with stored voiceprints
→ Apply threshold
→ Display speaker ID or Unknown Speaker
→ Recognize Again or Return Home
```
