# Speaker Recognition App — AI Agent Context

## 1. Project Overview

Build a touchscreen-friendly speaker recognition application for Raspberry Pi.

The system has two main functions:

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

The current project has those two feature, read all the current project structure and utilize it
---

## 2. Target Platform

- Device: Raspberry Pi
- Operating system: Raspberry Pi OS Lite / CLI-only environment
- Desktop environment: No full desktop environment
- Window manager: Openbox
- Display mode: Full-screen application
- Backend language: Python
- GUI framework: PySide6 using Qt Widgets
- Database: SQLite
- Audio input: USB microphone or Raspberry Pi-compatible microphone
- Execution environment: X11 started through `startx` and Openbox

Expected application launch:

```bash
startx /path/to/start-speaker-app.sh
```

Example launch script:

```bash
#!/bin/bash

openbox-session &
exec /path/to/venv/bin/python /path/to/speaker_app/main.py
```

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
app/
├── main.py
├── app_controller.py
├── config.py
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── home_page.py
│   ├── enroll_info_page.py
│   ├── enroll_recording_page.py
│   ├── enroll_processing_page.py
│   ├── recognition_page.py
│   ├── recognition_result_page.py
│   ├── message_dialog.py
│   └── styles.qss
│
├── services/
│   ├── __init__.py
│   ├── audio_service.py
│   ├── enrollment_service.py
│   ├── recognition_service.py
│   └── speaker_repository.py
│
├── model/
│   ├── __init__.py
│   └── model.pth
│
├── workers/
│   ├── __init__.py
│   └── backend_worker.py
│
├── domain/
│   ├── __init__.py
│   └── results.py
│
├── data/
│   ├── enrolled_speakers.pt
│   ├── temporary_audio/
│   └── enrollment_audio/
│
├── scripts/
│   └── start-speaker-app.sh
│
├── tests/
│   ├── test_repository.py
│   ├── test_embedding_comparison.py
│   └── test_services.py
│
├── requirements.txt
└── README.md
```

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

- `Start Enrollment`
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

### 6.3 Enrollment Authorization Page

Before the user can proceed with speaker enrollment, the application must require an authorization password.

Display:

- Password input field
- Optional password visibility toggle
- Short instruction such as `Enter the enrollment password to continue`

Buttons:

- `Confirm`
- `Back`

Example:

```text
┌────────────────────────────────────────┐
│       ENROLLMENT AUTHORIZATION         │
│                                        │
│  Speaker ID: USER_001                  │
│                                        │
│  Password: [ •••••••••••••••• ]       │
│                                        │
│          [      CONFIRM      ]         │
│                                        │
│          [        BACK       ]         │
└────────────────────────────────────────┘
```

Required behavior:

- The password field must use masked input.
- Pressing `Confirm` must validate the password before opening the recording page.
- A correct password navigates to the Enrollment Recording Page.
- An incorrect password keeps the user on the authorization page and displays a short error message.
- Clear the password field after every failed attempt.
- Do not print, log, or persist the plaintext password.
- Navigating back returns to the Enrollment Information Page without losing the entered speaker information.
- Cancelling enrollment clears the password and all pending enrollment data.
- The password authorizes access to enrollment only. It must not be stored with a speaker profile.

Security requirements:

- Never store the password as plaintext in source code, SQLite, logs, or configuration committed to version control.
- Store only a secure password hash using Argon2id, scrypt, or bcrypt.
- Verify the password in a dedicated authorization service.
- Use a configurable maximum number of failed attempts.
- After repeated failed attempts, temporarily disable confirmation or return to the Home Page.
- Use the password library's secure verification function rather than manually comparing hashes.

Recommended interface:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthorizationResult:
    accepted: bool
    remaining_attempts: int
    message: str


class EnrollmentAuthorizationService:
    def verify_password(
        self,
        password: str,
    ) -> AuthorizationResult:
        ...
```

The GUI must pass the password to this service, process the result, and discard the plaintext value immediately after verification.

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

Example:

- The user cannot access the Enrollment Recording Page until the enrollment password is accepted.
- The user cannot process enrollment before all required clips are accepted.
- The user cannot start recognition while enrollment is running.
- The user cannot navigate away during a database write unless cancellation is safely supported.
- Returning from authorization to speaker information must preserve the pending speaker ID and display name.
- Cancelling enrollment must clear all pending information, temporary password data, and temporary recordings.

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

Required interface:

```python
class EnrollmentAuthorizationService:
    def verify_password(
        self,
        password: str,
    ) -> AuthorizationResult:
        ...
```

Responsibilities:

- Validate the enrollment password before recording begins.
- Verify the password against a securely stored password hash.
- Track failed attempts for the current authorization session.
- Return the number of remaining attempts.
- Reset failed-attempt state after successful authorization.
- Never return, log, or expose the stored password hash.
- Never save the entered plaintext password.

The authorization service must remain independent of speaker enrollment data. The password must not be stored in the `speakers` table.

### 8.3 Audio Service

Required interface:

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

- Open and validate the microphone.
- Record PCM audio.
- Save audio as WAV.
- Check minimum duration.
- Check whether speech is present.
- Detect silence or extremely low energy.
- Detect clipping where practical.
- Return structured errors.

### 8.4 Embedding Extractor

Required interface:

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

### 8.5 Enrollment Service

Required interface:

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

### 8.6 Recognition Service

Required interface:

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

### Hardware Errors

- Microphone not detected
- Microphone is busy
- Recording device cannot be opened
- Audio stream stops unexpectedly

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
- Password verification service fails
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
Please check the USB microphone and try again.
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

Recommended log location:

```text
speaker_app/logs/speaker_app.log
```

Do not log raw embedding vectors unless explicitly required for debugging.

Do not log sensitive audio paths unnecessarily.

---

## 15. Configuration

Place adjustable values in `config.py` or a configuration file.

Example:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    enrollment_clip_count: int = 6
    enrollment_clip_duration_seconds: float = 5.0
    recognition_clip_duration_seconds: float = 5.0
    recognition_threshold: float = 0.5
    enrollment_password_hash_env: str = "ENROLLMENT_PASSWORD_HASH"
    max_enrollment_password_attempts: int = 3
    enrollment_authorization_timeout_seconds: int = 120
    sample_rate: int = 16000
    audio_channels: int = 1
    audio_dtype: str = "int16"
    database_path: Path = Path("data/enrolled_speakers.pt")
    temporary_audio_dir: Path = Path("data/temporary_audio")
    enrollment_audio_dir: Path = Path("data/enrollment_audio")
```

The recognition threshold above is only an example placeholder and must be replaced with the validated value from model evaluation.

`enrollment_password_hash_env` stores the environment-variable name, not the password hash itself. Provision the actual hash outside the source repository.

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

The UI is intended for a touchscreen.

Requirements:

- Full-screen main window
- Large touch targets
- Minimum button height around 60–80 pixels
- Clear status text
- High contrast
- Avoid small icons without labels
- Avoid nested menus
- Keep important actions centered
- Always provide a clear `Back`, `Cancel`, or `Return Home` action
- Prevent double-clicking from launching duplicate tasks

Suggested base window size during PC development:

```text
800 × 480
```

The final application should adapt to the actual display resolution.

Use layouts rather than absolute coordinates.

---

## 18. Application Startup

At startup:

1. Configure logging.
2. Create required directories.
3. Initialize the SQLite database.
4. Initialize the enrollment authorization service and verify that a password hash is configured.
5. Detect the microphone.
6. Load the embedding model once.
7. Build the main window.
8. Display hardware and model status.
9. Enter full-screen mode.

Example:

```python
import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.showFullScreen()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

During desktop development, support a command-line flag that runs the application in a normal window instead of full-screen mode.

---

## 19. Existing Backend Integration

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

Example subprocess result:

```json
{
  "success": true,
  "speaker_id": "USER_001",
  "similarity": 0.873,
  "message": "Speaker recognized"
}
```

---

## 20. Functional Acceptance Criteria

### Enrollment

Enrollment is considered complete when:

- The user enters a valid unique speaker ID.
- The user enters the correct enrollment authorization password.
- The system does not allow recording before authorization succeeds.
- The system records the configured number of accepted clips.
- Each clip passes audio validation.
- The model extracts a valid embedding from every accepted clip.
- The system creates a normalized prototype embedding.
- The database stores the speaker profile successfully.
- The GUI displays enrollment success.

### Recognition

Recognition is considered complete when:

- The system records a valid speech clip.
- The model extracts a valid query embedding.
- Compatible speaker profiles are available.
- Similarity is computed against all compatible profiles.
- The best match is selected.
- The recognition threshold is applied.
- The GUI displays either the speaker ID or `Unknown Speaker`.

### Responsiveness

- The interface must not freeze during recording or inference.
- Buttons must remain visually responsive.
- The user must see the current operation state.
- Errors must return the application to a recoverable state.

---

## 21. Non-Functional Requirements

- The application should start without internet access.
- The system should work entirely on-device.
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
- Add unit tests for repository operations and embedding comparison.
- Keep hardware-specific audio logic isolated from GUI code.

---

## 22. Initial Implementation Milestones

### Milestone 1 — GUI Prototype

Create:

- Main window
- Home page
- Enrollment information page
- Enrollment authorization page
- Enrollment recording page
- Recognition page
- Result page
- Navigation using `QStackedWidget`

Use dummy backend results.

### Milestone 2 — Audio Integration

Connect:

- Microphone detection
- WAV recording
- Audio validation
- Worker-thread execution

### Milestone 3 — Enrollment Integration

Connect:

- Embedding extraction
- Multi-clip prototype generation
- SQLite storage
- Enrollment status handling

### Milestone 4 — Recognition Integration

Connect:

- Query embedding extraction
- Database loading
- Cosine similarity
- Threshold decision
- Result display

### Milestone 5 — Raspberry Pi Deployment

Configure:

- Python virtual environment
- Required system packages
- Openbox
- X11
- Full-screen launch
- Automatic application startup
- Logging and crash recovery

---

## 23. AI Agent Instructions

When modifying this project, the AI agent must:

1. Read this file before making architectural changes.
2. Preserve separation between GUI and backend logic.
3. Never run long backend tasks in the Qt main thread.
4. Avoid replacing working backend algorithms without a clear reason.
5. Inspect existing code before inventing new interfaces.
6. Keep public interfaces typed and documented.
7. Return structured result objects from backend services.
8. Keep speaker-model loading centralized and reusable.
9. Preserve model-version and embedding-dimension validation.
10. Avoid hardcoding paths outside the configuration layer.
11. Avoid arbitrary recognition thresholds.
12. Add error handling for every hardware or file operation.
13. Keep the Raspberry Pi CLI-only deployment environment in mind.
14. Prefer incremental, testable changes.
15. Explain any database migration or incompatible change.
16. Do not expose raw exceptions directly in the GUI.
17. Do not store, print, or log the plaintext enrollment password.
18. Do not hardcode the enrollment password or its hash in committed source files.
19. Do not associate the enrollment password with individual speaker profiles.
20. Do not allow access to enrollment recording without successful authorization.
21. Do not delete enrollment audio unless the configured retention policy allows it.
22. Do not claim recognition accuracy without evaluation evidence.

---

## 24. Information Still Needed from the Existing Code

Before final backend integration, inspect and document:

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
- Current enrollment password provisioning method
- Password-hashing library and algorithm
- Maximum failed authorization attempts
- Enrollment authorization timeout behavior
- Whether CUDA, CPU, ONNX, Torch, or TensorFlow is used
- Expected Raspberry Pi inference latency
- Exact behavior when multiple speakers have similar scores

These values should replace placeholders in this document and in `config.py`.

---

## 25. Final Expected User Flow

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
