from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    display_name: str | None = None


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    display_name: str | None
    embedding: np.ndarray
    model_version: str
    number_of_samples: int
    created_at: str | None = None
    updated_at: str | None = None
