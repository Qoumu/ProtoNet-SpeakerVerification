"""Domain models shared by the application layers."""

from .results import (
    AudioValidationResult,
    AuthorizationResult,
    EnrollmentResult,
    RecognitionResult,
    RecordingResult,
    SpeakerProfile,
)

__all__ = [
    "AudioValidationResult",
    "AuthorizationResult",
    "EnrollmentResult",
    "RecognitionResult",
    "RecordingResult",
    "SpeakerProfile",
]
