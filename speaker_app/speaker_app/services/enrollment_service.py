from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from speaker_app.domain import EnrollmentResult, SpeakerProfile
from speaker_app.services.embedding_math import create_speaker_prototype
from speaker_app.services.speaker_repository import SpeakerRepository


LOGGER = logging.getLogger(__name__)


SPEAKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class EmbeddingExtractorProtocol(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

    def extract(self, audio_path: Path) -> np.ndarray: ...


class EnrollmentService:
    def __init__(
        self,
        repository: SpeakerRepository,
        extractor: EmbeddingExtractorProtocol,
        required_clip_count: int,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.required_clip_count = required_clip_count

    @staticmethod
    def validate_speaker_id(speaker_id: str) -> str | None:
        if not speaker_id:
            return "Speaker ID is required"
        if not SPEAKER_ID_PATTERN.fullmatch(speaker_id):
            return "Use only letters, numbers, underscores, and hyphens"
        return None

    def enroll(
        self, speaker_id: str, display_name: str | None, audio_paths: list[Path]
    ) -> EnrollmentResult:
        speaker_id = speaker_id.strip()
        display_name = display_name.strip() if display_name and display_name.strip() else None
        LOGGER.info("Building enrollment profile (speaker_id=%s, clips=%d)", speaker_id, len(audio_paths))
        validation_error = self.validate_speaker_id(speaker_id)
        if validation_error:
            LOGGER.warning("Enrollment rejected: %s", validation_error)
            return EnrollmentResult(False, speaker_id, 0, validation_error)
        if self.repository.exists(speaker_id):
            LOGGER.warning("Enrollment rejected: duplicate speaker_id=%s", speaker_id)
            return EnrollmentResult(False, speaker_id, 0, "Speaker ID already exists")
        if len(audio_paths) != self.required_clip_count:
            LOGGER.warning(
                "Enrollment rejected: expected %d clips, received %d",
                self.required_clip_count,
                len(audio_paths),
            )
            return EnrollmentResult(
                False,
                speaker_id,
                len(audio_paths),
                f"Exactly {self.required_clip_count} accepted clips are required",
            )

        embeddings = []
        for index, path in enumerate(audio_paths, start=1):
            LOGGER.debug("Extracting enrollment embedding %d/%d", index, len(audio_paths))
            embeddings.append(self.extractor.extract(path))
        prototype = create_speaker_prototype(embeddings)
        LOGGER.debug("Enrollment prototype created (dimension=%d)", prototype.size)
        if prototype.size != self.extractor.embedding_dimension:
            raise ValueError(
                f"Model returned {prototype.size} values; expected {self.extractor.embedding_dimension}"
            )
        self.repository.save(
            SpeakerProfile(
                speaker_id=speaker_id,
                display_name=display_name,
                embedding=prototype,
                model_version=self.extractor.model_version,
                number_of_samples=len(audio_paths),
            )
        )
        LOGGER.info("Enrollment profile stored (speaker_id=%s)", speaker_id)
        return EnrollmentResult(
            True, speaker_id, len(audio_paths), "Speaker enrolled successfully"
        )
