from __future__ import annotations

import logging
from pathlib import Path

from speaker_app.domain import RecognitionResult
from speaker_app.services.embedding_math import cosine_similarity
from speaker_app.services.enrollment_service import EmbeddingExtractorProtocol
from speaker_app.services.speaker_repository import SpeakerRepository


LOGGER = logging.getLogger(__name__)


class RecognitionService:
    def __init__(
        self,
        repository: SpeakerRepository,
        extractor: EmbeddingExtractorProtocol,
        threshold: float,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Recognition threshold must be between -1 and 1")
        self.repository = repository
        self.extractor = extractor
        self.threshold = threshold

    def recognize(self, audio_path: Path) -> RecognitionResult:
        LOGGER.info("Recognition comparison started")
        profiles = self.repository.get_all_compatible(
            self.extractor.model_version, self.extractor.embedding_dimension
        )
        if not profiles:
            LOGGER.warning("Recognition unavailable: no compatible enrolled profiles")
            return RecognitionResult(False, None, -1.0, "No compatible speakers are enrolled")

        query = self.extractor.extract(audio_path)
        LOGGER.debug("Query embedding extracted; comparing %d profiles", len(profiles))
        best_profile = None
        best_score = -1.0
        for profile in profiles:
            score = cosine_similarity(query, profile.embedding)
            LOGGER.debug("Similarity candidate: speaker_id=%s score=%.4f", profile.speaker_id, score)
            if score > best_score:
                best_profile = profile
                best_score = score

        if best_profile is None or best_score < self.threshold:
            LOGGER.info("No speaker accepted; best score=%.4f", best_score)
            return RecognitionResult(False, None, best_score, "Unknown speaker")
        LOGGER.info("Best match %s, score=%.4f", best_profile.speaker_id, best_score)
        return RecognitionResult(
            True,
            best_profile.speaker_id,
            best_score,
            "Speaker recognized",
            best_profile.display_name,
        )
