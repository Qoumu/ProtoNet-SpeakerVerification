from pathlib import Path

import numpy as np

from speaker_app.services.enrollment_service import EnrollmentService
from speaker_app.services.recognition_service import RecognitionService
from speaker_app.services.speaker_repository import SpeakerRepository


class FakeExtractor:
    model_version = "fake-v1"
    embedding_dimension = 3

    def __init__(self, vectors):
        self.vectors = vectors

    def extract(self, audio_path: Path) -> np.ndarray:
        return np.asarray(self.vectors[audio_path.name], dtype=np.float32)


def test_enroll_recognize_and_reject_unknown(tmp_path):
    repository = SpeakerRepository(tmp_path / "speakers.db")
    repository.initialize()
    vectors = {
        "a.wav": [1.0, 0.0, 0.0],
        "b.wav": [0.9, 0.1, 0.0],
        "known.wav": [1.0, 0.0, 0.0],
        "unknown.wav": [0.0, 0.0, 1.0],
    }
    extractor = FakeExtractor(vectors)
    enrollment = EnrollmentService(repository, extractor, required_clip_count=2)
    recognition = RecognitionService(repository, extractor, threshold=0.75)

    result = enrollment.enroll("alice", "Alice", [Path("a.wav"), Path("b.wav")])
    assert result.success
    assert recognition.recognize(Path("known.wav")).speaker_id == "alice"
    assert not recognition.recognize(Path("unknown.wav")).accepted


def test_duplicate_and_invalid_speaker_ids_are_rejected(tmp_path):
    repository = SpeakerRepository(tmp_path / "speakers.db")
    repository.initialize()
    extractor = FakeExtractor({"a.wav": [1, 0, 0]})
    enrollment = EnrollmentService(repository, extractor, required_clip_count=1)

    assert not enrollment.enroll("bad id", None, [Path("a.wav")]).success
    assert enrollment.enroll("alice", None, [Path("a.wav")]).success
    assert not enrollment.enroll("alice", None, [Path("a.wav")]).success
