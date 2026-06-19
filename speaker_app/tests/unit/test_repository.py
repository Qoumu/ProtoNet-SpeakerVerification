import sqlite3

import numpy as np
import pytest

from speaker_app.domain import SpeakerProfile
from speaker_app.services.speaker_repository import SpeakerRepository


def profile(speaker_id="alice", version="model-v1"):
    return SpeakerProfile(speaker_id, "Alice", np.array([1.0, 0.0]), version, 6)


def test_repository_persists_and_filters_profiles(tmp_path):
    repository = SpeakerRepository(tmp_path / "speakers.db")
    repository.initialize()
    repository.save(profile())
    repository.save(profile("bob", "model-v2"))

    assert repository.count() == 2
    loaded = repository.get("alice")
    assert loaded is not None
    np.testing.assert_array_equal(loaded.embedding, [1.0, 0.0])
    assert [item.speaker_id for item in repository.get_all()] == ["alice", "bob"]
    assert [item.speaker_id for item in repository.get_all_compatible("model-v1", 2)] == [
        "alice"
    ]


def test_repository_rejects_duplicate_by_default(tmp_path):
    repository = SpeakerRepository(tmp_path / "speakers.db")
    repository.initialize()
    repository.save(profile())
    with pytest.raises(sqlite3.IntegrityError):
        repository.save(profile())
    assert repository.delete("alice")
    assert not repository.exists("alice")

