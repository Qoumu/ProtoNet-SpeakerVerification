import numpy as np
import pytest

from speaker_app.services.embedding_math import (
    cosine_similarity,
    create_speaker_prototype,
    l2_normalize,
)


def test_prototype_and_similarity_are_normalized():
    prototype = create_speaker_prototype(
        [np.array([2.0, 0.0]), np.array([1.0, 1.0])]
    )
    assert np.linalg.norm(prototype) == pytest.approx(1.0)
    assert cosine_similarity(prototype, prototype) == pytest.approx(1.0)


def test_zero_embedding_is_rejected():
    with pytest.raises(ValueError, match="zero-length"):
        l2_normalize(np.zeros(3, dtype=np.float32))
