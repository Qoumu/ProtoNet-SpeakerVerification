from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length embedding")
    return (result / norm).astype(np.float32, copy=False)


def create_speaker_prototype(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one embedding is required")
    normalized = np.stack([l2_normalize(item) for item in embeddings])
    if len({item.shape[0] for item in normalized}) != 1:
        raise ValueError("Embedding dimensions are inconsistent")
    return l2_normalize(normalized.mean(axis=0))


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(l2_normalize(first), l2_normalize(second)))
