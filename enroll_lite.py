"""
Host-side speaker enrollment for embedding store generation.

This script generates either:
- a `.pt` enrollment store keyed by speaker ID (default)
- or a legacy `.npy` embedding matrix plus a sidecar speaker-ID text file

Dependencies: argparse, numpy, pathlib, torch (host only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


DEFAULT_NUM_SUB_PROTOTYPES = 3


def _normalize_embedding_vector(embedding: np.ndarray) -> np.ndarray:
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if emb.size == 0:
        raise ValueError("Embedding is empty")
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        raise ValueError("Embedding norm is zero")
    return emb / norm


def _run_kmeans(
    embeddings: np.ndarray,
    *,
    num_clusters: int,
    max_iters: int,
    seed: int,
) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"Expected a non-empty 2D embedding matrix, got shape {embeddings.shape}"
        )

    vectors = np.asarray(embeddings, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n_samples = vectors.shape[0]

    if n_samples == 1:
        centers = np.repeat(vectors, num_clusters, axis=0)
        return np.stack([_normalize_embedding_vector(center) for center in centers], axis=0)

    init_count = min(num_clusters, n_samples)
    center_indices = [int(rng.integers(n_samples))]
    while len(center_indices) < init_count:
        current = vectors[center_indices]
        distances = np.sum((vectors[:, None, :] - current[None, :, :]) ** 2, axis=2)
        min_distances = distances.min(axis=1)
        min_distances[center_indices] = 0.0
        if float(min_distances.sum()) <= 0.0:
            remaining = [idx for idx in range(n_samples) if idx not in center_indices]
            if not remaining:
                break
            center_indices.append(remaining[0])
            continue
        probs = min_distances / min_distances.sum()
        next_index = int(rng.choice(n_samples, p=probs))
        if next_index not in center_indices:
            center_indices.append(next_index)

    centers = vectors[center_indices].copy()
    if centers.shape[0] < num_clusters:
        pad = np.repeat(centers[-1:, :], num_clusters - centers.shape[0], axis=0)
        centers = np.concatenate([centers, pad], axis=0)

    for _ in range(max_iters):
        distances = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = distances.argmin(axis=1)
        new_centers = centers.copy()

        for cluster_idx in range(num_clusters):
            mask = assignments == cluster_idx
            if np.any(mask):
                new_centers[cluster_idx] = vectors[mask].mean(axis=0)
            else:
                farthest_index = int(distances.min(axis=1).argmax())
                new_centers[cluster_idx] = vectors[farthest_index]

        if np.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers

    return np.stack([_normalize_embedding_vector(center) for center in centers], axis=0)


def _collect_chunk_embeddings_torch(
    audio_files: Iterable[Path],
    *,
    model_path: Path,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    chunk_duration: Optional[float],
    chunk_overlap: float,
    verbose: bool,
) -> np.ndarray:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for host-side enrollment")

    from utils.model_functions import load_model
    import torch.nn.functional as F
    import torchaudio.functional as AF
    import torchaudio.transforms as AT

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    model = load_model(
        model_path,
        device=device,
        n_mels=n_mels,
        channels=512,
    )

    embeddings: list[np.ndarray] = []
    chunk_dur = chunk_duration or duration
    chunk_size = max(1, int(chunk_dur * sr))
    overlap_size = max(0, int(chunk_overlap * sr))
    step_size = max(1, chunk_size - overlap_size)
    target_frames = int(duration * sr / hop_length)
    mel_transform = AT.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        f_min=0.0,
        f_max=sr / 2.0,
    )
    db_transform = AT.AmplitudeToDB(stype="power")

    for audio_path in audio_files:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if verbose:
            print(f"  Processing: {audio_path.name}")

        waveform, sr_loaded = sf.read(str(audio_path), dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Failed to load a valid mono waveform from {audio_path}")

        waveform_tensor = torch.from_numpy(waveform)
        if int(sr_loaded) != sr:
            waveform_tensor = AF.resample(
                waveform_tensor,
                orig_freq=int(sr_loaded),
                new_freq=sr,
            )

        chunk_count = 0
        for start in range(0, int(waveform_tensor.shape[0]), step_size):
            end = start + chunk_size
            chunk = waveform_tensor[start:end]
            if chunk.shape[0] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[0]))

            mel = db_transform(mel_transform(chunk.float()))
            current_frames = int(mel.shape[1])
            if current_frames > target_frames:
                offset = (current_frames - target_frames) // 2
                mel = mel[:, offset : offset + target_frames]
            elif current_frames < target_frames:
                mel = F.pad(
                    mel,
                    (0, target_frames - current_frames),
                    value=float(torch.min(mel).item()),
                )
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

            with torch.no_grad():
                emb = model(mel.unsqueeze(0).to(device)).squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(_normalize_embedding_vector(emb))
            chunk_count += 1
            if end >= int(waveform_tensor.shape[0]):
                break

        if chunk_count == 0:
            if verbose:
                print(f"    Warning: No usable chunks from {audio_path.name}")
            continue

    if not embeddings:
        raise ValueError("No usable embeddings were generated from the supplied audio files.")

    return np.stack(embeddings, axis=0)


def enroll_speaker_torch(
    speaker_id: str,
    audio_files: Iterable[Path],
    *,
    model_path: Path = Path("ECAPATDNN_protonet_model.pth"),
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 512,
    hop_length: int = 256,
    chunk_duration: Optional[float] = None,
    chunk_overlap: float = 0.5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a normalized speaker embedding using the PyTorch model.

    Args:
        speaker_id: Unique speaker identifier
        audio_files: Iterable of audio file paths
        model_path: Path to pretrained model .pth
        ... (standard audio parameters)
        verbose: Print debug info

    Returns:
        Speaker embedding as float32 numpy array
    """
    embeddings = _collect_chunk_embeddings_torch(
        audio_files=audio_files,
        model_path=model_path,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        chunk_duration=chunk_duration,
        chunk_overlap=chunk_overlap,
        verbose=verbose,
    )
    return _normalize_embedding_vector(embeddings.mean(axis=0))


def build_enrollment_profile_torch(
    speaker_id: str,
    audio_files: Iterable[Path],
    *,
    model_path: Path = Path("ECAPATDNN_protonet_model.pth"),
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 512,
    hop_length: int = 256,
    chunk_duration: Optional[float] = None,
    chunk_overlap: float = 0.5,
    num_sub_prototypes: int = DEFAULT_NUM_SUB_PROTOTYPES,
    kmeans_max_iters: int = 50,
    kmeans_seed: int = 0,
    verbose: bool = False,
) -> dict[str, object]:
    chunk_embeddings = _collect_chunk_embeddings_torch(
        audio_files=audio_files,
        model_path=model_path,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        chunk_duration=chunk_duration,
        chunk_overlap=chunk_overlap,
        verbose=verbose,
    )

    primary_embedding = _normalize_embedding_vector(chunk_embeddings.mean(axis=0))
    sub_prototypes = _run_kmeans(
        chunk_embeddings,
        num_clusters=max(1, num_sub_prototypes),
        max_iters=max(1, kmeans_max_iters),
        seed=kmeans_seed,
    )

    if verbose:
        print(
            f"  Built enrollment profile for '{speaker_id}' "
            f"from {chunk_embeddings.shape[0]} chunk embedding(s)"
        )
        print(f"  Saved {sub_prototypes.shape[0]} sub-prototype(s)")

    return {
        "embedding": primary_embedding,
        "sub_prototypes": sub_prototypes.astype(np.float32, copy=False),
        "num_source_embeddings": int(chunk_embeddings.shape[0]),
        "num_sub_prototypes": int(sub_prototypes.shape[0]),
    }


def _default_speaker_ids_path(store_path: Path) -> Path:
    return store_path.with_name(f"{store_path.stem}_ids.txt")


def _resolve_store_path(store_path: Path) -> Path:
    store_path = Path(store_path)
    if store_path.suffix == "":
        return store_path.with_suffix(".pt")
    if store_path.suffix not in {".pt", ".npy"}:
        raise ValueError("--store-path must point to a .pt or .npy file")
    return store_path


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    return _normalize_embedding_vector(embedding)


def _embedding_tensor_from_payload(payload: object, *, store_path: Path) -> "torch.Tensor":
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for .pt enrollment stores")

    if isinstance(payload, dict):
        if "embedding" not in payload:
            raise ValueError(f"Speaker entry in {store_path} is missing an 'embedding' field")
        payload = payload["embedding"]

    if hasattr(payload, "detach"):
        tensor = payload.detach().cpu()
    elif isinstance(payload, np.ndarray):
        tensor = torch.from_numpy(payload)
    elif isinstance(payload, (list, tuple)):
        tensor = torch.as_tensor(payload)
    else:
        raise ValueError(
            f"Unsupported speaker entry type {type(payload).__name__} in {store_path}"
        )

    tensor = tensor.float().reshape(-1)
    if tensor.numel() == 0:
        raise ValueError(f"Speaker entry in {store_path} is empty")

    norm = float(torch.linalg.vector_norm(tensor).item())
    if norm == 0.0:
        raise ValueError(f"Speaker entry in {store_path} has zero norm")

    return tensor / norm


def _subprototype_tensors_from_payload(payload: object, *, store_path: Path) -> list["torch.Tensor"]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for .pt enrollment stores")

    if not isinstance(payload, dict) or "sub_prototypes" not in payload:
        return []

    sub_prototypes = payload["sub_prototypes"]
    if hasattr(sub_prototypes, "detach"):
        tensor = sub_prototypes.detach().cpu().float()
    elif isinstance(sub_prototypes, np.ndarray):
        tensor = torch.from_numpy(np.asarray(sub_prototypes, dtype=np.float32))
    elif isinstance(sub_prototypes, (list, tuple)):
        tensor = torch.as_tensor(sub_prototypes, dtype=torch.float32)
    else:
        raise ValueError(
            f"Unsupported sub_prototypes type {type(sub_prototypes).__name__} in {store_path}"
        )

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] == 0:
        raise ValueError(
            f"Expected sub_prototypes to have shape [N, D] in {store_path}, got {tuple(tensor.shape)}"
        )

    normalized: list[torch.Tensor] = []
    for row in tensor:
        normalized.append(_embedding_tensor_from_payload(row, store_path=store_path))
    return normalized


def load_enrollment_pt_store(store_path: Path) -> dict[str, "torch.Tensor"]:
    """Load a `.pt` speaker store keyed by speaker ID."""
    store_path = Path(store_path)
    if store_path.suffix != ".pt":
        raise ValueError("Expected a .pt enrollment store")
    if not store_path.exists():
        return {}
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for .pt enrollment stores")

    payload = torch.load(store_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {store_path}, got {type(payload).__name__}")

    store: dict[str, torch.Tensor] = {}
    embedding_dim: Optional[int] = None
    for speaker_id, value in payload.items():
        if not isinstance(speaker_id, str) or not speaker_id or "\n" in speaker_id:
            raise ValueError(f"Invalid speaker ID in {store_path}: {speaker_id!r}")

        embedding = _embedding_tensor_from_payload(value, store_path=store_path)
        if embedding_dim is None:
            embedding_dim = int(embedding.numel())
        elif embedding.numel() != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch in {store_path}: expected {embedding_dim}, "
                f"got {embedding.numel()} for speaker '{speaker_id}'"
            )
        store[speaker_id] = embedding

    return store


def load_enrollment_pt_store_entries(store_path: Path) -> dict[str, dict[str, object]]:
    """Load a `.pt` speaker store with optional per-speaker sub-prototypes."""
    store_path = Path(store_path)
    if store_path.suffix != ".pt":
        raise ValueError("Expected a .pt enrollment store")
    if not store_path.exists():
        return {}
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for .pt enrollment stores")

    payload = torch.load(store_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {store_path}, got {type(payload).__name__}")

    store: dict[str, dict[str, object]] = {}
    embedding_dim: Optional[int] = None
    for speaker_id, value in payload.items():
        if not isinstance(speaker_id, str) or not speaker_id or "\n" in speaker_id:
            raise ValueError(f"Invalid speaker ID in {store_path}: {speaker_id!r}")

        embedding = _embedding_tensor_from_payload(value, store_path=store_path)
        sub_prototypes = _subprototype_tensors_from_payload(value, store_path=store_path)
        if embedding_dim is None:
            embedding_dim = int(embedding.numel())
        elif embedding.numel() != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch in {store_path}: expected {embedding_dim}, "
                f"got {embedding.numel()} for speaker '{speaker_id}'"
            )

        for sub_idx, sub_prototype in enumerate(sub_prototypes):
            if sub_prototype.numel() != embedding_dim:
                raise ValueError(
                    f"Sub-prototype dimension mismatch in {store_path}: expected {embedding_dim}, "
                    f"got {sub_prototype.numel()} for speaker '{speaker_id}' entry {sub_idx}"
                )

        store[speaker_id] = {
            "embedding": embedding,
            "sub_prototypes": sub_prototypes,
        }

    return store


def load_enrollment_npy_store(
    store_path: Path,
    speaker_ids_path: Optional[Path] = None,
) -> tuple[list[str], np.ndarray]:
    """Load the speaker embedding matrix and its row-order speaker IDs."""
    store_path = Path(store_path)
    speaker_ids_path = speaker_ids_path or _default_speaker_ids_path(store_path)

    if not store_path.exists():
        return [], np.empty((0, 0), dtype=np.float32)

    embeddings = np.load(store_path)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embedding matrix in {store_path}, got shape {embeddings.shape}"
        )
    embeddings = embeddings.astype(np.float32, copy=False)

    if not speaker_ids_path.exists():
        raise FileNotFoundError(f"Speaker ID sidecar not found: {speaker_ids_path}")

    speaker_ids = [
        line.strip()
        for line in speaker_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(set(speaker_ids)) != len(speaker_ids):
        raise ValueError(f"Duplicate speaker IDs found in {speaker_ids_path}")
    if len(speaker_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Speaker ID count ({len(speaker_ids)}) does not match embedding rows "
            f"({embeddings.shape[0]})"
        )

    return speaker_ids, embeddings


def save_enrollment_pt(
    speaker_id: str,
    embedding: np.ndarray,
    store_path: Path,
    *,
    sub_prototypes: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> None:
    """Save or update speaker embeddings in a `.pt` store keyed by speaker ID."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for .pt enrollment stores")

    store_path = Path(store_path)
    if store_path.suffix != ".pt":
        raise ValueError("--store-path must point to a .pt file")
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if not speaker_id or "\n" in speaker_id:
        raise ValueError("speaker_id must be non-empty and must not contain newlines")

    normalized = _normalize_embedding(embedding)
    store = load_enrollment_pt_store(store_path)
    action = "Updated" if speaker_id in store else "Created"
    if store and speaker_id not in store:
        action = "Appended"

    entry: torch.Tensor | dict[str, object]
    if sub_prototypes is None:
        entry = torch.from_numpy(normalized)
    else:
        subprototype_array = np.asarray(sub_prototypes, dtype=np.float32)
        if subprototype_array.ndim == 1:
            subprototype_array = subprototype_array.reshape(1, -1)
        if subprototype_array.ndim != 2:
            raise ValueError(
                f"sub_prototypes must be 1D or 2D, got shape {subprototype_array.shape}"
            )
        if subprototype_array.shape[1] != normalized.shape[0]:
            raise ValueError(
                f"Sub-prototype dimension mismatch: embedding has {normalized.shape[0]}, "
                f"sub_prototypes have {subprototype_array.shape[1]}"
            )

        normalized_subprototypes = np.stack(
            [_normalize_embedding_vector(row) for row in subprototype_array],
            axis=0,
        )
        entry = {
            "embedding": torch.from_numpy(normalized),
            "sub_prototypes": torch.from_numpy(normalized_subprototypes),
            "num_sub_prototypes": int(normalized_subprototypes.shape[0]),
        }

    store[speaker_id] = entry
    torch.save(store, store_path)

    if verbose:
        print(f"{action} enrollment for '{speaker_id}':")
        print(f"  Speaker store: {store_path} ({len(store)} speaker(s), format=pt)")
        if sub_prototypes is not None:
            print(f"  Sub-prototypes: {np.asarray(sub_prototypes).shape[0]}")


def save_enrollment_npy(
    speaker_id: str,
    embedding: np.ndarray,
    store_path: Path,
    speaker_ids_path: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Save or update speaker embeddings in a legacy `.npy` matrix.

    Args:
        speaker_id: Speaker identifier
        embedding: Embedding vector (1D float32)
        store_path: Path to the .npy embedding matrix
        speaker_ids_path: Optional sidecar text file for row-order speaker IDs
        verbose: Print debug info
    """
    store_path = Path(store_path)
    if store_path.suffix != ".npy":
        raise ValueError("--store-path must point to a .npy file")

    speaker_ids_path = speaker_ids_path or _default_speaker_ids_path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if not speaker_id or "\n" in speaker_id:
        raise ValueError("speaker_id must be non-empty and must not contain newlines")

    embedding = _normalize_embedding(embedding)
    speaker_ids, embeddings = load_enrollment_npy_store(store_path, speaker_ids_path)

    if embeddings.size == 0:
        embeddings = embedding.reshape(1, -1)
        speaker_ids = [speaker_id]
        action = "Created"
    else:
        if embeddings.shape[1] != embedding.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: store has {embeddings.shape[1]}, "
                f"new embedding has {embedding.shape[0]}"
            )
        if speaker_id in speaker_ids:
            index = speaker_ids.index(speaker_id)
            embeddings[index] = embedding
            action = "Updated"
        else:
            embeddings = np.vstack([embeddings, embedding.reshape(1, -1)])
            speaker_ids.append(speaker_id)
            action = "Appended"

    np.save(store_path, embeddings.astype(np.float32, copy=False))
    speaker_ids_path.write_text("\n".join(speaker_ids) + "\n", encoding="utf-8")

    if verbose:
        print(f"{action} enrollment for '{speaker_id}':")
        print(f"  Embeddings: {store_path} (shape={embeddings.shape}, dtype=float32)")
        print(f"  Speaker IDs: {speaker_ids_path}")


def save_enrollment(
    speaker_id: str,
    embedding: np.ndarray,
    store_path: Path = Path("enrolled_speakers.pt"),
    speaker_ids_path: Optional[Path] = None,
    sub_prototypes: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> None:
    """Save or update speaker embeddings in a `.pt` or legacy `.npy` store."""
    store_path = _resolve_store_path(store_path)

    if store_path.suffix == ".pt":
        if speaker_ids_path is not None:
            raise ValueError("--speaker-ids-path is only supported with .npy stores")
        save_enrollment_pt(
            speaker_id=speaker_id,
            embedding=embedding,
            store_path=store_path,
            sub_prototypes=sub_prototypes,
            verbose=verbose,
        )
        return

    if sub_prototypes is not None and verbose:
        print("Warning: legacy .npy enrollment stores do not persist sub-prototypes; saving primary only.")

    save_enrollment_npy(
        speaker_id=speaker_id,
        embedding=embedding,
        store_path=store_path,
        speaker_ids_path=speaker_ids_path,
        verbose=verbose,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll speaker audio into a `.pt` or legacy `.npy` embedding store."
    )
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="Unique speaker identifier"
    )
    parser.add_argument(
        "--audio-files",
        required=True,
        nargs="+",
        type=Path,
        help="Audio files for enrollment (space-separated)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ECAPATDNN_protonet_model.pth"),
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("enrolled_speakers.pt"),
        help="Path to save the enrolled-speaker store (.pt default, legacy .npy also supported)"
    )
    parser.add_argument(
        "--speaker-ids-path",
        type=Path,
        default=None,
        help="Optional path for row-order speaker IDs when --store-path is a .npy file"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate"
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel bins"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in seconds"
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk duration in seconds (default: use target duration)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.5,
        help="Chunk overlap in seconds"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--num-sub-prototypes",
        type=int,
        default=DEFAULT_NUM_SUB_PROTOTYPES,
        help="Number of k-means sub-prototypes to save for each speaker in a .pt store."
    )
    parser.add_argument(
        "--kmeans-max-iters",
        type=int,
        default=50,
        help="Maximum number of k-means iterations for sub-prototype clustering."
    )
    parser.add_argument(
        "--kmeans-seed",
        type=int,
        default=0,
        help="Random seed for k-means sub-prototype initialization."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"Enrolling speaker: {args.speaker_id}")
        print(f"Audio files: {len(args.audio_files)}")

    try:
        profile = build_enrollment_profile_torch(
            speaker_id=args.speaker_id,
            audio_files=args.audio_files,
            model_path=args.model_path,
            sr=args.sr,
            n_mels=args.n_mels,
            duration=args.duration,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            chunk_duration=args.chunk_duration,
            chunk_overlap=args.chunk_overlap,
            num_sub_prototypes=args.num_sub_prototypes,
            kmeans_max_iters=args.kmeans_max_iters,
            kmeans_seed=args.kmeans_seed,
            verbose=args.verbose,
        )

        save_enrollment(
            speaker_id=args.speaker_id,
            embedding=np.asarray(profile["embedding"], dtype=np.float32),
            store_path=args.store_path,
            speaker_ids_path=args.speaker_ids_path,
            sub_prototypes=np.asarray(profile["sub_prototypes"], dtype=np.float32),
            verbose=args.verbose,
        )

        print(f"Successfully enrolled speaker '{args.speaker_id}'")

    except Exception as e:
        print(f"Enrollment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
