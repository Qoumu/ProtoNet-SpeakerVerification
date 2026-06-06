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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for host-side enrollment")

    import torch.nn.functional as F
    from utils.data_preprocessing import audio_chunking, audio_to_mel_spectrogram
    from utils.model_functions import load_model

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    model = load_model(
        model_path,
        device=device,
        n_mels=n_mels,
        emb_dim=64,
        channels=512,
    )

    embeddings: list[torch.Tensor] = []
    chunk_dur = chunk_duration or duration

    for audio_path in audio_files:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if verbose:
            print(f"  Processing: {audio_path.name}")

        _, y, sr_loaded = audio_to_mel_spectrogram(
            audio_path=audio_path,
            sr=sr,
            duration=None,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            to_db=True,
        )

        mel_chunks = audio_chunking(
            y,
            sr_loaded,
            chunk_duration=chunk_dur,
            overlap_duration=chunk_overlap,
            return_mels=True,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            target_duration=duration,
        )

        if not mel_chunks:
            if verbose:
                print(f"    Warning: No usable chunks from {audio_path.name}")
            continue

        for mel in mel_chunks:
            with torch.no_grad():
                emb = model(mel.to(device)).squeeze(0).cpu()
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(f"No usable embeddings for speaker {speaker_id}")

    stacked = torch.stack(embeddings, dim=0)
    speaker_embedding = F.normalize(stacked.mean(dim=0), p=2, dim=0).cpu()
    return speaker_embedding.numpy().astype(np.float32)


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
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if emb.size == 0:
        raise ValueError("Embedding is empty")
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        raise ValueError("Embedding norm is zero")
    return emb / norm


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

    store[speaker_id] = torch.from_numpy(normalized)
    torch.save(store, store_path)

    if verbose:
        print(f"{action} enrollment for '{speaker_id}':")
        print(f"  Speaker store: {store_path} ({len(store)} speaker(s), format=pt)")


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
            verbose=verbose,
        )
        return

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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print(f"Enrolling speaker: {args.speaker_id}")
        print(f"Audio files: {len(args.audio_files)}")

    try:
        embedding = enroll_speaker_torch(
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
            verbose=args.verbose,
        )

        save_enrollment(
            speaker_id=args.speaker_id,
            embedding=embedding,
            store_path=args.store_path,
            speaker_ids_path=args.speaker_ids_path,
            verbose=args.verbose,
        )

        print(f"Successfully enrolled speaker '{args.speaker_id}'")

    except Exception as e:
        print(f"Enrollment failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
