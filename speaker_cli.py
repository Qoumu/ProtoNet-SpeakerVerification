from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from enroll_lite import load_enrollment_pt_store, save_enrollment_pt
from model.ECAPATDNN import ECAPATDNNBackbone


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "ECAPATDNN_DataAug_protonet_model.pth"
DEFAULT_STORE_PATH = PROJECT_ROOT / "output" / "enrolled_speakers.pt"
DEFAULT_THRESHOLD = 0.70


def _positive_path(path_value: Path, *, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x_flat = np.asarray(x, dtype=np.float32).reshape(-1)
    y_flat = np.asarray(y, dtype=np.float32).reshape(-1)
    denom = max(float(np.linalg.norm(x_flat) * np.linalg.norm(y_flat)), eps)
    return float(np.dot(x_flat, y_flat) / denom)


def _build_embedding(
    audio_files: Sequence[Path],
    *,
    model_path: Path,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    chunk_duration: float | None,
    chunk_overlap: float,
    verbose: bool,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device=device, n_mels=n_mels)
    embeddings: list[torch.Tensor] = []
    effective_chunk_duration = chunk_duration or duration

    for audio_path in audio_files:
        waveform = _load_waveform(audio_path, target_sr=sr)
        mel_chunks = _waveform_to_mel_chunks(
            waveform,
            sr=sr,
            n_mels=n_mels,
            duration=duration,
            n_fft=n_fft,
            hop_length=hop_length,
            chunk_duration=effective_chunk_duration,
            chunk_overlap=chunk_overlap,
        )
        if verbose:
            print(f"Processed {audio_path} into {len(mel_chunks)} chunk(s)")
        for mel_chunk in mel_chunks:
            with torch.no_grad():
                embedding = model(mel_chunk.to(device)).squeeze(0).cpu()
            embeddings.append(embedding)

    if not embeddings:
        raise ValueError("No usable embeddings were generated from the provided audio.")

    speaker_embedding = F.normalize(torch.stack(embeddings, dim=0).mean(dim=0), p=2, dim=0).cpu()
    return speaker_embedding.numpy().astype(np.float32)


def _load_model(
    model_path: Path,
    *,
    device: torch.device,
    n_mels: int,
) -> ECAPATDNNBackbone:
    model = ECAPATDNNBackbone(n_mels=n_mels, channels=512, emb_dim=64)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_waveform(audio_path: Path, *, target_sr: int) -> torch.Tensor:
    waveform_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform_np, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"Unsupported or empty audio file: {audio_path}")

    waveform_tensor = torch.from_numpy(waveform)
    if sample_rate != target_sr:
        waveform_tensor = torchaudio.functional.resample(waveform_tensor, sample_rate, target_sr)
    return waveform_tensor


def _pad_or_crop_mel(mel: torch.Tensor, *, target_frames: int) -> torch.Tensor:
    current_frames = mel.shape[-1]
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif current_frames < target_frames:
        pad_width = target_frames - current_frames
        mel = F.pad(mel, (0, pad_width), value=float(mel.min().item()))
    return mel


def _waveform_to_mel_chunks(
    waveform: torch.Tensor,
    *,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    chunk_duration: float,
    chunk_overlap: float,
) -> list[torch.Tensor]:
    if waveform.numel() == 0:
        return []

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    chunk_size = max(1, int(chunk_duration * sr))
    overlap_size = max(0, int(chunk_overlap * sr))
    step_size = max(1, chunk_size - overlap_size)
    target_frames = int(duration * sr / hop_length)

    mel_chunks: list[torch.Tensor] = []
    for start in range(0, waveform.shape[0], step_size):
        end = start + chunk_size
        chunk = waveform[start:end]
        if chunk.shape[0] < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - chunk.shape[0]))
        mel = mel_transform(chunk)
        mel = db_transform(mel)
        mel = _pad_or_crop_mel(mel, target_frames=target_frames)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mel_chunks.append(mel.unsqueeze(0))
        if end >= waveform.shape[0]:
            break

    return mel_chunks


def _add_shared_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the .pth model.")
    parser.add_argument("--store-path", type=Path, default=DEFAULT_STORE_PATH, help="Path to the enrolled speaker .pt store.")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sample rate.")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel bin count.")
    parser.add_argument("--duration", type=float, default=3.0, help="Target duration in seconds per chunk.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size.")
    parser.add_argument("--hop-length", type=int, default=256, help="Hop length.")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk duration in seconds. Defaults to --duration.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.5,
        help="Chunk overlap in seconds.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    if not normalized_argv or normalized_argv[0].startswith("-"):
        normalized_argv = ["verify", *normalized_argv]

    parser = argparse.ArgumentParser(
        description="Container-oriented speaker enrollment and verification CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a query voice against the enrolled speaker store.",
    )
    verify_parser.add_argument(
        "--input-audio",
        required=True,
        nargs="+",
        type=Path,
        help="One or more query audio files.",
    )
    verify_parser.add_argument(
        "--speaker-id",
        default=None,
        help="Optional claimed speaker ID. When set, verification is evaluated against that specific speaker.",
    )
    verify_parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity required for acceptance.",
    )
    verify_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top matches to print.",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    _add_shared_audio_args(verify_parser)

    enroll_parser = subparsers.add_parser(
        "enroll",
        help="Enroll or update a speaker in the .pt store.",
    )
    enroll_parser.add_argument("--speaker-id", required=True, help="Speaker identifier to create or update.")
    enroll_parser.add_argument(
        "--audio-files",
        required=True,
        nargs="+",
        type=Path,
        help="Enrollment audio files for the speaker.",
    )
    _add_shared_audio_args(enroll_parser)

    return parser.parse_args(normalized_argv)


def _verify(args: argparse.Namespace) -> int:
    model_path = _positive_path(args.model_path, label="Model")
    store_path = _positive_path(args.store_path, label="Enrollment store")
    audio_files = [_positive_path(path, label="Input audio") for path in args.input_audio]

    store = load_enrollment_pt_store(store_path)
    if not store:
        raise ValueError(f"Enrollment store is empty: {store_path}")

    if args.speaker_id is not None and args.speaker_id not in store:
        available = ", ".join(sorted(store.keys())[:10])
        raise ValueError(
            f"Speaker '{args.speaker_id}' not found in {store_path}. Available speakers: {available}"
        )

    query_embedding = _build_embedding(
        audio_files=audio_files,
        model_path=model_path,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        verbose=args.verbose,
    )

    scored_matches = []
    for speaker_id, embedding in store.items():
        similarity = _cosine_similarity(query_embedding, embedding.numpy())
        scored_matches.append(
            {
                "speaker_id": speaker_id,
                "similarity": similarity,
                "distance": 1.0 - similarity,
            }
        )

    scored_matches.sort(key=lambda item: item["similarity"], reverse=True)
    best_match = scored_matches[0]

    if args.speaker_id is None:
        decision_speaker = best_match["speaker_id"]
        decision_similarity = best_match["similarity"]
        accepted = decision_similarity >= args.match_threshold
        mode = "identify"
    else:
        claimed_match = next(item for item in scored_matches if item["speaker_id"] == args.speaker_id)
        decision_speaker = claimed_match["speaker_id"]
        decision_similarity = claimed_match["similarity"]
        accepted = decision_similarity >= args.match_threshold
        mode = "verify"

    result = {
        "mode": mode,
        "accepted": accepted,
        "threshold": float(args.match_threshold),
        "query_audio_files": [str(path) for path in audio_files],
        "store_path": str(store_path),
        "model_path": str(model_path),
        "best_match": best_match,
        "decision": {
            "speaker_id": decision_speaker,
            "similarity": decision_similarity,
            "distance": 1.0 - decision_similarity,
        },
        "top_matches": scored_matches[: max(1, args.top_k)],
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Mode: {mode}")
        print(f"Model: {model_path}")
        print(f"Store: {store_path}")
        print(f"Query files: {', '.join(str(path) for path in audio_files)}")
        print(f"Best match: {best_match['speaker_id']}")
        print(f"Best similarity: {best_match['similarity']:.6f}")
        if args.speaker_id is not None:
            print(f"Claimed speaker: {args.speaker_id}")
            print(f"Claimed similarity: {decision_similarity:.6f}")
        print(f"Threshold: {args.match_threshold:.6f}")
        print(f"Decision: {'accepted' if accepted else 'rejected'}")
        print("Top matches:")
        for item in result["top_matches"]:
            print(
                f"  {item['speaker_id']}: similarity={item['similarity']:.6f} "
                f"distance={item['distance']:.6f}"
            )

    return 0 if accepted else 3


def _enroll(args: argparse.Namespace) -> int:
    model_path = _positive_path(args.model_path, label="Model")
    audio_files = [_positive_path(path, label="Enrollment audio") for path in args.audio_files]
    store_path = Path(args.store_path).expanduser().resolve()

    embedding = _build_embedding(
        audio_files=audio_files,
        model_path=model_path,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        verbose=args.verbose,
    )
    save_enrollment_pt(
        speaker_id=args.speaker_id,
        embedding=embedding,
        store_path=store_path,
        verbose=args.verbose,
    )

    print(f"Enrolled speaker: {args.speaker_id}")
    print(f"Store: {store_path}")
    print(f"Audio files: {', '.join(str(path) for path in audio_files)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "verify":
        return _verify(args)
    if args.command == "enroll":
        return _enroll(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
