from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT

from utils.model_functions import load_model


@dataclass(frozen=True)
class TrialResult:
    enroll_count: int
    speaker_id: str
    test_file: str
    predicted_speaker: str
    expected_speaker: str
    similarity: float
    correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a speaker dataset by progressively increasing the number of "
            "enrollment clips per speaker and testing the remaining clips."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root dataset directory. Each subdirectory is treated as one speaker.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("output/ECAPATDNN_protonet_model.pth"),
        help="Path to pretrained PyTorch checkpoint.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Audio sample rate.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel bins.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in seconds.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length.",
    )
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
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("output/progressive_enrollment_eval.csv"),
        help="Where to write per-trial results as CSV.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/progressive_enrollment_summary.json"),
        help="Where to write summary metrics as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress.",
    )
    return parser.parse_args()


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.stem.lower())
    key: list[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def _speaker_audio_files(speaker_dir: Path) -> list[Path]:
    wav_files = sorted(speaker_dir.glob("*.wav"), key=lambda path: path.name.lower())
    enroll_files = sorted(
        [path for path in wav_files if path.stem.lower().startswith("enroll")],
        key=_natural_key,
    )
    other_files = sorted(
        [path for path in wav_files if path not in enroll_files],
        key=lambda path: path.name.lower(),
    )
    ordered = enroll_files + other_files
    if len(ordered) < 2:
        raise ValueError(
            f"Speaker '{speaker_dir.name}' must contain at least 2 .wav files, found {len(ordered)}"
        )
    return ordered


def _load_speaker_dataset(dataset_dir: Path) -> dict[str, list[Path]]:
    speaker_dirs = sorted([path for path in dataset_dir.iterdir() if path.is_dir()])
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {dataset_dir}")

    dataset: dict[str, list[Path]] = {}
    expected_file_count: int | None = None
    for speaker_dir in speaker_dirs:
        audio_files = _speaker_audio_files(speaker_dir)
        if expected_file_count is None:
            expected_file_count = len(audio_files)
        elif len(audio_files) != expected_file_count:
            raise ValueError(
                "All speakers must have the same number of .wav files for progressive "
                f"evaluation. '{speaker_dir.name}' has {len(audio_files)}, expected {expected_file_count}."
            )
        dataset[speaker_dir.name] = audio_files
    return dataset


def _pad_or_crop_mel(mel: torch.Tensor, *, target_frames: int) -> torch.Tensor:
    current_frames = int(mel.shape[1])
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif current_frames < target_frames:
        pad_frames = target_frames - current_frames
        pad_value = float(torch.min(mel).item())
        mel = F.pad(mel, (0, pad_frames), value=pad_value)
    return mel


def _waveform_chunks(
    waveform: np.ndarray,
    *,
    sr: int,
    chunk_duration: float,
    overlap_duration: float,
) -> list[np.ndarray]:
    if waveform.size == 0:
        return []

    chunk_size = int(chunk_duration * sr)
    overlap_size = int(overlap_duration * sr)
    step_size = max(1, chunk_size - overlap_size)

    chunks: list[np.ndarray] = []
    for start in range(0, len(waveform), step_size):
        end = start + chunk_size
        chunk = waveform[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")
        chunks.append(chunk.astype(np.float32, copy=False))
        if end >= len(waveform):
            break
    return chunks


def _mel_tensor_from_chunk(
    chunk: np.ndarray,
    *,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    target_duration: float,
) -> torch.Tensor:
    waveform = torch.from_numpy(chunk).float()
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

    mel = db_transform(mel_transform(waveform))
    target_frames = int(target_duration * sr / hop_length)
    mel = _pad_or_crop_mel(mel, target_frames=target_frames)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel.unsqueeze(0)


def _embedding_from_audio_files(
    model: torch.nn.Module,
    audio_files: Iterable[Path],
    *,
    device: torch.device,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    chunk_duration: float | None,
    chunk_overlap: float,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    chunk_dur = chunk_duration or duration

    for audio_path in audio_files:
        waveform, sr_loaded = sf.read(str(audio_path), dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Failed to load a valid mono waveform from {audio_path}")
        if sr_loaded != sr:
            waveform = AF.resample(
                torch.from_numpy(waveform),
                orig_freq=int(sr_loaded),
                new_freq=sr,
            ).cpu().numpy()
            sr_loaded = sr

        wave_chunks = _waveform_chunks(
            waveform,
            sr=sr_loaded,
            chunk_duration=chunk_dur,
            overlap_duration=chunk_overlap,
        )

        for chunk in wave_chunks:
            mel = _mel_tensor_from_chunk(
                chunk,
                sr=sr_loaded,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                target_duration=duration,
            )
            with torch.no_grad():
                emb = model(mel.to(device)).squeeze(0).cpu()
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No usable embeddings were produced from the supplied audio files.")

    stacked = torch.stack(embeddings, dim=0)
    return F.normalize(stacked.mean(dim=0), p=2, dim=0).cpu()


def _evaluate_progressive(
    dataset: dict[str, list[Path]],
    *,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> list[TrialResult]:
    total_files = len(next(iter(dataset.values())))
    max_enroll = total_files - 1
    results: list[TrialResult] = []

    for enroll_count in range(1, max_enroll + 1):
        prototypes: dict[str, torch.Tensor] = {}
        for speaker_id, audio_files in dataset.items():
            enroll_files = audio_files[:enroll_count]
            if args.verbose:
                print(
                    f"[enroll={enroll_count}] building prototype for {speaker_id} "
                    f"from {len(enroll_files)} file(s)"
                )
            prototypes[speaker_id] = _embedding_from_audio_files(
                model,
                enroll_files,
                device=device,
                sr=args.sr,
                n_mels=args.n_mels,
                duration=args.duration,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                chunk_duration=args.chunk_duration,
                chunk_overlap=args.chunk_overlap,
            )

        speaker_ids = list(prototypes.keys())
        prototype_matrix = torch.stack([prototypes[speaker_id] for speaker_id in speaker_ids], dim=0)

        for speaker_id, audio_files in dataset.items():
            test_files = audio_files[enroll_count:]
            for test_file in test_files:
                query_embedding = _embedding_from_audio_files(
                    model,
                    [test_file],
                    device=device,
                    sr=args.sr,
                    n_mels=args.n_mels,
                    duration=args.duration,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    chunk_duration=args.chunk_duration,
                    chunk_overlap=args.chunk_overlap,
                )

                scores = torch.matmul(prototype_matrix, query_embedding)
                best_index = int(torch.argmax(scores).item())
                predicted_speaker = speaker_ids[best_index]
                similarity = float(scores[best_index].item())
                correct = predicted_speaker == speaker_id

                if args.verbose:
                    print(
                        f"[enroll={enroll_count}] {speaker_id}/{test_file.name} -> "
                        f"{predicted_speaker} ({similarity:.4f}) {'OK' if correct else 'FAIL'}"
                    )

                results.append(
                    TrialResult(
                        enroll_count=enroll_count,
                        speaker_id=speaker_id,
                        test_file=test_file.name,
                        predicted_speaker=predicted_speaker,
                        expected_speaker=speaker_id,
                        similarity=similarity,
                        correct=correct,
                    )
                )

    return results


def _write_csv(results: list[TrialResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "enroll_count",
                "speaker_id",
                "test_file",
                "predicted_speaker",
                "expected_speaker",
                "similarity",
                "correct",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "enroll_count": item.enroll_count,
                    "speaker_id": item.speaker_id,
                    "test_file": item.test_file,
                    "predicted_speaker": item.predicted_speaker,
                    "expected_speaker": item.expected_speaker,
                    "similarity": f"{item.similarity:.8f}",
                    "correct": int(item.correct),
                }
            )


def _summarize(results: list[TrialResult]) -> dict[str, object]:
    by_enroll: dict[int, list[TrialResult]] = {}
    for item in results:
        by_enroll.setdefault(item.enroll_count, []).append(item)

    summary_rows: list[dict[str, object]] = []
    overall_correct = 0
    for enroll_count in sorted(by_enroll):
        bucket = by_enroll[enroll_count]
        correct = sum(1 for item in bucket if item.correct)
        total = len(bucket)
        overall_correct += correct
        summary_rows.append(
            {
                "enroll_count": enroll_count,
                "num_trials": total,
                "num_correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )

    return {
        "overall_trials": len(results),
        "overall_correct": overall_correct,
        "overall_accuracy": overall_correct / len(results) if results else 0.0,
        "by_enroll_count": summary_rows,
    }


def _write_json(summary: dict[str, object], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    dataset = _load_speaker_dataset(dataset_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path,
        device=device,
        n_mels=args.n_mels,
        emb_dim=64,
        channels=512,
    )

    print(f"Dataset: {dataset_dir}")
    print(f"Speakers: {len(dataset)}")
    print(f"Files per speaker: {len(next(iter(dataset.values())))}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")

    results = _evaluate_progressive(
        dataset,
        model=model,
        device=device,
        args=args,
    )
    summary = _summarize(results)

    _write_csv(results, args.csv_out)
    _write_json(summary, args.json_out)

    print("\nSummary by enroll count:")
    for row in summary["by_enroll_count"]:
        print(
            f"  enroll={row['enroll_count']}: "
            f"{row['num_correct']}/{row['num_trials']} "
            f"({row['accuracy']:.2%})"
        )

    print(
        f"\nOverall: {summary['overall_correct']}/{summary['overall_trials']} "
        f"({summary['overall_accuracy']:.2%})"
    )
    print(f"CSV: {args.csv_out.resolve()}")
    print(f"JSON: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
