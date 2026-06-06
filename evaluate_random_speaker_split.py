from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT

from model.ECAPATDNN import ECAPATDNNBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly enroll a subset of speakers, evaluate the remaining clips from "
            "those speakers, and test all clips from non-enrolled speakers as impostors."
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
        help="Path to the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260602,
        help="Random seed used to choose enrolled speakers.",
    )
    parser.add_argument(
        "--num-enrolled-speakers",
        type=int,
        default=5,
        help="How many speakers to sample for enrollment.",
    )
    parser.add_argument(
        "--enrolled-speakers",
        nargs="+",
        default=None,
        help="Explicit speaker IDs to enroll. If set, random sampling is skipped.",
    )
    parser.add_argument(
        "--enroll-count",
        type=int,
        default=5,
        help="How many clips per enrolled speaker to use for enrollment.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.70],
        help="Acceptance thresholds to summarize.",
    )
    parser.add_argument(
        "--decision-metric",
        choices=["similarity", "softmax"],
        default="similarity",
        help="Metric used for threshold-based accept/reject summaries.",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0,
        help="Temperature applied before softmax when --decision-metric=softmax.",
    )
    parser.add_argument(
        "--softmax-temperatures",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional list of temperatures to sweep when --decision-metric=softmax. "
            "If omitted, --softmax-temperature is used."
        ),
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Audio sample rate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target duration in seconds after pad/crop.",
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
        "--json-out",
        type=Path,
        default=Path("output/random_speaker_split_summary.json"),
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--detailed-json-out",
        type=Path,
        default=Path("output/random_speaker_split_detailed.json"),
        help="Where to write the full per-clip similarity logs.",
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


def _speaker_files(speaker_dir: Path) -> list[Path]:
    return sorted(speaker_dir.glob("*.wav"), key=_natural_key)


def _load_checkpoint_config(model_path: Path) -> tuple[dict[str, torch.Tensor], int, int, int]:
    state = torch.load(model_path, map_location="cpu")
    emb_dim = int(state["fc.1.weight"].shape[0])
    channels = int(state["layer1.bn.weight"].shape[0])
    n_mels = int(state["layer1.conv.weight"].shape[1])
    return state, emb_dim, channels, n_mels


def _build_model(model_path: Path, *, device: torch.device) -> tuple[torch.nn.Module, int, int, int]:
    state, emb_dim, channels, n_mels = _load_checkpoint_config(model_path)
    model = ECAPATDNNBackbone(n_mels=n_mels, channels=channels, emb_dim=emb_dim)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, emb_dim, channels, n_mels


def _make_preprocessor(
    *,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
):
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

    def preprocess_wav(path: Path) -> torch.Tensor:
        waveform, sr_loaded = sf.read(str(path), dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Invalid waveform: {path}")

        wave = torch.from_numpy(waveform)
        if int(sr_loaded) != sr:
            wave = AF.resample(wave, orig_freq=int(sr_loaded), new_freq=sr)

        mel = db_transform(mel_transform(wave.float()))
        current_frames = int(mel.shape[1])
        if current_frames > target_frames:
            start = (current_frames - target_frames) // 2
            mel = mel[:, start : start + target_frames]
        elif current_frames < target_frames:
            pad_frames = target_frames - current_frames
            pad_value = float(torch.min(mel).item())
            mel = torch.nn.functional.pad(mel, (0, pad_frames), value=pad_value)

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel.unsqueeze(0).float()

    return preprocess_wav


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    denom = max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-8)
    return float(np.dot(x, y) / denom)


def _softmax_scores(
    scores: dict[str, float],
    *,
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError(f"softmax temperature must be > 0, got {temperature}")

    labels = list(scores.keys())
    values = np.asarray([scores[label] for label in labels], dtype=np.float64) / temperature
    shifted = values - np.max(values)
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    return {
        label: float(prob)
        for label, prob in zip(labels, probs)
    }


def _embedding_from_wav(
    model: torch.nn.Module,
    preprocess_wav,
    path: Path,
    *,
    device: torch.device,
) -> np.ndarray:
    mel = preprocess_wav(path).to(device)
    with torch.no_grad():
        emb = model(mel).squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        raise ValueError(f"Zero-norm embedding for {path}")
    return emb / norm


def _summarize_threshold(
    *,
    threshold: float,
    genuine_trials: list[dict[str, object]],
    impostor_trials: list[dict[str, object]],
    decision_metric: str,
    softmax_temperature: float | None,
) -> dict[str, object]:
    genuine_metric_key = "similarity" if decision_metric == "similarity" else "probability"
    impostor_metric_key = "best_similarity" if decision_metric == "similarity" else "best_probability"

    genuine_accepted = [t for t in genuine_trials if float(t[genuine_metric_key]) >= threshold]
    impostor_false_accepts = [t for t in impostor_trials if float(t[impostor_metric_key]) >= threshold]

    genuine_by_speaker: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "total": 0})
    for trial in genuine_trials:
        speaker = str(trial["expected"])
        genuine_by_speaker[speaker]["total"] += 1
        genuine_by_speaker[speaker]["accepted"] += int(float(trial[genuine_metric_key]) >= threshold)

    false_accepts_by_impostor = Counter(str(t["impostor_speaker"]) for t in impostor_false_accepts)
    false_accepts_by_target = Counter(str(t["best_target"]) for t in impostor_false_accepts)

    return {
        "threshold": threshold,
        "decision_metric": decision_metric,
        "softmax_temperature": softmax_temperature,
        "genuine_accepts": len(genuine_accepted),
        "genuine_total": len(genuine_trials),
        "genuine_accept_rate": len(genuine_accepted) / len(genuine_trials) if genuine_trials else 0.0,
        "false_accepts": len(impostor_false_accepts),
        "impostor_total": len(impostor_trials),
        "false_accept_rate": len(impostor_false_accepts) / len(impostor_trials) if impostor_trials else 0.0,
        "genuine_accepts_by_speaker": {
            speaker: {
                "accepted": data["accepted"],
                "total": data["total"],
                "accept_rate": data["accepted"] / data["total"] if data["total"] else 0.0,
            }
            for speaker, data in genuine_by_speaker.items()
        },
        "false_accepts_by_impostor": dict(false_accepts_by_impostor),
        "false_accepts_by_target": dict(false_accepts_by_target),
    }


def _attach_softmax_probabilities(
    trials: list[dict[str, object]],
    *,
    label_key: str,
    score_key: str,
    probability_key: str,
    best_probability_key: str,
    temperature: float,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for trial in trials:
        scores = dict(trial[score_key])
        probabilities = _softmax_scores(scores, temperature=temperature)
        best_label = str(trial[label_key])
        updated = dict(trial)
        updated[probability_key] = probabilities
        updated[best_probability_key] = float(probabilities[best_label])
        enriched.append(updated)
    return enriched


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    speaker_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for speaker_dir in speaker_dirs:
        files = _speaker_files(speaker_dir)
        if len(files) < args.enroll_count + 1:
            raise ValueError(
                f"Speaker '{speaker_dir.name}' needs at least {args.enroll_count + 1} clips, "
                f"found {len(files)}."
            )

    if args.enrolled_speakers:
        requested = args.enrolled_speakers
        by_name = {speaker_dir.name: speaker_dir for speaker_dir in speaker_dirs}
        missing = [speaker for speaker in requested if speaker not in by_name]
        if missing:
            raise ValueError(
                f"Requested enrolled speaker(s) not found in {dataset_dir}: {', '.join(missing)}"
            )
        selected_dirs = sorted([by_name[speaker] for speaker in requested], key=lambda p: p.name)
    else:
        if len(speaker_dirs) < args.num_enrolled_speakers:
            raise ValueError(
                f"Requested {args.num_enrolled_speakers} enrolled speakers, "
                f"but only found {len(speaker_dirs)} under {dataset_dir}."
            )
        rng = random.Random(args.sample_seed)
        selected_dirs = sorted(rng.sample(speaker_dirs, args.num_enrolled_speakers), key=lambda p: p.name)
    selected_speakers = [p.name for p in selected_dirs]
    non_selected_speakers = [p.name for p in speaker_dirs if p.name not in selected_speakers]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, emb_dim, channels, n_mels = _build_model(model_path, device=device)
    preprocess_wav = _make_preprocessor(
        sr=args.sr,
        n_mels=n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    print(f"Dataset: {dataset_dir}")
    print(f"Model: {model_path}")
    if args.enrolled_speakers:
        print("Sample seed: explicit speaker list")
    else:
        print(f"Sample seed: {args.sample_seed}")
    print(f"Enrolled speakers: {selected_speakers}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Decision metric: {args.decision_metric}")
    if args.decision_metric == "softmax":
        print(f"Softmax temperature: {args.softmax_temperature}")
        if args.softmax_temperatures:
            print(f"Softmax temperatures: {args.softmax_temperatures}")
    print(f"Device: {device}")

    prototypes: dict[str, np.ndarray] = {}
    enrolled_file_map: dict[str, list[str]] = {}
    for speaker_dir in selected_dirs:
        files = _speaker_files(speaker_dir)
        enrolled_file_map[speaker_dir.name] = [f.name for f in files]
        enroll_files = files[: args.enroll_count]
        emb_stack = np.stack(
            [
                _embedding_from_wav(
                    model,
                    preprocess_wav,
                    audio_path,
                    device=device,
                )
                for audio_path in enroll_files
            ],
            axis=0,
        )
        proto = emb_stack.mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-8)
        prototypes[speaker_dir.name] = proto.astype(np.float32)

    genuine_trials: list[dict[str, object]] = []
    confusion = {speaker: Counter() for speaker in selected_speakers}
    for speaker_dir in selected_dirs:
        files = _speaker_files(speaker_dir)
        for test_file in files[args.enroll_count :]:
            query = _embedding_from_wav(model, preprocess_wav, test_file, device=device)
            scores = {
                target: _cosine_similarity(query, proto)
                for target, proto in prototypes.items()
            }
            predicted = max(scores, key=scores.get)
            row = {
                "speaker_type": "genuine",
                "expected": speaker_dir.name,
                "test_file": test_file.name,
                "predicted": predicted,
                "similarity": float(scores[predicted]),
                "scores": scores,
                "correct": predicted == speaker_dir.name,
            }
            genuine_trials.append(row)
            confusion[speaker_dir.name][predicted] += 1
            if args.verbose:
                print(
                    f"[genuine] {speaker_dir.name}/{test_file.name} -> "
                    f"{predicted} ({row['similarity']:.4f})"
                )

    impostor_trials: list[dict[str, object]] = []
    for speaker_dir in speaker_dirs:
        if speaker_dir.name in selected_speakers:
            continue
        files = _speaker_files(speaker_dir)
        for test_file in files:
            query = _embedding_from_wav(model, preprocess_wav, test_file, device=device)
            scores = {
                target: _cosine_similarity(query, proto)
                for target, proto in prototypes.items()
            }
            best_target = max(scores, key=scores.get)
            row = {
                "speaker_type": "impostor",
                "impostor_speaker": speaker_dir.name,
                "test_file": test_file.name,
                "best_target": best_target,
                "best_similarity": float(scores[best_target]),
                "scores": scores,
            }
            impostor_trials.append(row)
            if args.verbose:
                print(
                    f"[impostor] {speaker_dir.name}/{test_file.name} -> "
                    f"{best_target} ({row['best_similarity']:.4f})"
                )

    thresholds = sorted(set(args.thresholds))

    if args.decision_metric == "softmax":
        detailed_genuine_trials = _attach_softmax_probabilities(
            genuine_trials,
            label_key="predicted",
            score_key="scores",
            probability_key="probabilities",
            best_probability_key="probability",
            temperature=args.softmax_temperature,
        )
        detailed_impostor_trials = _attach_softmax_probabilities(
            impostor_trials,
            label_key="best_target",
            score_key="scores",
            probability_key="probabilities",
            best_probability_key="best_probability",
            temperature=args.softmax_temperature,
        )

        temperatures = args.softmax_temperatures or [args.softmax_temperature]
        temperature_summaries = []
        for temperature in temperatures:
            temp_genuine_trials = _attach_softmax_probabilities(
                genuine_trials,
                label_key="predicted",
                score_key="scores",
                probability_key="probabilities",
                best_probability_key="probability",
                temperature=temperature,
            )
            temp_impostor_trials = _attach_softmax_probabilities(
                impostor_trials,
                label_key="best_target",
                score_key="scores",
                probability_key="probabilities",
                best_probability_key="best_probability",
                temperature=temperature,
            )
            temperature_summaries.append(
                {
                    "softmax_temperature": temperature,
                    "threshold_summaries": [
                        _summarize_threshold(
                            threshold=threshold,
                            genuine_trials=temp_genuine_trials,
                            impostor_trials=temp_impostor_trials,
                            decision_metric=args.decision_metric,
                            softmax_temperature=temperature,
                        )
                        for threshold in thresholds
                    ],
                }
            )
    else:
        detailed_genuine_trials = genuine_trials
        detailed_impostor_trials = impostor_trials
        temperature_summaries = [
            {
                "softmax_temperature": None,
                "threshold_summaries": [
                    _summarize_threshold(
                        threshold=threshold,
                        genuine_trials=genuine_trials,
                        impostor_trials=impostor_trials,
                        decision_metric=args.decision_metric,
                        softmax_temperature=None,
                    )
                    for threshold in thresholds
                ],
            }
        ]

    summary = {
        "model": str(model_path),
        "dataset": str(dataset_dir),
        "sample_seed": None if args.enrolled_speakers else args.sample_seed,
        "embedding_dim": emb_dim,
        "channels": channels,
        "n_mels": n_mels,
        "decision_metric": args.decision_metric,
        "softmax_temperature": args.softmax_temperature,
        "softmax_temperatures": args.softmax_temperatures,
        "enrolled_speakers": selected_speakers,
        "non_enrolled_speakers": non_selected_speakers,
        "enroll_count": args.enroll_count,
        "genuine_classification": {
            "overall_correct": sum(int(t["correct"]) for t in genuine_trials),
            "overall_total": len(genuine_trials),
            "overall_accuracy": (
                sum(int(t["correct"]) for t in genuine_trials) / len(genuine_trials)
                if genuine_trials
                else 0.0
            ),
            "confusion": {speaker: dict(confusion[speaker]) for speaker in selected_speakers},
        },
        "temperature_summaries": temperature_summaries,
    }

    detailed = {
        "model": str(model_path),
        "dataset": str(dataset_dir),
        "sample_seed": None if args.enrolled_speakers else args.sample_seed,
        "decision_metric": args.decision_metric,
        "softmax_temperature": args.softmax_temperature,
        "softmax_temperatures": args.softmax_temperatures,
        "enrolled_speakers": selected_speakers,
        "non_enrolled_speakers": non_selected_speakers,
        "enroll_count": args.enroll_count,
        "thresholds": thresholds,
        "speaker_file_map": enrolled_file_map,
        "genuine_trials": detailed_genuine_trials,
        "impostor_trials": detailed_impostor_trials,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.detailed_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.detailed_json_out.write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    print(f"Summary JSON: {args.json_out.resolve()}")
    print(f"Detailed JSON: {args.detailed_json_out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
