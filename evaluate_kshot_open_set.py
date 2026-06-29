from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT

from enroll_lite import _run_kmeans
from speaker_app.model.ecapa_tdnn import ECAPATDNNBackbone
from speaker_app.services.embedding_math import cosine_similarity
from speaker_app.services.enrollment_service import EnrollmentService
from speaker_app.services.speaker_repository import SpeakerRepository
from utils.data_preprocessing import apply_vad


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int
    n_mels: int
    duration: float
    n_fft: int
    hop_length: int
    vad_enabled: bool
    vad_top_db: float
    vad_frame_length: int
    vad_hop_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 1..K-shot open-set speaker recognition experiment. A fixed set "
            "of speakers is enrolled; remaining clips from enrolled speakers plus "
            "all clips from non-enrolled speakers are used as queries."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("audio_data"),
        help="Root dataset directory. Each direct child directory is one speaker.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("output/ecapa_tdnn_protonet_model.pth"),
        help="Path to the ECAPA-TDNN ProtoNet checkpoint.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/ecapa_tdnn_protonet_kshot_open_set_eval.json"),
        help="Where to write the detailed JSON report.",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("output/ecapa_tdnn_protonet_kshot_open_set_dbs"),
        help="Directory where per-k SQLite speaker databases are written.",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version stored with enrolled profiles. Defaults to ecapa-tdnn:<model filename>.",
    )
    parser.add_argument(
        "--num-enrolled-speakers",
        type=int,
        default=6,
        help="Number of speakers to enroll.",
    )
    parser.add_argument(
        "--enrolled-speakers",
        nargs="+",
        default=None,
        help="Explicit speaker IDs to enroll. If set, speaker selection is skipped.",
    )
    parser.add_argument(
        "--speaker-selection",
        choices=["first", "random"],
        default="first",
        help="How to choose enrolled speakers when --enrolled-speakers is not set.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=36,
        help="Random seed used when --speaker-selection=random.",
    )
    parser.add_argument(
        "--max-enroll-count",
        type=int,
        default=6,
        help="Evaluate k-shot enrollment for k=1..this value.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Fixed operating threshold used for reported Accuracy/GAR/FAR/FNR.",
    )
    parser.add_argument(
        "--prototype-mode",
        choices=["primary", "primary-plus-subprototypes"],
        default="primary-plus-subprototypes",
        help=(
            "primary uses one DB prototype per speaker; primary-plus-subprototypes "
            "adds KMeans sub-prototypes and scores by max similarity."
        ),
    )
    parser.add_argument(
        "--num-sub-prototypes",
        type=int,
        default=3,
        help="Number of KMeans sub-prototypes per enrolled speaker.",
    )
    parser.add_argument(
        "--kmeans-max-iters",
        type=int,
        default=50,
        help="Maximum KMeans iterations for sub-prototype clustering.",
    )
    parser.add_argument(
        "--kmeans-seed",
        type=int,
        default=0,
        help="Random seed for KMeans sub-prototype initialization.",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--n-mels", type=int, default=None, help="Mel bins. Defaults to checkpoint value.")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target model audio duration in seconds. The speaker_app runtime default is 3.0.",
    )
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size.")
    parser.add_argument("--hop-length", type=int, default=256, help="STFT hop length.")
    parser.add_argument(
        "--vad-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply energy VAD before embedding extraction. The speaker_app runtime default is disabled.",
    )
    parser.add_argument("--vad-top-db", type=float, default=10.0, help="VAD threshold in dB.")
    parser.add_argument("--vad-frame-length", type=int, default=2048, help="VAD frame length.")
    parser.add_argument("--vad-hop-length", type=int, default=258, help="VAD hop length.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true", help="Print per-file progress.")
    return parser.parse_args()


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _speaker_files(speaker_dir: Path) -> list[Path]:
    files = [
        path
        for path in speaker_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    enroll_files = sorted(
        [path for path in files if path.stem.lower().startswith("enroll")],
        key=_natural_key,
    )
    other_files = sorted(
        [path for path in files if path not in enroll_files],
        key=_natural_key,
    )
    return enroll_files + other_files


def _load_dataset(dataset_dir: Path) -> dict[str, list[Path]]:
    speaker_dirs = sorted([path for path in dataset_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if not speaker_dirs:
        raise ValueError(f"No speaker directories found in {dataset_dir}")

    dataset: dict[str, list[Path]] = {}
    for speaker_dir in speaker_dirs:
        files = _speaker_files(speaker_dir)
        if files:
            dataset[speaker_dir.name] = files
    if not dataset:
        raise ValueError(f"No supported audio files found in {dataset_dir}")
    return dataset


def _select_speakers(
    dataset: dict[str, list[Path]],
    *,
    explicit_speakers: list[str] | None,
    num_enrolled: int,
    max_enroll_count: int,
    selection: str,
    seed: int,
) -> list[str]:
    eligible = sorted(
        speaker for speaker, files in dataset.items() if len(files) > max_enroll_count
    )
    if explicit_speakers:
        missing = [speaker for speaker in explicit_speakers if speaker not in dataset]
        ineligible = [
            speaker
            for speaker in explicit_speakers
            if speaker in dataset and len(dataset[speaker]) <= max_enroll_count
        ]
        if missing:
            raise ValueError(f"Requested speaker(s) not found: {', '.join(missing)}")
        if ineligible:
            raise ValueError(
                "Requested speaker(s) need at least "
                f"{max_enroll_count + 1} files: {', '.join(ineligible)}"
            )
        if len(explicit_speakers) != num_enrolled:
            raise ValueError(
                f"--enrolled-speakers contains {len(explicit_speakers)} speaker(s), "
                f"but --num-enrolled-speakers is {num_enrolled}."
            )
        return sorted(explicit_speakers)

    if len(eligible) < num_enrolled:
        raise ValueError(
            f"Need {num_enrolled} speakers with at least {max_enroll_count + 1} clips; "
            f"found {len(eligible)}."
        )

    if selection == "first":
        return eligible[:num_enrolled]

    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eligible, size=num_enrolled, replace=False).tolist())


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")

    if isinstance(payload, dict) and not all(torch.is_tensor(value) for value in payload.values()):
        payload = next(
            (payload[key] for key in ("state_dict", "model_state_dict", "model") if key in payload),
            payload,
        )
    if not isinstance(payload, dict) or not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError(f"Checkpoint does not look like a PyTorch state_dict: {model_path}")

    if payload and all(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return payload


def _infer_model_config(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    try:
        emb_dim = int(state["fc.1.weight"].shape[0])
        channels = int(state["layer1.bn.weight"].shape[0])
        n_mels = int(state["layer1.conv.weight"].shape[1])
    except KeyError as exc:
        raise ValueError("Checkpoint is missing expected ECAPA-TDNN backbone keys.") from exc
    return n_mels, channels, emb_dim


def _build_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, int, int, int]:
    state = _load_state_dict(model_path)
    n_mels, channels, emb_dim = _infer_model_config(state)
    model = ECAPATDNNBackbone(n_mels=n_mels, channels=channels, embedding_dimension=emb_dim)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, n_mels, channels, emb_dim


class CachedEmbeddingExtractor:
    def __init__(
        self,
        embeddings: dict[str, dict[Path, np.ndarray]],
        *,
        model_version: str,
        embedding_dimension: int,
    ) -> None:
        self._model_version = model_version
        self._embedding_dimension = embedding_dimension
        self._by_path = {
            audio_path.resolve(): embedding
            for speaker_embeddings in embeddings.values()
            for audio_path, embedding in speaker_embeddings.items()
        }

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def extract(self, audio_path: Path) -> np.ndarray:
        key = Path(audio_path).resolve()
        try:
            return self._by_path[key].copy()
        except KeyError as exc:
            raise ValueError(f"No cached embedding for {audio_path}") from exc


class AudioPreprocessor:
    def __init__(self, config: PreprocessConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.target_samples = max(1, int(round(config.duration * config.sample_rate)))
        self.target_frames = int(config.duration * config.sample_rate / config.hop_length)
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.n_fft,
            n_mels=config.n_mels,
            f_min=0.0,
            f_max=config.sample_rate / 2.0,
            power=2.0,
        ).to(device)
        self.db_transform = AT.AmplitudeToDB(stype="power").to(device)

    def _load_waveform(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"Audio file is empty or invalid: {audio_path}")
        if not np.all(np.isfinite(waveform)):
            raise ValueError(f"Audio file contains non-finite samples: {audio_path}")

        if self.config.vad_enabled:
            waveform = apply_vad(
                waveform,
                top_db=self.config.vad_top_db,
                frame_length=self.config.vad_frame_length,
                hop_length=self.config.vad_hop_length,
            )

        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        if int(sample_rate) != self.config.sample_rate:
            tensor = AF.resample(tensor, orig_freq=int(sample_rate), new_freq=self.config.sample_rate)
        return tensor.float()

    def _pad_or_crop_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        current_samples = int(waveform.shape[0])
        if current_samples > self.target_samples:
            start = (current_samples - self.target_samples) // 2
            waveform = waveform[start : start + self.target_samples]
        elif current_samples < self.target_samples:
            waveform = F.pad(waveform, (0, self.target_samples - current_samples))
        return waveform

    def _pad_or_crop_mel(self, mel: torch.Tensor) -> torch.Tensor:
        current_frames = int(mel.shape[-1])
        if current_frames > self.target_frames:
            start = (current_frames - self.target_frames) // 2
            mel = mel[:, start : start + self.target_frames]
        elif current_frames < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - current_frames), value=float(mel.min().item()))
        return mel

    def __call__(self, audio_path: Path) -> torch.Tensor:
        waveform = self._pad_or_crop_waveform(self._load_waveform(audio_path)).to(self.device)
        mel = self.db_transform(self.mel_transform(waveform))
        mel = self._pad_or_crop_mel(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel.unsqueeze(0).float()


def _extract_embeddings(
    dataset: dict[str, list[Path]],
    *,
    model: torch.nn.Module,
    preprocessor: AudioPreprocessor,
    device: torch.device,
    verbose: bool,
) -> dict[str, dict[Path, np.ndarray]]:
    embeddings: dict[str, dict[Path, np.ndarray]] = {}
    total = sum(len(files) for files in dataset.values())
    processed = 0

    with torch.inference_mode():
        for speaker_id, files in dataset.items():
            speaker_embeddings: dict[Path, np.ndarray] = {}
            for audio_path in files:
                processed += 1
                if verbose:
                    print(f"[embedding {processed}/{total}] {speaker_id}/{audio_path.name}")
                mel = preprocessor(audio_path)
                embedding = model(mel.to(device)).squeeze(0)
                embedding = F.normalize(embedding, p=2, dim=0).cpu().numpy().astype(np.float32)
                speaker_embeddings[audio_path] = embedding
            embeddings[speaker_id] = speaker_embeddings
    return embeddings


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _trial_sort_key(trial: dict[str, object]) -> tuple[str, str]:
    return (str(trial["query_speaker"]), str(trial["query_file"]))


def _build_speaker_prototype_bank(
    *,
    profile_embedding: np.ndarray,
    enrollment_embeddings: list[np.ndarray],
    prototype_mode: str,
    num_sub_prototypes: int,
    kmeans_max_iters: int,
    kmeans_seed: int,
) -> list[dict[str, object]]:
    bank: list[dict[str, object]] = [
        {
            "prototype_type": "primary",
            "prototype_index": 0,
            "embedding": np.asarray(profile_embedding, dtype=np.float32).reshape(-1),
            "source": "speaker_app_db_profile",
            "cluster_size": len(enrollment_embeddings),
        }
    ]
    if prototype_mode == "primary":
        return bank

    if num_sub_prototypes < 1:
        raise ValueError("--num-sub-prototypes must be >= 1 when sub-prototypes are enabled")
    if not enrollment_embeddings:
        raise ValueError("At least one enrollment embedding is required for sub-prototypes")

    matrix = np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in enrollment_embeddings])
    sub_prototypes = _run_kmeans(
        matrix,
        num_clusters=num_sub_prototypes,
        max_iters=max(1, kmeans_max_iters),
        seed=kmeans_seed,
    )
    assignments = np.sum(
        (matrix[:, None, :] - sub_prototypes[None, :, :]) ** 2,
        axis=2,
    ).argmin(axis=1)
    cluster_sizes = np.bincount(assignments, minlength=num_sub_prototypes)

    for index, sub_prototype in enumerate(sub_prototypes):
        bank.append(
            {
                "prototype_type": "sub",
                "prototype_index": index,
                "embedding": np.asarray(sub_prototype, dtype=np.float32).reshape(-1),
                "source": "kmeans_enrollment_embeddings",
                "cluster_size": int(cluster_sizes[index]),
            }
        )
    return bank


def _score_prototype_bank(
    query_embedding: np.ndarray,
    prototype_bank: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, float], dict[str, list[dict[str, object]]], dict[str, object]]:
    speaker_scores: dict[str, float] = {}
    prototype_scores: dict[str, list[dict[str, object]]] = {}
    best: dict[str, object] | None = None

    for speaker_id, prototypes in prototype_bank.items():
        scored: list[dict[str, object]] = []
        for prototype in prototypes:
            similarity = cosine_similarity(query_embedding, np.asarray(prototype["embedding"]))
            row = {
                "prototype_type": prototype["prototype_type"],
                "prototype_index": prototype["prototype_index"],
                "similarity": similarity,
                "cluster_size": prototype["cluster_size"],
            }
            scored.append(row)
            if best is None or similarity > float(best["similarity"]):
                best = {
                    "speaker_id": speaker_id,
                    "prototype_type": prototype["prototype_type"],
                    "prototype_index": prototype["prototype_index"],
                    "similarity": similarity,
                    "cluster_size": prototype["cluster_size"],
                }
        prototype_scores[speaker_id] = scored
        speaker_scores[speaker_id] = max(float(item["similarity"]) for item in scored)

    if best is None:
        raise ValueError("Cannot score query without prototypes")
    return speaker_scores, prototype_scores, best


def _summarize_at_threshold(
    trials: list[dict[str, object]],
    *,
    threshold: float,
) -> dict[str, object]:
    genuine = [trial for trial in trials if trial["trial_type"] == "genuine"]
    impostors = [trial for trial in trials if trial["trial_type"] == "impostor"]

    correct_accepts = 0
    false_rejects = 0
    wrong_accepts = 0
    closed_set_correct = 0
    genuine_accepts_any = 0
    per_speaker: dict[str, Counter[str]] = {}

    for trial in genuine:
        expected = str(trial["expected_speaker"])
        predicted = str(trial["predicted_speaker"])
        accepted = float(trial["best_similarity"]) >= threshold
        correct_prediction = predicted == expected
        closed_set_correct += int(correct_prediction)
        genuine_accepts_any += int(accepted)
        per_speaker.setdefault(expected, Counter())
        if accepted and correct_prediction:
            correct_accepts += 1
            per_speaker[expected]["correct_accept"] += 1
        elif accepted:
            wrong_accepts += 1
            per_speaker[expected]["wrong_accept"] += 1
        else:
            false_rejects += 1
            per_speaker[expected]["false_reject"] += 1
        per_speaker[expected]["total"] += 1

    false_accepts = sum(int(float(trial["best_similarity"]) >= threshold) for trial in impostors)
    true_rejects = len(impostors) - false_accepts
    total_trials = len(genuine) + len(impostors)
    correct_total = correct_accepts + true_rejects

    return {
        "threshold": threshold,
        "accuracy": correct_total / total_trials if total_trials else 0.0,
        "GAR": correct_accepts / len(genuine) if genuine else 0.0,
        "FAR": false_accepts / len(impostors) if impostors else 0.0,
        "FNR": (len(genuine) - correct_accepts) / len(genuine) if genuine else 0.0,
        "genuine_detection_accept_rate": genuine_accepts_any / len(genuine) if genuine else 0.0,
        "closed_set_genuine_accuracy": closed_set_correct / len(genuine) if genuine else 0.0,
        "counts": {
            "total_trials": total_trials,
            "correct_total": correct_total,
            "genuine_total": len(genuine),
            "impostor_total": len(impostors),
            "genuine_correct_accepts": correct_accepts,
            "genuine_false_rejects": false_rejects,
            "genuine_wrong_accepts": wrong_accepts,
            "impostor_false_accepts": false_accepts,
            "impostor_true_rejects": true_rejects,
        },
        "per_enrolled_speaker": {
            speaker: {
                "total": counts["total"],
                "correct_accepts": counts["correct_accept"],
                "false_rejects": counts["false_reject"],
                "wrong_accepts": counts["wrong_accept"],
                "GAR": counts["correct_accept"] / counts["total"] if counts["total"] else 0.0,
            }
            for speaker, counts in sorted(per_speaker.items())
        },
    }


def _threshold_candidates(trials: list[dict[str, object]]) -> list[float]:
    scores = sorted({float(trial["best_similarity"]) for trial in trials})
    if not scores:
        return [0.0]
    return [-1.000001, *scores, 1.000001]


def _compute_eer(sweep: list[dict[str, object]]) -> dict[str, float]:
    if not sweep:
        return {"eer": math.nan, "threshold": math.nan, "FAR": math.nan, "FNR": math.nan}

    sorted_sweep = sorted(sweep, key=lambda row: float(row["threshold"]))
    thresholds = np.asarray([float(row["threshold"]) for row in sorted_sweep], dtype=np.float64)
    far = np.asarray([float(row["FAR"]) for row in sorted_sweep], dtype=np.float64)
    fnr = np.asarray([float(row["FNR"]) for row in sorted_sweep], dtype=np.float64)
    diff = far - fnr

    crossing_indexes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if crossing_indexes.size:
        idx = int(crossing_indexes[0])
        x1, x2 = thresholds[idx], thresholds[idx + 1]
        far1, far2 = far[idx], far[idx + 1]
        fnr1, fnr2 = fnr[idx], fnr[idx + 1]
        diff1, diff2 = diff[idx], diff[idx + 1]
        denom = diff1 - diff2
        if abs(denom) < 1e-12:
            weight = 0.0
        else:
            weight = diff1 / denom
        threshold = x1 + weight * (x2 - x1)
        eer_far = far1 + weight * (far2 - far1)
        eer_fnr = fnr1 + weight * (fnr2 - fnr1)
        eer = (eer_far + eer_fnr) / 2.0
        return {
            "eer": float(eer),
            "threshold": float(threshold),
            "FAR": float(eer_far),
            "FNR": float(eer_fnr),
        }

    idx = int(np.argmin(np.abs(diff)))
    eer = (far[idx] + fnr[idx]) / 2.0
    return {
        "eer": float(eer),
        "threshold": float(thresholds[idx]),
        "FAR": float(far[idx]),
        "FNR": float(fnr[idx]),
    }


def _best_accuracy_row(sweep: list[dict[str, object]]) -> dict[str, object]:
    return max(
        sweep,
        key=lambda row: (
            float(row["accuracy"]),
            -float(row["FAR"]),
            float(row["GAR"]),
            -abs(float(row["FAR"]) - float(row["FNR"])),
        ),
    )


def _evaluate_k(
    *,
    k: int,
    dataset: dict[str, list[Path]],
    extractor: CachedEmbeddingExtractor,
    enrolled_speakers: list[str],
    dataset_dir: Path,
    operating_threshold: float,
    database_path: Path,
    prototype_mode: str,
    num_sub_prototypes: int,
    kmeans_max_iters: int,
    kmeans_seed: int,
) -> dict[str, object]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    repository = SpeakerRepository(database_path)
    repository.initialize()
    enrollment = EnrollmentService(repository, extractor, required_clip_count=k)

    enrollment_files: dict[str, list[str]] = {}
    for speaker in enrolled_speakers:
        files = dataset[speaker]
        enroll_files = files[:k]
        enrollment_files[speaker] = [_relative(path, dataset_dir) for path in enroll_files]
        result = enrollment.enroll(speaker, speaker, enroll_files)
        if not result.success:
            raise RuntimeError(f"Enrollment failed for {speaker} at k={k}: {result.message}")

    profiles = repository.get_all_compatible(
        extractor.model_version,
        extractor.embedding_dimension,
    )
    if len(profiles) != len(enrolled_speakers):
        raise RuntimeError(
            f"Expected {len(enrolled_speakers)} compatible profiles at k={k}; found {len(profiles)}"
        )
    profiles_by_speaker = {profile.speaker_id: profile for profile in profiles}
    prototype_bank: dict[str, list[dict[str, object]]] = {}
    prototype_summary: dict[str, list[dict[str, object]]] = {}
    for speaker in enrolled_speakers:
        enroll_files = dataset[speaker][:k]
        bank = _build_speaker_prototype_bank(
            profile_embedding=profiles_by_speaker[speaker].embedding,
            enrollment_embeddings=[extractor.extract(path) for path in enroll_files],
            prototype_mode=prototype_mode,
            num_sub_prototypes=num_sub_prototypes,
            kmeans_max_iters=kmeans_max_iters,
            kmeans_seed=kmeans_seed,
        )
        prototype_bank[speaker] = bank
        prototype_summary[speaker] = [
            {
                "prototype_type": item["prototype_type"],
                "prototype_index": item["prototype_index"],
                "source": item["source"],
                "cluster_size": item["cluster_size"],
            }
            for item in bank
        ]

    trials: list[dict[str, object]] = []
    for query_speaker, files in dataset.items():
        query_files = files[k:] if query_speaker in enrolled_speakers else files
        for query_file in query_files:
            query_embedding = extractor.extract(query_file)
            scores, prototype_scores, best_prototype = _score_prototype_bank(
                query_embedding,
                prototype_bank,
            )
            predicted = max(scores, key=scores.get)
            best_similarity = float(scores[predicted])
            accepted = best_similarity >= operating_threshold
            recognized_speaker = predicted if accepted else None
            if query_speaker in enrolled_speakers:
                expected_similarity = float(scores[query_speaker])
                trials.append(
                    {
                        "trial_type": "genuine",
                        "query_speaker": query_speaker,
                        "query_file": _relative(query_file, dataset_dir),
                        "expected_speaker": query_speaker,
                        "predicted_speaker": predicted,
                        "recognized_speaker": recognized_speaker,
                        "accepted_at_operating_threshold": accepted,
                        "best_similarity": best_similarity,
                        "expected_similarity": expected_similarity,
                        "closed_set_correct": predicted == query_speaker,
                        "scores": scores,
                        "prototype_scores": prototype_scores,
                        "best_prototype": best_prototype,
                    }
                )
            else:
                trials.append(
                    {
                        "trial_type": "impostor",
                        "query_speaker": query_speaker,
                        "query_file": _relative(query_file, dataset_dir),
                        "expected_speaker": None,
                        "predicted_speaker": predicted,
                        "recognized_speaker": recognized_speaker,
                        "accepted_at_operating_threshold": accepted,
                        "best_similarity": best_similarity,
                        "scores": scores,
                        "prototype_scores": prototype_scores,
                        "best_prototype": best_prototype,
                    }
                )

    trials = sorted(trials, key=_trial_sort_key)
    thresholds = _threshold_candidates(trials)
    sweep = [
        _summarize_at_threshold(trials, threshold=threshold)
        for threshold in thresholds
    ]
    eer = _compute_eer(sweep)
    metrics_at_eer_threshold = _summarize_at_threshold(trials, threshold=eer["threshold"])
    best_accuracy = _best_accuracy_row(sweep)
    metrics_at_operating_threshold = _summarize_at_threshold(
        trials,
        threshold=operating_threshold,
    )

    genuine_trials = [trial for trial in trials if trial["trial_type"] == "genuine"]
    impostor_trials = [trial for trial in trials if trial["trial_type"] == "impostor"]

    return {
        "k": k,
        "database": {
            "path": str(database_path),
            "profile_count": repository.count(),
            "compatible_profile_count": len(profiles),
            "stored_profile_ids": [profile.speaker_id for profile in profiles],
        },
        "prototype_mode": prototype_mode,
        "num_sub_prototypes": 0 if prototype_mode == "primary" else num_sub_prototypes,
        "prototypes_per_speaker": len(next(iter(prototype_bank.values()))) if prototype_bank else 0,
        "prototype_summary": prototype_summary,
        "enrollment_files": enrollment_files,
        "query_counts": {
            "genuine": len(genuine_trials),
            "impostor": len(impostor_trials),
            "total": len(trials),
        },
        "operating_threshold": operating_threshold,
        "metrics_at_operating_threshold": metrics_at_operating_threshold,
        "metrics_at_best_accuracy_threshold": best_accuracy,
        "eer": eer,
        "metrics_at_eer_threshold": metrics_at_eer_threshold,
        "threshold_sweep": sweep,
        "trials": trials,
    }


def _top_level_best(results: list[dict[str, object]]) -> dict[str, object]:
    best_by_operating_threshold = max(
        results,
        key=lambda row: (
            float(row["metrics_at_operating_threshold"]["accuracy"]),
            -float(row["metrics_at_operating_threshold"]["FAR"]),
            float(row["metrics_at_operating_threshold"]["GAR"]),
            -int(row["k"]),
        ),
    )
    best_by_accuracy = max(
        results,
        key=lambda row: (
            float(row["metrics_at_best_accuracy_threshold"]["accuracy"]),
            -float(row["metrics_at_best_accuracy_threshold"]["FAR"]),
            -float(row["eer"]["eer"]),
            -int(row["k"]),
        ),
    )
    best_by_eer = min(
        results,
        key=lambda row: (
            float(row["eer"]["eer"]),
            -float(row["metrics_at_best_accuracy_threshold"]["accuracy"]),
            int(row["k"]),
        ),
    )
    return {
        "by_operating_threshold": {
            "k": best_by_operating_threshold["k"],
            "threshold": best_by_operating_threshold["operating_threshold"],
            "accuracy": best_by_operating_threshold["metrics_at_operating_threshold"]["accuracy"],
            "GAR": best_by_operating_threshold["metrics_at_operating_threshold"]["GAR"],
            "FAR": best_by_operating_threshold["metrics_at_operating_threshold"]["FAR"],
            "FNR": best_by_operating_threshold["metrics_at_operating_threshold"]["FNR"],
            "eer": best_by_operating_threshold["eer"]["eer"],
        },
        "by_best_accuracy_threshold": {
            "k": best_by_accuracy["k"],
            "accuracy": best_by_accuracy["metrics_at_best_accuracy_threshold"]["accuracy"],
            "GAR": best_by_accuracy["metrics_at_best_accuracy_threshold"]["GAR"],
            "FAR": best_by_accuracy["metrics_at_best_accuracy_threshold"]["FAR"],
            "FNR": best_by_accuracy["metrics_at_best_accuracy_threshold"]["FNR"],
            "threshold": best_by_accuracy["metrics_at_best_accuracy_threshold"]["threshold"],
            "eer": best_by_accuracy["eer"]["eer"],
        },
        "by_lowest_eer": {
            "k": best_by_eer["k"],
            "eer": best_by_eer["eer"]["eer"],
            "eer_threshold": best_by_eer["eer"]["threshold"],
            "accuracy_at_best_accuracy_threshold": best_by_eer["metrics_at_best_accuracy_threshold"]["accuracy"],
        },
    }


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()
    json_out = args.json_out.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    database_run_root = (args.database_root.resolve() / run_id)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if args.max_enroll_count < 1:
        raise ValueError("--max-enroll-count must be >= 1")
    if args.num_enrolled_speakers < 1:
        raise ValueError("--num-enrolled-speakers must be >= 1")
    if not -1.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between -1 and 1")
    if args.num_sub_prototypes < 1:
        raise ValueError("--num-sub-prototypes must be >= 1")
    if args.kmeans_max_iters < 1:
        raise ValueError("--kmeans-max-iters must be >= 1")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = _load_dataset(dataset_dir)
    model, inferred_n_mels, channels, emb_dim = _build_model(model_path, device)
    model_version = args.model_version or f"ecapa-tdnn:{model_path.name}"
    n_mels = inferred_n_mels if args.n_mels is None else args.n_mels
    if n_mels != inferred_n_mels:
        raise ValueError(
            f"--n-mels={n_mels} does not match checkpoint input bins ({inferred_n_mels})."
        )

    enrolled_speakers = _select_speakers(
        dataset,
        explicit_speakers=args.enrolled_speakers,
        num_enrolled=args.num_enrolled_speakers,
        max_enroll_count=args.max_enroll_count,
        selection=args.speaker_selection,
        seed=args.sample_seed,
    )
    non_enrolled_speakers = sorted(speaker for speaker in dataset if speaker not in enrolled_speakers)

    preprocess_config = PreprocessConfig(
        sample_rate=args.sr,
        n_mels=n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        vad_enabled=args.vad_enabled,
        vad_top_db=args.vad_top_db,
        vad_frame_length=args.vad_frame_length,
        vad_hop_length=args.vad_hop_length,
    )
    preprocessor = AudioPreprocessor(preprocess_config, device)

    print(f"Dataset: {dataset_dir}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Enrolled speakers: {enrolled_speakers}")
    print(f"Non-enrolled speakers: {non_enrolled_speakers}")
    print(f"K-shot range: 1..{args.max_enroll_count}")
    print(f"Operating threshold: {args.threshold}")
    print(f"Prototype mode: {args.prototype_mode}")
    if args.prototype_mode == "primary-plus-subprototypes":
        print(f"Sub-prototypes per speaker: {args.num_sub_prototypes}")
    print(f"Model version: {model_version}")
    print(f"SQLite DB root: {database_run_root}")
    print(f"Total speakers: {len(dataset)}")
    print(f"Total audio files: {sum(len(files) for files in dataset.values())}")

    embeddings = _extract_embeddings(
        dataset,
        model=model,
        preprocessor=preprocessor,
        device=device,
        verbose=args.verbose,
    )
    extractor = CachedEmbeddingExtractor(
        embeddings,
        model_version=model_version,
        embedding_dimension=emb_dim,
    )

    results = [
        _evaluate_k(
            k=k,
            dataset=dataset,
            extractor=extractor,
            enrolled_speakers=enrolled_speakers,
            dataset_dir=dataset_dir,
            operating_threshold=args.threshold,
            database_path=database_run_root / f"k_{k:02d}" / "speakers.db",
            prototype_mode=args.prototype_mode,
            num_sub_prototypes=args.num_sub_prototypes,
            kmeans_max_iters=args.kmeans_max_iters,
            kmeans_seed=args.kmeans_seed,
        )
        for k in range(1, args.max_enroll_count + 1)
    ]

    report = {
        "experiment": "ecapa_tdnn_protonet_kshot_open_set",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(model_path),
            "model_version": model_version,
            "architecture": "ECAPATDNNBackbone",
            "embedding_dim": emb_dim,
            "channels": channels,
            "n_mels": n_mels,
        },
        "dataset": {
            "path": str(dataset_dir),
            "speaker_count": len(dataset),
            "audio_file_count": sum(len(files) for files in dataset.values()),
            "speaker_file_counts": {speaker: len(files) for speaker, files in sorted(dataset.items())},
        },
        "selection": {
            "num_enrolled_speakers": args.num_enrolled_speakers,
            "speaker_selection": "explicit" if args.enrolled_speakers else args.speaker_selection,
            "sample_seed": None if args.speaker_selection == "first" or args.enrolled_speakers else args.sample_seed,
            "enrolled_speakers": enrolled_speakers,
            "non_enrolled_speakers": non_enrolled_speakers,
        },
        "preprocessing": {
            "sample_rate": preprocess_config.sample_rate,
            "duration": preprocess_config.duration,
            "n_mels": preprocess_config.n_mels,
            "n_fft": preprocess_config.n_fft,
            "hop_length": preprocess_config.hop_length,
            "vad_enabled": preprocess_config.vad_enabled,
            "vad_top_db": preprocess_config.vad_top_db,
            "vad_frame_length": preprocess_config.vad_frame_length,
            "vad_hop_length": preprocess_config.vad_hop_length,
            "normalization": "per-sample mean/std after mel dB conversion",
        },
        "prototype_strategy": {
            "mode": args.prototype_mode,
            "primary_prototype": "Stored SQLite profile produced by EnrollmentService/create_speaker_prototype.",
            "sub_prototypes": (
                "Disabled"
                if args.prototype_mode == "primary"
                else (
                    f"{args.num_sub_prototypes} KMeans centers from enrollment embeddings; "
                    "recognition score is max cosine over primary + sub-prototypes."
                )
            ),
            "num_sub_prototypes": 0
            if args.prototype_mode == "primary"
            else args.num_sub_prototypes,
            "prototypes_per_speaker": 1
            if args.prototype_mode == "primary"
            else 1 + args.num_sub_prototypes,
            "kmeans_max_iters": args.kmeans_max_iters,
            "kmeans_seed": args.kmeans_seed,
        },
        "service_pipeline": {
            "speaker_app_files_referenced": [
                "speaker_app/speaker_app/services/enrollment_service.py",
                "speaker_app/speaker_app/services/recognition_service.py",
                "speaker_app/speaker_app/services/speaker_repository.py",
                "speaker_app/speaker_app/services/embedding_math.py",
                "speaker_app/speaker_app/model/embedding_extractor.py",
            ],
            "enrollment": "EnrollmentService.enroll writes create_speaker_prototype output to SQLite SpeakerRepository for each k-shot run.",
            "recognition": "Queries load compatible SpeakerRepository profiles; this experiment can additionally score KMeans sub-prototypes and applies the same threshold rule to the best speaker score.",
            "database_run_root": str(database_run_root),
        },
        "metric_definitions": {
            "decision_rule": "accept predicted_speaker if best cosine similarity across that speaker's prototype bank >= threshold; otherwise unknown",
            "operating_threshold": args.threshold,
            "accuracy": "open-set accuracy over all query clips: genuine must be accepted as the true enrolled speaker; impostors must be rejected",
            "GAR": "genuine accept rate: genuine clips accepted as the true enrolled speaker / genuine query clips",
            "FAR": "false accept rate: impostor clips accepted as any enrolled speaker / impostor query clips",
            "FNR": "false negative rate: genuine clips not accepted as the true enrolled speaker / genuine query clips; this includes rejects and wrong accepts",
            "EER": "threshold where FAR and FNR are closest/interpolated using the open-set decision rule above",
        },
        "best_k": _top_level_best(results),
        "k_shot_results": results,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(f"Fixed operating threshold = {args.threshold:.4f}")
    print("k  accuracy  GAR      FAR      FNR      EER")
    for row in results:
        metrics = row["metrics_at_operating_threshold"]
        print(
            f"{row['k']:>1}  "
            f"{metrics['accuracy']:.4f}    "
            f"{metrics['GAR']:.4f}   "
            f"{metrics['FAR']:.4f}   "
            f"{metrics['FNR']:.4f}   "
            f"{row['eer']['eer']:.4f}"
        )
    print(f"\nJSON: {json_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
