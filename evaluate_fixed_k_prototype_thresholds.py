from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from evaluate_kshot_open_set import (
    AudioPreprocessor,
    CachedEmbeddingExtractor,
    PreprocessConfig,
    _build_model,
    _evaluate_k,
    _extract_embeddings,
    _load_dataset,
    _select_speakers,
    _summarize_at_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed 6-shot enrollment with one prototype per speaker versus "
            "one primary prototype plus three KMeans sub-prototypes."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("audio_data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("output/ecapa_tdnn_protonet_model.pth"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/ecapa_tdnn_protonet_fixed6_prototype_threshold_eval.json"),
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("output/ecapa_tdnn_protonet_fixed6_prototype_threshold_dbs"),
    )
    parser.add_argument(
        "--enroll-count",
        type=int,
        default=6,
        help="Fixed number of enrollment clips per enrolled speaker.",
    )
    parser.add_argument(
        "--enrolled-speakers",
        nargs="+",
        default=["spk03", "spk04", "spk06", "spk07", "spk11", "spk12"],
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--num-sub-prototypes", type=int, default=3)
    parser.add_argument("--kmeans-max-iters", type=int, default=50)
    parser.add_argument("--kmeans-seed", type=int, default=0)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=None)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--vad-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vad-top-db", type=float, default=10.0)
    parser.add_argument("--vad-frame-length", type=int, default=2048)
    parser.add_argument("--vad-hop-length", type=int, default=258)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _table_row(
    *,
    prototype_label: str,
    prototypes_per_speaker: int,
    threshold: float,
    result: dict[str, object],
) -> dict[str, object]:
    metrics = _summarize_at_threshold(result["trials"], threshold=threshold)
    return {
        "prototype_setting": prototype_label,
        "prototypes_per_speaker": prototypes_per_speaker,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "GAR": metrics["GAR"],
        "FAR": metrics["FAR"],
        "FNR": metrics["FNR"],
        "EER": result["eer"]["eer"],
        "eer_threshold": result["eer"]["threshold"],
        "counts": metrics["counts"],
    }


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()
    json_out = args.json_out.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    database_run_root = args.database_root.resolve() / run_id

    if args.enroll_count < 1:
        raise ValueError("--enroll-count must be >= 1")
    if args.num_sub_prototypes < 1:
        raise ValueError("--num-sub-prototypes must be >= 1")
    for threshold in args.thresholds:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("All thresholds must be between -1 and 1")

    device = _select_device(args.device)
    dataset = _load_dataset(dataset_dir)
    model, inferred_n_mels, channels, emb_dim = _build_model(model_path, device)
    n_mels = inferred_n_mels if args.n_mels is None else args.n_mels
    if n_mels != inferred_n_mels:
        raise ValueError(
            f"--n-mels={n_mels} does not match checkpoint input bins ({inferred_n_mels})."
        )

    enrolled_speakers = _select_speakers(
        dataset,
        explicit_speakers=args.enrolled_speakers,
        num_enrolled=len(args.enrolled_speakers),
        max_enroll_count=args.enroll_count,
        selection="first",
        seed=0,
    )
    non_enrolled_speakers = sorted(speaker for speaker in dataset if speaker not in enrolled_speakers)
    model_version = args.model_version or f"ecapa-tdnn:{model_path.name}"

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
    print(f"Fixed k-shot: {args.enroll_count}")
    print(f"Thresholds: {args.thresholds}")
    print(f"SQLite DB root: {database_run_root}")

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

    run_specs = [
        ("1_prototype", "primary", 1),
        ("4_prototypes", "primary-plus-subprototypes", 1 + args.num_sub_prototypes),
    ]
    prototype_runs: dict[str, object] = {}
    comparison_table: list[dict[str, object]] = []

    for label, mode, prototypes_per_speaker in run_specs:
        result = _evaluate_k(
            k=args.enroll_count,
            dataset=dataset,
            extractor=extractor,
            enrolled_speakers=enrolled_speakers,
            dataset_dir=dataset_dir,
            operating_threshold=args.thresholds[0],
            database_path=database_run_root / label / "speakers.db",
            prototype_mode=mode,
            num_sub_prototypes=args.num_sub_prototypes,
            kmeans_max_iters=args.kmeans_max_iters,
            kmeans_seed=args.kmeans_seed,
        )
        prototype_runs[label] = result
        for threshold in args.thresholds:
            comparison_table.append(
                _table_row(
                    prototype_label=label,
                    prototypes_per_speaker=prototypes_per_speaker,
                    threshold=threshold,
                    result=result,
                )
            )

    report = {
        "experiment": "ecapa_tdnn_protonet_fixed6_prototype_threshold_comparison",
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
            "enroll_count": args.enroll_count,
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
        },
        "prototype_strategy": {
            "baseline": "1 primary prototype saved by EnrollmentService/create_speaker_prototype.",
            "multi_prototype": (
                "1 primary prototype plus "
                f"{args.num_sub_prototypes} KMeans sub-prototypes from the same "
                f"{args.enroll_count} enrollment embeddings; score is max cosine."
            ),
            "kmeans_max_iters": args.kmeans_max_iters,
            "kmeans_seed": args.kmeans_seed,
        },
        "thresholds": args.thresholds,
        "comparison_table": comparison_table,
        "prototype_runs": prototype_runs,
        "database_run_root": str(database_run_root),
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary:")
    print("prototypes  threshold  accuracy  GAR      FAR      FNR      EER")
    for row in comparison_table:
        print(
            f"{row['prototypes_per_speaker']:>10}  "
            f"{row['threshold']:.1f}       "
            f"{row['accuracy']:.4f}    "
            f"{row['GAR']:.4f}   "
            f"{row['FAR']:.4f}   "
            f"{row['FNR']:.4f}   "
            f"{row['EER']:.4f}"
        )
    print(f"\nJSON: {json_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
