from __future__ import annotations

import argparse
from pathlib import Path

from PrototypicalNetwork.train import train_prototypical_network
from utils.general import build_prototypical_dataset
from utils.paths import get_default_librispeech_root, get_project_root


PROJECT_ROOT = get_project_root()


def _default_dataset_root() -> Path:
    audio_data_root = PROJECT_ROOT / "audio_data"
    if audio_data_root.exists():
        return audio_data_root

    librispeech_root = get_default_librispeech_root()
    train_clean_root = librispeech_root / "train-clean-100"
    if train_clean_root.exists():
        return train_clean_root

    return audio_data_root


def _detect_audio_ext(dataset_root: Path) -> str:
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        if next(dataset_root.rglob(f"*{ext}"), None) is not None:
            return ext
    return ".flac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the prototypical ECAPA-TDNN model from an existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the existing checkpoint used to initialize fine-tuning.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Root directory containing speaker audio files.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None,
        help="Audio file extension to scan, for example .wav or .flac. Default: auto-detect.",
    )
    parser.add_argument("--num-speakers", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--min-samples-per-speaker", type=int, default=15)
    parser.add_argument("--max-samples-per-speaker", type=int, default=None)
    parser.add_argument("--n-way", type=int, default=10)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--val-episodes", type=int, default=2)
    parser.add_argument("--test-episodes", type=int, default=None)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument(
        "--proto-scale",
        type=float,
        default=30.0,
        help="Cosine scale used in the prototypical loss.",
    )
    parser.add_argument(
        "--proto-margin",
        type=float,
        default=0.2,
        help="Cosine margin applied to target logits in the prototypical loss.",
    )
    parser.add_argument(
        "--augmentation-probability",
        type=float,
        default=0.3,
        help="Probability of applying waveform augmentation during training.",
    )
    parser.add_argument(
        "--augmentation-rir-dir",
        type=Path,
        default=PROJECT_ROOT / "rirs_noises" / "RIRS_NOISES" / "real_rirs_isotropic_noises",
        help="Directory containing RIR/noise assets for augmentation.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=PROJECT_ROOT / "output" / "ECAPATDNN_protonet_finetuned.pth",
        help="Path to save the best fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "ECAPATDNN_protonet_finetuned_curves.png",
        help="Path to save training curves.",
    )
    parser.add_argument(
        "--det-curve-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "ECAPATDNN_protonet_finetuned_det_curve.png",
        help="Path to save the DET curve.",
    )
    parser.add_argument("--eval-seed", type=int, default=67)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-train-augment",
        action="store_true",
        help="Disable waveform augmentation during fine-tuning.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    audio_ext = args.ext or _detect_audio_ext(dataset_root)

    dataset_list, _ = build_prototypical_dataset(
        root=dataset_root,
        num_speakers=args.num_speakers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_samples_per_speaker=args.min_samples_per_speaker,
        max_samples_per_speaker=args.max_samples_per_speaker,
        ext=audio_ext,
        seed=args.seed,
    )

    unique_label_ids = {item["label_id"] for item in dataset_list}
    print(f"Fine-tune checkpoint: {args.checkpoint}")
    print(f"Dataset root: {dataset_root}")
    print(f"Audio extension: {audio_ext}")
    print(f"Dataset size: {len(dataset_list)}")
    print(f"Number of speakers: {len(unique_label_ids)}")

    train_prototypical_network(
        dataset_list=dataset_list,
        train_mode=True,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episodes=args.episodes,
        n_val_episodes=args.val_episodes,
        n_test_episodes=args.test_episodes,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        train_augment=not args.no_train_augment,
        augmentation_probability=args.augmentation_probability,
        augmentation_rir_dir=args.augmentation_rir_dir,
        augmentation_kwargs={
            "snr_range": (5, 20),
            "waveform_dropout_range": (0.05, 0.15),
            "freq_dropout_range": (0.05, 0.15),
            "shift_range": (-0.2, 0.2),
            "speed_range": (0.9, 1.1),
        },
        show_progress=not args.no_progress,
        proto_scale=args.proto_scale,
        proto_margin=args.proto_margin,
        init_checkpoint_path=args.checkpoint,
        model_path=args.output_model,
        eval_seed=args.eval_seed,
        plot_path=args.plot_path,
        det_curve_path=args.det_curve_path,
    )


if __name__ == "__main__":
    main()
