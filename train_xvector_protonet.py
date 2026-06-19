from __future__ import annotations

import argparse
from pathlib import Path

from PrototypicalNetwork.train import train_prototypical_network
from utils.general import build_prototypical_dataset
from utils.paths import get_default_librispeech_root, get_project_root


PROJECT_ROOT = get_project_root()


def _default_dataset_root() -> Path:
    return get_default_librispeech_root() / "train-clean-100"


def _detect_audio_ext(dataset_root: Path) -> str:
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        if next(dataset_root.rglob(f"*{ext}"), None) is not None:
            return ext
    return ".flac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an x-vector backbone with prototypical-network episodes.",
    )
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--ext", type=str, default=None)
    parser.add_argument("--num-speakers", type=int, default=250)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-samples-per-speaker", type=int, default=20)
    parser.add_argument("--max-samples-per-speaker", type=int, default=None)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--n-query", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--val-episodes", type=int, default=2)
    parser.add_argument("--test-episodes", type=int, default=50)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--tdnn-channels", type=int, default=512)
    parser.add_argument("--stats-channels", type=int, default=1500)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--training-loss-mode",
        choices=("angular_proto", "aam_softmax", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--proto-scale", type=float, default=30.0)
    parser.add_argument("--proto-margin", type=float, default=0.15)
    parser.add_argument("--aam-scale", type=float, default=30.0)
    parser.add_argument("--aam-margin", type=float, default=0.15)
    parser.add_argument("--hybrid-proto-weight", type=float, default=0.8)
    parser.add_argument("--hybrid-aam-weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "output" / "xvector_protonet_model.pth")
    parser.add_argument("--plot-path", type=Path, default=PROJECT_ROOT / "output" / "xvector_protonet_curves.png")
    parser.add_argument("--det-curve-path", type=Path, default=PROJECT_ROOT / "output" / "xvector_protonet_det_curve.png")
    parser.add_argument("--augmentation-probability", type=float, default=0.2)
    parser.add_argument(
        "--augmentation-rir-dir",
        type=Path,
        default=PROJECT_ROOT / "rirs_noises" / "RIRS_NOISES" / "real_rirs_isotropic_noises",
    )
    parser.add_argument("--vad-enabled", dest="vad_enabled", action="store_true")
    parser.add_argument("--no-vad", dest="vad_enabled", action="store_false")
    parser.set_defaults(vad_enabled=True)
    parser.add_argument("--vad-top-db", type=float, default=10.0)
    parser.add_argument("--vad-frame-length", type=int, default=2048)
    parser.add_argument("--vad-hop-length", type=int, default=258)
    parser.add_argument("--eval-seed", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training if CUDA is unavailable. By default training requires CUDA.",
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

    print(f"Dataset root: {dataset_root}")
    print(f"Audio extension: {audio_ext}")
    print(f"Dataset size: {len(dataset_list)}")

    train_prototypical_network(
        dataset_list=dataset_list,
        train_mode=True,
        backbone="xvector",
        embedding_dim=args.embedding_dim,
        xvector_tdnn_channels=args.tdnn_channels,
        xvector_stats_channels=args.stats_channels,
        xvector_dropout=args.dropout,
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
        vad_enabled=args.vad_enabled,
        vad_top_db=args.vad_top_db,
        vad_frame_length=args.vad_frame_length,
        vad_hop_length=args.vad_hop_length,
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
        training_loss_mode=args.training_loss_mode,
        proto_scale=args.proto_scale,
        proto_margin=args.proto_margin,
        aam_scale=args.aam_scale,
        aam_margin=args.aam_margin,
        hybrid_proto_weight=args.hybrid_proto_weight,
        hybrid_aam_weight=args.hybrid_aam_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        init_checkpoint_path=args.checkpoint,
        require_cuda=not args.allow_cpu,
        model_path=args.output_model,
        eval_seed=args.eval_seed,
        plot_path=args.plot_path,
        det_curve_path=args.det_curve_path,
    )


if __name__ == "__main__":
    main()
