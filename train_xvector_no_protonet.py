from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from PrototypicalNetwork.train import AAMSoftmaxLoss, _get_tqdm, _plot_det_curve, _select_training_device
from model.XVector import XVectorBackbone
from train_ecapa_tdnn_no_protonet import (
    _default_dataset_root,
    _detect_audio_ext,
    _evaluate_verification,
    _make_dataset,
    _plot_curves,
    _remap_labels,
)
from utils.general import build_prototypical_dataset
from utils.paths import get_project_root


PROJECT_ROOT = get_project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train x-vector with supervised AAM-Softmax, without prototypical episodes.",
    )
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--ext", type=str, default=None)
    parser.add_argument("--num-speakers", type=int, default=250)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-samples-per-speaker", type=int, default=15)
    parser.add_argument("--max-samples-per-speaker", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--tdnn-channels", type=int, default=512)
    parser.add_argument("--stats-channels", type=int, default=1500)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aam-scale", type=float, default=30.0)
    parser.add_argument("--aam-margin", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-pairs", type=int, default=20000)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "output" / "xvector_aam_model.pth")
    parser.add_argument("--plot-path", type=Path, default=PROJECT_ROOT / "output" / "xvector_aam_curves.png")
    parser.add_argument("--det-curve-path", type=Path, default=PROJECT_ROOT / "output" / "xvector_aam_det_curve.png")
    parser.add_argument("--augmentation-probability", type=float, default=0.3)
    parser.add_argument(
        "--augmentation-rir-dir",
        type=Path,
        default=PROJECT_ROOT / "rirs_noises" / "RIRS_NOISES" / "real_rirs_isotropic_noises",
    )
    parser.add_argument("--vad-enabled", action="store_true")
    parser.add_argument("--vad-top-db", type=float, default=10.0)
    parser.add_argument("--vad-frame-length", type=int, default=2048)
    parser.add_argument("--vad-hop-length", type=int, default=258)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    train_data = [item for item in dataset_list if item["split"] == "train"]
    val_data = [item for item in dataset_list if item["split"] == "valid"]
    test_data = [item for item in dataset_list if item["split"] == "test"]
    train_speaker_ids = sorted({item["label_id"] for item in train_data})
    classifier_label_map = {label_id: idx for idx, label_id in enumerate(train_speaker_ids)}

    if len(train_speaker_ids) < 2:
        raise ValueError("AAM-Softmax training needs at least two train speakers.")

    device = _select_training_device(require_cuda=not args.allow_cpu)
    print(f"Dataset root: {dataset_root}")
    print(f"Audio extension: {audio_ext}")
    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print("Training mode: x-vector + AAM-Softmax, no prototypical episodes")

    train_dataset = _make_dataset(train_data, args, augment=not args.no_train_augment)
    val_dataset = _make_dataset(val_data, args, augment=False)
    test_dataset = _make_dataset(test_data, args, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_batch_size = max(2, args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = XVectorBackbone(
        n_mels=args.n_mels,
        tdnn_channels=args.tdnn_channels,
        stats_channels=args.stats_channels,
        emb_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"x-vector checkpoint not found: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
        print(f"Loaded initial weights from: {args.checkpoint}")

    aam_loss = AAMSoftmaxLoss(
        embedding_dim=args.embedding_dim,
        num_classes=len(train_speaker_ids),
        scale=args.aam_scale,
        margin=args.aam_margin,
    ).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(aam_loss.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    best_val_eer = float("inf")
    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "train_acc": [], "val_eer": []}
    tqdm = _get_tqdm()

    for epoch in range(1, args.epochs + 1):
        model.train()
        aam_loss.train()
        losses: list[float] = []
        accs: list[float] = []

        batch_iter = train_loader
        if not args.no_progress and tqdm is not None:
            batch_iter = tqdm(train_loader, desc=f"x-vector epoch {epoch}/{args.epochs}", leave=False)

        for waveforms, labels in batch_iter:
            if waveforms.size(0) < 2:
                continue

            mels = train_dataset.waveforms_to_mels(waveforms, device)
            labels = _remap_labels(labels.to(device), classifier_label_map)

            optimizer.zero_grad()
            embeddings = model(mels)
            loss, acc = aam_loss(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            accs.append(float(acc.item() * 100.0))
            if not args.no_progress and tqdm is not None:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item() * 100.0:.2f}%")

        if not losses:
            raise RuntimeError("No training batches were processed. Check batch size and dataset size.")

        avg_train_loss = float(np.mean(losses))
        avg_train_acc = float(np.mean(accs))
        history["epoch"].append(float(epoch))
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        val_eer = None
        if len(val_data) > 1:
            try:
                val_eer, _, _, _, _ = _evaluate_verification(
                    model,
                    val_loader,
                    device,
                    max_pairs=args.max_eval_pairs,
                    seed=args.seed + epoch,
                )
                history["val_eer"].append(val_eer)
                scheduler.step(val_eer)
            except ValueError as exc:
                print(f"[WARN] Validation EER skipped: {exc}")
                history["val_eer"].append(float("nan"))
                scheduler.step(avg_train_loss)
        else:
            history["val_eer"].append(float("nan"))
            scheduler.step(avg_train_loss)

        if val_eer is not None:
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, "
                f"Val EER: {val_eer * 100:.2f}%"
            )
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                torch.save(model.state_dict(), args.output_model)
                print(f"  Saved best model: {args.output_model}")
        else:
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%"
            )
            if epoch == 1:
                torch.save(model.state_dict(), args.output_model)

    _plot_curves(history, args.plot_path)

    if args.output_model.exists():
        model.load_state_dict(torch.load(args.output_model, map_location=device))

    if len(test_data) > 1:
        try:
            test_eer, far, fnr, eer_far, eer_fnr = _evaluate_verification(
                model,
                test_loader,
                device,
                max_pairs=args.max_eval_pairs,
                seed=args.seed + 10_000,
            )
            print(f"\nTest EER: {test_eer * 100:.2f}%")
            _plot_det_curve(far, fnr, test_eer, eer_far, eer_fnr, args.det_curve_path)
        except ValueError as exc:
            print(f"[WARN] Test EER skipped: {exc}")

    print(f"\nTraining completed. Backbone checkpoint: {args.output_model.resolve()}")


if __name__ == "__main__":
    main()
