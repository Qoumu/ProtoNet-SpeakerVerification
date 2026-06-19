from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from PrototypicalNetwork.train import (
    AAMSoftmaxLoss,
    _compute_eer,
    _get_tqdm,
    _plot_det_curve,
    _select_training_device,
)
from model.ECAPATDNN import ECAPATDNNBackbone
from utils.data_augmentation import DataAugmentation
from utils.data_preprocessing import SpeakerDataset
from utils.general import build_prototypical_dataset
from utils.paths import get_default_librispeech_root, get_project_root


PROJECT_ROOT = get_project_root()


def _default_dataset_root() -> Path:
    audio_data_root = PROJECT_ROOT / "audio_data"
    if audio_data_root.exists():
        return audio_data_root

    librispeech_train = get_default_librispeech_root() / "train-clean-100"
    if librispeech_train.exists():
        return librispeech_train

    return audio_data_root


def _detect_audio_ext(dataset_root: Path) -> str:
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        if next(dataset_root.rglob(f"*{ext}"), None) is not None:
            return ext
    return ".flac"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ECAPA-TDNN with supervised AAM-Softmax, without prototypical episodes.",
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
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--aam-scale", type=float, default=30.0)
    parser.add_argument("--aam-margin", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-pairs", type=int, default=20000)
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "output" / "ecapa_tdnn_aam_model.pth")
    parser.add_argument("--plot-path", type=Path, default=PROJECT_ROOT / "output" / "ecapa_tdnn_aam_curves.png")
    parser.add_argument("--det-curve-path", type=Path, default=PROJECT_ROOT / "output" / "ecapa_tdnn_aam_det_curve.png")
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


class _WaveformSpeakerDataset(SpeakerDataset):
    """Return waveform batches so mel extraction can run on the target device."""

    def __getitem__(self, idx):
        return self.load_waveform_tensor(idx)


def _remap_labels(labels: torch.Tensor, label_map: dict[int, int]) -> torch.Tensor:
    mapped = [label_map[int(label)] for label in labels.detach().cpu().tolist()]
    return torch.tensor(mapped, device=labels.device, dtype=torch.long)


def _make_dataset(
    samples: list[dict],
    args: argparse.Namespace,
    *,
    augment: bool,
) -> SpeakerDataset:
    augmenter = None
    if augment:
        augmenter = DataAugmentation(
            sample_rate=args.sr,
            rir_dir=str(args.augmentation_rir_dir),
            p=args.augmentation_probability,
            snr_range=(5, 20),
            waveform_dropout_range=(0.05, 0.15),
            freq_dropout_range=(0.05, 0.15),
            shift_range=(-0.2, 0.2),
            speed_range=(0.9, 1.1),
        )

    return _WaveformSpeakerDataset(
        samples,
        sr=args.sr,
        n_mels=args.n_mels,
        duration=args.duration,
        augment=augment,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        waveform_augmenter=augmenter,
        vad_enabled=args.vad_enabled,
        vad_top_db=args.vad_top_db,
        vad_frame_length=args.vad_frame_length,
        vad_hop_length=args.vad_hop_length,
    )


def _extract_embeddings(
    model: ECAPATDNNBackbone,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        dataset = loader.dataset
        for batch_inputs, batch_labels in loader:
            if batch_inputs.dim() == 2 and hasattr(dataset, "waveforms_to_mels"):
                mels = dataset.waveforms_to_mels(batch_inputs, device)
            else:
                mels = batch_inputs.to(device)
            embeddings.append(model(mels).cpu())
            labels.append(batch_labels.cpu())

    if not embeddings:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def _sample_verification_scores(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if embeddings.size(0) < 2:
        raise ValueError("Need at least two samples for verification evaluation.")

    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels_np = labels.numpy()
    rng = random.Random(seed)

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels_np.tolist()):
        by_label.setdefault(int(label), []).append(idx)

    if len(by_label) < 2:
        raise ValueError("Need at least two speakers for verification evaluation.")

    all_pair_count = embeddings.size(0) * (embeddings.size(0) - 1) // 2
    pairs: list[tuple[int, int, int]] = []

    if all_pair_count <= max_pairs:
        for i in range(embeddings.size(0)):
            for j in range(i + 1, embeddings.size(0)):
                pairs.append((i, j, int(labels_np[i] == labels_np[j])))
    else:
        positive_labels = [label for label, idxs in by_label.items() if len(idxs) >= 2]
        if not positive_labels:
            raise ValueError("Need at least one speaker with two samples for positive pairs.")

        target_pos = max_pairs // 2
        target_neg = max_pairs - target_pos
        speaker_ids = list(by_label.keys())

        for _ in range(target_pos):
            label = rng.choice(positive_labels)
            i, j = rng.sample(by_label[label], 2)
            pairs.append((i, j, 1))

        for _ in range(target_neg):
            label_i, label_j = rng.sample(speaker_ids, 2)
            i = rng.choice(by_label[label_i])
            j = rng.choice(by_label[label_j])
            pairs.append((i, j, 0))

    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    scores = (embeddings[left] * embeddings[right]).sum(dim=1).numpy()
    pair_labels = np.asarray([pair[2] for pair in pairs], dtype=np.int32)
    return scores, pair_labels


def _evaluate_verification(
    model: ECAPATDNNBackbone,
    loader: DataLoader,
    device: torch.device,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    embeddings, labels = _extract_embeddings(model, loader, device)
    scores, pair_labels = _sample_verification_scores(
        embeddings,
        labels,
        max_pairs=max_pairs,
        seed=seed,
    )
    eer, far, fnr, _, eer_threshold, eer_far, eer_fnr = _compute_eer(scores, pair_labels)
    return eer, far, fnr, eer_far, eer_fnr


def _plot_curves(history: dict[str, list[float]], output_path: Path) -> None:
    if not history["epoch"]:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], marker="o", label="Train Acc")
    if history["val_eer"] and np.isfinite(np.asarray(history["val_eer"], dtype=np.float64)).any():
        plt.plot(history["epoch"], [eer * 100.0 for eer in history["val_eer"]], marker="s", label="Val EER")
    plt.xlabel("Epoch")
    plt.ylabel("Percent")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Training curves saved to {output_path}")


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
    print("Training mode: ECAPA-TDNN + AAM-Softmax, no prototypical episodes")

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

    model = ECAPATDNNBackbone(
        n_mels=args.n_mels,
        channels=args.channels,
        emb_dim=args.embedding_dim,
    ).to(device)
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
            batch_iter = tqdm(train_loader, desc=f"ECAPA epoch {epoch}/{args.epochs}", leave=False)

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
