from __future__ import annotations

import argparse
from pathlib import Path

import torch

from PrototypicalNetwork.train import *
from utils.general import *
from utils.paths import get_default_librispeech_root, get_project_root

N_WAY = 5
K_SHOT = 5
N_QUERY = 15
N_VAL_EPISODES = 2
N_TEST_EPISODES = 50
PROJECT_ROOT = get_project_root()
MODEL_PATH = PROJECT_ROOT / "output" / "ecapa_tdnn_protonet_model.pth"
PLOT_PATH = PROJECT_ROOT / "output" / "ecapa_tdnn_protonet_curves.png"
DET_CURVE_PATH = PROJECT_ROOT / "output" / "ecapa_tdnn_protonet_det_curve_test.png"
DATASET_ROOT = get_default_librispeech_root()
LOCAL_DATASET_ROOT = PROJECT_ROOT / "audio_data"
BACKBONE = "ecapa"  # Options: ecapa, xvector
EMBEDDING_DIM = 192
XVECTOR_TDNN_CHANNELS = 512
XVECTOR_STATS_CHANNELS = 1500
XVECTOR_DROPOUT = 0.1

TRAIN_MODE = False
NUM_SPEAKERS = 40
TRAIN_RATIO = 0.0
VAL_RATIO = 0.0
TEST_RATIO = 1.0
MIN_SAMPLES_PER_SPEAKER = 20
MAX_SAMPLES_PER_SPEAKER = None
N_EPISODES = 500
TRAIN_AUGMENT = True
EVAL_SEED = 36
VAD_ENABLED = True
VAD_TOP_DB = 10.0
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 258
TRAINING_LOSS_MODE = "hybrid"  # Options: "angular_proto", "aam_softmax", "hybrid"
PROTO_SCALE = 30.0
PROTO_MARGIN = 0.15
AAM_SCALE = 30.0
AAM_MARGIN = 0.15
HYBRID_PROTO_WEIGHT = 0.8
HYBRID_AAM_WEIGHT = 0.2
PROTO_LR = 1e-4
PROTO_WEIGHT_DECAY = 0.01
REQUIRE_CUDA = True


def _default_dataset_root() -> Path:
    # if LOCAL_DATASET_ROOT.exists():
    #     return LOCAL_DATASET_ROOT

    return DATASET_ROOT / "test-clean"


def _detect_audio_ext(dataset_root: Path) -> str:
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        if next(dataset_root.rglob(f"*{ext}"), None) is not None:
            return ext
    return ".flac"


def main() -> None:
    dataset_root = _default_dataset_root().resolve()
    audio_ext = _detect_audio_ext(dataset_root)

    dataset_list, label_map = build_prototypical_dataset(
        root=dataset_root,
        num_speakers=NUM_SPEAKERS,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        min_samples_per_speaker=MIN_SAMPLES_PER_SPEAKER,
        max_samples_per_speaker=MAX_SAMPLES_PER_SPEAKER,
        ext=audio_ext,
        seed=42,
    )

    unique_label_ids = set(item["label_id"] for item in dataset_list)
    n_speakers = len(unique_label_ids)

    print(f"Train mode: {TRAIN_MODE}")
    print(f"Dataset root: {dataset_root}")
    print(f"Audio extension: {audio_ext}")
    print(f"Dataset size: {len(dataset_list)}")
    print(f"Number of speakers: {n_speakers}")
    print(f"Backbone: {BACKBONE}")
    print("Initial checkpoint: None")
    print(f"Unique labels: {set(item['label'] for item in dataset_list)}")

    train_samples = [item for item in dataset_list if item["split"] == "train"]
    val_samples = [item for item in dataset_list if item["split"] == "valid"]
    test_samples = [item for item in dataset_list if item["split"] == "test"]

    print("\nData Split Summary:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Valid: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")

    train_prototypical_network(
        dataset_list=dataset_list,
        train_mode=TRAIN_MODE,
        n_way=N_WAY,
        k_shot=K_SHOT,
        n_query=N_QUERY,
        n_episodes=N_EPISODES,
        n_val_episodes=N_VAL_EPISODES,
        n_test_episodes=N_TEST_EPISODES,
        sr=16000,
        n_mels=80,
        duration=5.0,
        vad_enabled=VAD_ENABLED,
        vad_top_db=VAD_TOP_DB,
        vad_frame_length=VAD_FRAME_LENGTH,
        vad_hop_length=VAD_HOP_LENGTH,
        train_augment=TRAIN_AUGMENT,
        training_loss_mode=TRAINING_LOSS_MODE,
        backbone=BACKBONE,
        embedding_dim=EMBEDDING_DIM,
        xvector_tdnn_channels=XVECTOR_TDNN_CHANNELS,
        xvector_stats_channels=XVECTOR_STATS_CHANNELS,
        xvector_dropout=XVECTOR_DROPOUT,
        proto_scale=PROTO_SCALE,
        proto_margin=PROTO_MARGIN,
        aam_scale=AAM_SCALE,
        aam_margin=AAM_MARGIN,
        hybrid_proto_weight=HYBRID_PROTO_WEIGHT,
        hybrid_aam_weight=HYBRID_AAM_WEIGHT,
        lr=PROTO_LR,
        weight_decay=PROTO_WEIGHT_DECAY,
        init_checkpoint_path=None,
        require_cuda=REQUIRE_CUDA,
        augmentation_probability=0.2,
        augmentation_rir_dir=PROJECT_ROOT / "rirs_noises" / "RIRS_NOISES" / "real_rirs_isotropic_noises",
        augmentation_kwargs={
            "snr_range": (5, 20),
            "waveform_dropout_range": (0.05, 0.15),
            "freq_dropout_range": (0.05, 0.15),
            "shift_range": (-0.2, 0.2),
            "speed_range": (0.9, 1.1),
        },
        model_path=MODEL_PATH,
        eval_seed=EVAL_SEED,
        plot_path=PLOT_PATH,
        det_curve_path=DET_CURVE_PATH,
    )


if __name__ == "__main__":
    main()
