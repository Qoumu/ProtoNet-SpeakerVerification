from __future__ import annotations

import argparse
from pathlib import Path

import torch

from PrototypicalNetwork.train import *
from utils.general import *
from utils.paths import get_default_librispeech_root, get_project_root

N_WAY = 10
K_SHOT = 5
N_QUERY = 20
N_VAL_EPISODES = 2
N_TEST_EPISODES = None
PROJECT_ROOT = get_project_root()
MODEL_PATH = PROJECT_ROOT / "output" / "ECAPATDNN_protonet_model.pth"
PLOT_PATH = PROJECT_ROOT / "output" / "ECAPATDNN_protonet_curves.png"
DET_CURVE_PATH = PROJECT_ROOT / "output" / "ECAPATDNN_protonet_det_curve.png"
DATASET_ROOT = get_default_librispeech_root()

TRAIN_MODE = True
NUM_SPEAKERS = 250
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
MIN_SAMPLES_PER_SPEAKER = 15
MAX_SAMPLES_PER_SPEAKER = None
N_EPISODES = 500
TRAIN_AUGMENT = True
EVAL_SEED = 67
VAD_ENABLED = True
VAD_TOP_DB = 30.0
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 512
TRAINING_LOSS_MODE = "aam_softmax"
PROTO_SCALE = 30.0
PROTO_MARGIN = 0.2
AAM_SCALE = 30.0
AAM_MARGIN = 0.2
HYBRID_PROTO_WEIGHT = 1.0
HYBRID_AAM_WEIGHT = 1.0


def main() -> None:
    dataset_list, label_map = build_prototypical_dataset(
        root=DATASET_ROOT / "train-clean-100",
        num_speakers=NUM_SPEAKERS,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        min_samples_per_speaker=MIN_SAMPLES_PER_SPEAKER,
        max_samples_per_speaker=MAX_SAMPLES_PER_SPEAKER,
        seed=42,
    )

    unique_label_ids = set(item["label_id"] for item in dataset_list)
    n_speakers = len(unique_label_ids)

    print(f"Train mode: {TRAIN_MODE}")
    print(f"Dataset root: {DATASET_ROOT / 'train-clean-100'}")
    print(f"Dataset size: {len(dataset_list)}")
    print(f"Number of speakers: {n_speakers}")
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
        proto_scale=PROTO_SCALE,
        proto_margin=PROTO_MARGIN,
        aam_scale=AAM_SCALE,
        aam_margin=AAM_MARGIN,
        hybrid_proto_weight=HYBRID_PROTO_WEIGHT,
        hybrid_aam_weight=HYBRID_AAM_WEIGHT,
        augmentation_probability=0.3,
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
