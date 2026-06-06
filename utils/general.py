from __future__ import annotations

from pathlib import Path
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import random
import torch
import torch.nn.functional as F

def _speaker_id_from_path(audio_path: Path, root: Path) -> str | None:
    rel_parts = audio_path.relative_to(root).parts
    dir_parts = rel_parts[:-1]

    for part in dir_parts:
        if part.isdigit():
            return part

    for part in dir_parts:
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            return digits

    if dir_parts:
        return dir_parts[0]

    return None


def _collect_audio_by_speaker(root: Path, ext: str) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for audio_path in root.rglob(f"*{ext}"):
        if not audio_path.is_file():
            continue
        speaker_id = _speaker_id_from_path(audio_path, root)
        if speaker_id is None:
            continue
        speakers.setdefault(speaker_id, []).append(audio_path)
    return speakers


def _select_speakers(
    speakers: dict[str, list[Path]],
    num_speakers: int,
    rng: random.Random,
) -> dict[str, list[Path]]:
    speaker_ids = sorted(speakers.keys(), key=lambda s: int(s) if s.isdigit() else s)
    if len(speaker_ids) <= num_speakers:
        chosen_ids = speaker_ids
    else:
        chosen_ids = sorted(rng.sample(speaker_ids, num_speakers))
    return {speaker_id: speakers[speaker_id] for speaker_id in chosen_ids}


def _build_label_map(speaker_ids: Iterable[str]) -> dict[str, int]:
    sorted_ids = sorted(speaker_ids, key=lambda s: int(s) if s.isdigit() else s)
    return {speaker_id: idx for idx, speaker_id in enumerate(sorted_ids)}


def _sample_clips_per_speaker(
    speaker_clips: dict[str, list[Path]],
    *,
    train_per_speaker: int,
    valid_per_speaker: int,
    test_per_speaker: int,
    rng: random.Random,
) -> dict[str, dict[str, list[Path]]]:
    splits: dict[str, dict[str, list[Path]]] = {
        "train": {},
        "valid": {},
        "test": {},
    }
    total_needed = train_per_speaker + valid_per_speaker + test_per_speaker

    for speaker_id, clips in speaker_clips.items():
        clips_sorted = sorted(clips)
        if len(clips_sorted) < total_needed:
            raise ValueError(
                f"Speaker {speaker_id} has {len(clips_sorted)} clips, "
                f"need at least {total_needed}."
            )
        chosen = rng.sample(clips_sorted, total_needed)
        idx = 0
        for split_name, take in (
            ("train", train_per_speaker),
            ("valid", valid_per_speaker),
            ("test", test_per_speaker),
        ):
            splits[split_name][speaker_id] = sorted(chosen[idx : idx + take])
            idx += take
    return splits


def select_librispeech_speakers(
    root: str | Path,
    *,
    num_speakers: int = 10,
    ext: str = ".flac",
    seed: int = 0,
) -> dict[str, list[Path]]:
    """
    Traverse LibriSpeech and select a subset of speakers.
    Returns a mapping: speaker_id -> list of clip paths.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_path}")

    speakers = _collect_audio_by_speaker(root_path, ext)
    if not speakers:
        raise ValueError(f"No audio files found under {root_path}")

    rng = random.Random(seed)
    return _select_speakers(speakers, num_speakers, rng)


def build_librispeech_sample_list(
    root: str | Path,
    *,
    num_speakers: int = 10,
    train_per_speaker: int = 12,
    valid_per_speaker: int = 4,
    test_per_speaker: int = 4,
    ext: str = ".flac",
    seed: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    """
    Return a flat list of samples and a label map.

    Each speaker contributes train_per_speaker + valid_per_speaker + test_per_speaker clips.
    The list contains dicts with keys: audio_filepath, label, label_id, split.
    """
    rng = random.Random(seed)
    speakers = select_librispeech_speakers(
        root, num_speakers=num_speakers, ext=ext, seed=seed
    )
    label_map = _build_label_map(speakers.keys())
    splits = _sample_clips_per_speaker(
        speakers,
        train_per_speaker=train_per_speaker,
        valid_per_speaker=valid_per_speaker,
        test_per_speaker=test_per_speaker,
        rng=rng,
    )

    samples: list[dict] = []
    for split_name in ("train", "valid", "test"):
        for speaker_id, clips in splits[split_name].items():
            label_id = label_map[speaker_id]
            for clip in clips:
                samples.append(
                    {
                        "audio_filepath": str(clip),
                        "label": speaker_id,
                        "label_id": label_id,
                        "split": split_name,
                    }
                )
    return samples, label_map

def build_prototypical_dataset(
    root: str | Path,
    *,
    num_speakers: int = 30,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    min_samples_per_speaker: int = 15,
    max_samples_per_speaker: int | None = None,
    ext: str = ".flac",
    seed: int = 0,
) -> Tuple[List[dict], Dict[str, int]]:
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_path}")
    
    # Collect all speakers and their audio files
    print("Collecting audio files...")
    all_speakers = _collect_audio_by_speaker(root_path, ext)
    
    if not all_speakers:
        raise ValueError(f"No audio files found under {root_path}")
    
    print(f"Found {len(all_speakers)} speakers total")
    
    # Filter speakers by minimum samples
    valid_speakers = {}
    for speaker_id, clips in all_speakers.items():
        if len(clips) >= min_samples_per_speaker:
            valid_speakers[speaker_id] = clips
        else:
            print(f"  Excluding speaker {speaker_id}: only {len(clips)} samples (need {min_samples_per_speaker})")
    
    print(f"Valid speakers: {len(valid_speakers)}")
    
    # Select subset of speakers
    rng = random.Random(seed)
    speaker_ids = sorted(valid_speakers.keys(), key=lambda s: int(s) if s.isdigit() else s)
    
    if len(speaker_ids) < num_speakers:
        print(f"Warning: Only {len(speaker_ids)} speakers available, using all")
        selected_speaker_ids = speaker_ids
    else:
        selected_speaker_ids = sorted(rng.sample(speaker_ids, num_speakers))
    
    print(f"Selected {len(selected_speaker_ids)} speakers for dataset")
    
    # Split speakers (NOT samples!) into train/val/test
    rng.shuffle(selected_speaker_ids)
    
    n_train = int(len(selected_speaker_ids) * train_ratio)
    n_val = int(len(selected_speaker_ids) * val_ratio)
    n_test = len(selected_speaker_ids) - n_train - n_val
    
    train_speakers = selected_speaker_ids[:n_train]
    val_speakers = selected_speaker_ids[n_train:n_train + n_val]
    test_speakers = selected_speaker_ids[n_train + n_val:]
    
    print(f"\nSpeaker Split (DISJOINT - No Overlap!):")
    print(f"  Train speakers: {len(train_speakers)} ({train_ratio*100:.0f}%)")
    print(f"  Valid speakers: {len(val_speakers)} ({val_ratio*100:.0f}%)")
    print(f"  Test speakers: {len(test_speakers)} ({test_ratio*100:.0f}%)")
    
    # Verify no overlap
    assert len(set(train_speakers) & set(val_speakers)) == 0, "Train and val speakers overlap!"
    assert len(set(train_speakers) & set(test_speakers)) == 0, "Train and test speakers overlap!"
    assert len(set(val_speakers) & set(test_speakers)) == 0, "Val and test speakers overlap!"
    print("✅ No speaker overlap between splits")
    
    # Create label map (all speakers get a label_id)
    label_map = {speaker_id: idx for idx, speaker_id in enumerate(selected_speaker_ids)}
    
    # Build dataset list
    dataset_list = []
    
    rng = random.Random(seed)
    
    # Helper function to select samples for a speaker
    def get_speaker_samples(speaker_id, clips):
        """Select samples for a speaker, respecting max_samples_per_speaker"""
        if max_samples_per_speaker is None or len(clips) <= max_samples_per_speaker:
            # Use all clips
            return clips
        else:
            # Randomly sample max_samples_per_speaker clips
            return rng.sample(sorted(clips), max_samples_per_speaker)
    
    # Add all samples from train speakers
    for speaker_id in train_speakers:
        clips = get_speaker_samples(speaker_id, valid_speakers[speaker_id])
        label_id = label_map[speaker_id]
        for clip in clips:
            dataset_list.append({
                'audio_filepath': str(clip),
                'label': speaker_id,
                'label_id': label_id,
                'split': 'train'
            })
    
    # Add all samples from validation speakers
    for speaker_id in val_speakers:
        clips = get_speaker_samples(speaker_id, valid_speakers[speaker_id])
        label_id = label_map[speaker_id]
        for clip in clips:
            dataset_list.append({
                'audio_filepath': str(clip),
                'label': speaker_id,
                'label_id': label_id,
                'split': 'valid'
            })
    
    # Add all samples from test speakers
    for speaker_id in test_speakers:
        clips = get_speaker_samples(speaker_id, valid_speakers[speaker_id])
        label_id = label_map[speaker_id]
        for clip in clips:
            dataset_list.append({
                'audio_filepath': str(clip),
                'label': speaker_id,
                'label_id': label_id,
                'split': 'test'
            })
    
    # Print statistics
    train_samples = [item for item in dataset_list if item['split'] == 'train']
    val_samples = [item for item in dataset_list if item['split'] == 'valid']
    test_samples = [item for item in dataset_list if item['split'] == 'test']
    
    print(f"\nSample Statistics:")
    if max_samples_per_speaker:
        print(f"  Max samples per speaker: {max_samples_per_speaker}")
    print(f"  Train: {len(train_samples)} samples from {len(train_speakers)} speakers")
    print(f"  Valid: {len(val_samples)} samples from {len(val_speakers)} speakers")
    print(f"  Test: {len(test_samples)} samples from {len(test_speakers)} speakers")
    print(f"  Total: {len(dataset_list)} samples")
    
    # Samples per speaker statistics
    from collections import Counter
    train_counts = Counter(item['label_id'] for item in train_samples)
    val_counts = Counter(item['label_id'] for item in val_samples)
    test_counts = Counter(item['label_id'] for item in test_samples)

    def format_split_counts(split_name: str, counts: Counter) -> str:
        if not counts:
            return f"  {split_name}: 0 speakers"
        values = list(counts.values())
        return (
            f"  {split_name}: min={min(values)}, max={max(values)}, "
            f"avg={sum(values) / len(values):.1f}"
        )
    
    print(f"\nSamples per speaker:")
    print(format_split_counts("Train", train_counts))
    print(format_split_counts("Valid", val_counts))
    print(format_split_counts("Test", test_counts))
    
    return dataset_list, label_map

def verify_prototypical_dataset(dataset_list: List[dict], n_way: int, k_shot: int, n_query: int):
    """
    Verify that dataset is properly split for Prototypical Networks
    """
    print("\n" + "="*60)
    print("Verifying Dataset for Prototypical Network Training")
    print("="*60)
    
    train_samples = [item for item in dataset_list if item['split'] == 'train']
    val_samples = [item for item in dataset_list if item['split'] == 'valid']
    test_samples = [item for item in dataset_list if item['split'] == 'test']
    
    train_speakers = set(item['label_id'] for item in train_samples)
    val_speakers = set(item['label_id'] for item in val_samples)
    test_speakers = set(item['label_id'] for item in test_samples)
    
    # Critical check: NO speaker overlap
    print("\n1. Checking speaker overlap (should be ZERO):")
    overlap_train_val = train_speakers & val_speakers
    overlap_train_test = train_speakers & test_speakers
    overlap_val_test = val_speakers & test_speakers
    
    if overlap_train_val:
        print(f"  ❌ CRITICAL ERROR: {len(overlap_train_val)} speakers in both train and val!")
        return False
    else:
        print(f"  ✅ Train/Val: No overlap")
    
    if overlap_train_test:
        print(f"  ❌ CRITICAL ERROR: {len(overlap_train_test)} speakers in both train and test!")
        return False
    else:
        print(f"  ✅ Train/Test: No overlap")
    
    if overlap_val_test:
        print(f"  ❌ CRITICAL ERROR: {len(overlap_val_test)} speakers in both val and test!")
        return False
    else:
        print(f"  ✅ Val/Test: No overlap")
    
    # Check if enough speakers for n-way
    print(f"\n2. Checking if splits have enough speakers for {n_way}-way learning:")
    required_speakers = n_way
    
    if len(train_speakers) < required_speakers:
        print(f"  ❌ Train: Only {len(train_speakers)} speakers (need {required_speakers})")
        return False
    else:
        print(f"  ✅ Train: {len(train_speakers)} speakers (need {required_speakers})")
    
    if len(val_speakers) < required_speakers:
        print(f"  ❌ Valid: Only {len(val_speakers)} speakers (need {required_speakers})")
        return False
    else:
        print(f"  ✅ Valid: {len(val_speakers)} speakers (need {required_speakers})")
    
    if len(test_speakers) < required_speakers:
        print(f"  ❌ Test: Only {len(test_speakers)} speakers (need {required_speakers})")
        return False
    else:
        print(f"  ✅ Test: {len(test_speakers)} speakers (need {required_speakers})")
    
    # Check samples per speaker
    print(f"\n3. Checking if speakers have enough samples for {k_shot}-shot {n_query}-query:")
    required_samples = k_shot + n_query
    
    from collections import Counter
    
    for split_name, samples in [("Train", train_samples), ("Valid", val_samples), ("Test", test_samples)]:
        speaker_counts = Counter(item['label_id'] for item in samples)
        min_samples = min(speaker_counts.values())
        
        if min_samples < required_samples:
            print(f"  ❌ {split_name}: Some speakers have only {min_samples} samples (need {required_samples})")
            return False
        else:
            print(f"  ✅ {split_name}: All speakers have at least {min_samples} samples (need {required_samples})")
    
    print("\n" + "="*60)
    print("✅ Dataset is correctly formatted for Prototypical Networks!")
    print("="*60)
    return True

def compute_prototypes(support_embeddings, support_labels, n_classes):
    prototypes = torch.zeros(n_classes, support_embeddings.size(1)).to(support_embeddings.device)
    for cls in range(n_classes):
        # Find all support samples for class c
        class_mask = (support_labels == cls)
        class_embeddings = support_embeddings[class_mask]

        # Compute prototype as mean of embeddings
        prototypes[cls] = class_embeddings.mean(dim=0)

    return prototypes

def cosine_distance(x, y, eps = 1e-8):
    """
    Compute cosine distance between two tensors
    x: [n_query, embedding_dim]
    y: [n_classes, embedding_dim]
    Returns: [n_query, n_classes]
    """
    # L2 normalize to avoid singularity
    x = F.normalize(x, p=2, dim=1, eps=eps)   # [n_query, D]
    y = F.normalize(y, p=2, dim=1, eps=eps)   # [n_classes, D]

    # Cosine similarity
    cos_sim = torch.matmul(x, y.T)            # [n_query, n_classes]

    # Cosine distance
    return 1.0 - cos_sim
