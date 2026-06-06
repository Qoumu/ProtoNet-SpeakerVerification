import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from pathlib import Path
from typing import Any, Optional, List
import matplotlib.pyplot as plt

from model.CNN import SpeakerCNN
from model.ECAPATDNN import ECAPATDNNBackbone
from utils.data_augmentation import DataAugmentation
from utils.data_preprocessing import SpeakerDataset
from utils.general import *

class AngularPrototypicalLoss(nn.Module):
    """Angular prototypical loss over episodic class prototypes."""

    def __init__(self, n_classes: int, n_query: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.n_classes = n_classes
        self.n_query = n_query
        self.scale = scale
        self.margin = margin

    def forward(self, cosine_scores: torch.Tensor):
        scores = cosine_scores.view(self.n_classes, self.n_query, -1)

        target_inds = torch.arange(0, self.n_classes, device=scores.device)
        target_inds = target_inds.view(self.n_classes, 1)
        target_inds = target_inds.expand(self.n_classes, self.n_query).long()

        adjusted_scores = scores
        if self.margin != 0.0:
            target_scores = scores.gather(2, target_inds.unsqueeze(2)).squeeze(2)
            clamped = target_scores.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            target_theta = torch.acos(clamped)
            target_logits = torch.cos(target_theta + self.margin)
            adjusted_scores = scores.clone()
            adjusted_scores.scatter_(2, target_inds.unsqueeze(2), target_logits.unsqueeze(2))

        scaled_scores = adjusted_scores * self.scale
        loss_val = F.cross_entropy(
            scaled_scores.reshape(-1, self.n_classes),
            target_inds.reshape(-1),
        )
        acc_val = scores.argmax(dim=2).eq(target_inds).float().mean()
        return loss_val, acc_val


class AAMSoftmaxLoss(nn.Module):
    """ArcFace/AAM-Softmax classifier head for speaker embedding training."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(normalized_embeddings, normalized_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=cosine.dtype)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale

        loss_val = F.cross_entropy(logits, labels)
        acc_val = logits.argmax(dim=1).eq(labels).float().mean()
        return loss_val, acc_val
    
def _get_tqdm():
    try:
        from tqdm import tqdm
    except Exception:
        return None
    return tqdm


def _maybe_get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
        for key in ("state_dict", "model_state_dict", "model", "net"):
            state = checkpoint.get(key)
            if isinstance(state, dict) and state and all(torch.is_tensor(value) for value in state.values()):
                return state
    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or a dict containing "
        "'state_dict'/'model_state_dict'."
    )


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _speaker_sample_counts(split_data: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in split_data:
        label_id = item["label_id"]
        counts[label_id] = counts.get(label_id, 0) + 1
    return counts


def _validate_episode_configuration(
    *,
    split_name: str,
    split_data: list[dict],
    n_way: int,
    k_shot: int,
    n_query: int,
) -> None:
    speaker_counts = _speaker_sample_counts(split_data)
    if not speaker_counts:
        raise ValueError(f"{split_name} split is empty.")

    speaker_total = len(speaker_counts)
    if speaker_total < n_way:
        raise ValueError(
            f"{split_name} split has {speaker_total} speakers, but n_way={n_way}. "
            f"Set n_way <= {speaker_total} or change the speaker split."
        )

    required_samples = k_shot + n_query
    min_samples = min(speaker_counts.values())
    if min_samples < required_samples:
        raise ValueError(
            f"{split_name} split requires at least {required_samples} samples per speaker "
            f"for k_shot={k_shot} and n_query={n_query}, but the minimum is {min_samples}. "
            f"Reduce k_shot/n_query or change the dataset filtering."
        )


def sample_episode(dataset_list, n_way, k_shot, n_query):
    """
    Sample one episode for few-shot learning

    Args:
        dataset_list: list of dicts with 'audio_filepath', 'label_id', 'split'
        n_way: number of classes (speakers) per episode
        k_shot: number of support examples per class
        n_query: number of query examples per class

    Returns:
        support_data: list of support samples
        query_data: list of query samples
    """
    # Group samples by speaker
    speaker_dict = {}
    for item in dataset_list:
        label_id = item['label_id']
        if label_id not in speaker_dict:
            speaker_dict[label_id] = []
        speaker_dict[label_id].append(item)

    # Randomly select n_way speakers
    available_speakers = list(speaker_dict.keys())
    selected_speakers = random.sample(available_speakers, n_way)

    support_data = []
    query_data = []

    for new_label, speaker_id in enumerate(selected_speakers):
        samples = speaker_dict[speaker_id]

        # Need at least k_shot + n_query samples
        if len(samples) < k_shot + n_query:
            raise ValueError(f"Speaker {speaker_id} has only {len(samples)} samples, need {k_shot + n_query}")

        # Randomly sample support and query
        selected = random.sample(samples, k_shot + n_query)

        for i, sample in enumerate(selected):
            # Create new dict with remapped label (0 to n_way-1)
            new_sample = sample.copy()
            new_sample['episode_label'] = new_label

            if i < k_shot:
                support_data.append(new_sample)
            else:
                query_data.append(new_sample)

    return support_data, query_data


def _build_filepath_index(dataset) -> dict[str, int]:
    return {item['audio_filepath']: i for i, item in enumerate(dataset.dataset_list)}


def _prepare_episode_tensors(support_data, query_data, dataset, device):
    filepath_to_idx = _build_filepath_index(dataset)

    support_waveforms = []
    support_labels = []
    support_global_labels = []
    for sample in support_data:
        idx = filepath_to_idx[sample['audio_filepath']]
        waveform, _ = dataset.load_waveform_tensor(idx)
        support_waveforms.append(waveform)
        support_labels.append(sample['episode_label'])
        support_global_labels.append(sample['label_id'])

    query_waveforms = []
    query_labels = []
    query_global_labels = []
    for sample in query_data:
        idx = filepath_to_idx[sample['audio_filepath']]
        waveform, _ = dataset.load_waveform_tensor(idx)
        query_waveforms.append(waveform)
        query_labels.append(sample['episode_label'])
        query_global_labels.append(sample['label_id'])

    support_waveforms = torch.stack(support_waveforms)
    query_waveforms = torch.stack(query_waveforms)
    support_mels = dataset.waveforms_to_mels(support_waveforms, device)
    support_labels = torch.tensor(support_labels, device=device)
    support_global_labels = torch.tensor(support_global_labels, device=device)
    query_mels = dataset.waveforms_to_mels(query_waveforms, device)
    query_labels = torch.tensor(query_labels, device=device)
    query_global_labels = torch.tensor(query_global_labels, device=device)

    return (
        support_mels,
        support_labels,
        support_global_labels,
        query_mels,
        query_labels,
        query_global_labels,
    )


def _forward_episode(model, support_data, query_data, dataset, device, n_way):
    (
        support_mels,
        support_labels,
        support_global_labels,
        query_mels,
        query_labels,
        query_global_labels,
    ) = _prepare_episode_tensors(
        support_data=support_data,
        query_data=query_data,
        dataset=dataset,
        device=device,
    )

    support_embeddings = model(support_mels)
    query_embeddings = model(query_mels)
    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
    prototype_scores = model.pn_predict(query_embeddings, prototypes)

    return {
        "support_embeddings": support_embeddings,
        "query_embeddings": query_embeddings,
        "prototypes": prototypes,
        "prototype_scores": prototype_scores,
        "query_episode_labels": query_labels,
        "support_global_labels": support_global_labels,
        "query_global_labels": query_global_labels,
    }


def _extract_verification_scores(logits: torch.Tensor, query_labels: torch.Tensor):
    similarities = logits.detach().cpu().numpy()
    labels = query_labels.detach().cpu().numpy()

    positive_scores = similarities[np.arange(len(labels)), labels]
    positive_labels = np.ones_like(positive_scores, dtype=np.int32)

    negative_mask = np.ones_like(similarities, dtype=bool)
    negative_mask[np.arange(len(labels)), labels] = False
    negative_scores = similarities[negative_mask]
    negative_labels = np.zeros_like(negative_scores, dtype=np.int32)

    scores = np.concatenate([positive_scores, negative_scores])
    score_labels = np.concatenate([positive_labels, negative_labels])
    return scores, score_labels


def _compute_eer(scores: np.ndarray, labels: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires both positive and negative verification scores.")

    order = np.argsort(scores)[::-1]
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    true_positives = np.cumsum(labels_sorted == 1)
    false_positives = np.cumsum(labels_sorted == 0)

    distinct = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct, len(scores_sorted) - 1]

    far = false_positives[threshold_idxs] / negatives
    fnr = (positives - true_positives[threshold_idxs]) / positives
    thresholds = scores_sorted[threshold_idxs]

    far = np.r_[0.0, far, 1.0]
    fnr = np.r_[1.0, fnr, 0.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]

    diff = fnr - far
    crossing = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    if crossing.size > 0:
        idx = int(crossing[0])
        far_1, far_2 = far[idx], far[idx + 1]
        fnr_1, fnr_2 = fnr[idx], fnr[idx + 1]
        thr_1, thr_2 = thresholds[idx], thresholds[idx + 1]
        diff_1, diff_2 = diff[idx], diff[idx + 1]
        denom = diff_1 - diff_2
        if abs(denom) < 1e-12:
            eer = (far_1 + fnr_1) / 2.0
            eer_far = far_1
            eer_fnr = fnr_1
            eer_threshold = thr_1
        else:
            weight = diff_1 / denom
            eer = far_1 + weight * (far_2 - far_1)
            eer_far = far_1 + weight * (far_2 - far_1)
            eer_fnr = fnr_1 + weight * (fnr_2 - fnr_1)
            eer_threshold = thr_1 + weight * (thr_2 - thr_1)
    else:
        idx = int(np.argmin(np.abs(diff)))
        eer = (far[idx] + fnr[idx]) / 2.0
        eer_far = far[idx]
        eer_fnr = fnr[idx]
        eer_threshold = thresholds[idx]

    return float(eer), far, fnr, thresholds, float(eer_threshold), float(eer_far), float(eer_fnr)


def _plot_det_curve(
    far: np.ndarray,
    fnr: np.ndarray,
    eer: float,
    eer_far: float,
    eer_fnr: float,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_prob = 1e-4
    max_prob = 1.0 - min_prob

    far = np.clip(far, min_prob, max_prob)
    fnr = np.clip(fnr, min_prob, max_prob)

    normal = torch.distributions.Normal(0.0, 1.0)
    far_det = normal.icdf(torch.tensor(far, dtype=torch.float32)).cpu().numpy()
    fnr_det = normal.icdf(torch.tensor(fnr, dtype=torch.float32)).cpu().numpy()
    eer_far_det = float(normal.icdf(torch.tensor(np.clip(eer_far, min_prob, max_prob), dtype=torch.float32)).item())
    eer_fnr_det = float(normal.icdf(torch.tensor(np.clip(eer_fnr, min_prob, max_prob), dtype=torch.float32)).item())

    tick_probs = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40], dtype=np.float32) / 100.0
    tick_positions = normal.icdf(torch.tensor(tick_probs)).cpu().numpy()
    tick_labels = [f"{prob * 100:g}" for prob in tick_probs]

    plt.figure(figsize=(7, 7))
    plt.plot(far_det, fnr_det, linewidth=2, label="DET")
    plt.scatter([eer_far_det], [eer_fnr_det], color="red", s=35, label=f"EER {eer * 100:.2f}%")
    plt.annotate(
        f"({eer_far * 100:.2f}%, {eer_fnr * 100:.2f}%)",
        xy=(eer_far_det, eer_fnr_det),
        xytext=(8, -12),
        textcoords="offset points",
        color="red",
        fontsize=9,
    )
    plt.xlabel("False Alarm Rate (%)")
    plt.ylabel("Miss Rate (%)")
    plt.title("Detection Error Tradeoff Curve")
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"DET curve saved to {output_path}")

def _remap_classifier_labels(global_labels: torch.Tensor, label_to_index: dict[int, int]) -> torch.Tensor:
    mapped = [label_to_index[int(label)] for label in global_labels.detach().cpu().tolist()]
    return torch.tensor(mapped, device=global_labels.device, dtype=torch.long)


def _prototype_accuracy(prototype_scores: torch.Tensor, query_labels: torch.Tensor) -> float:
    predicted = prototype_scores.argmax(dim=1)
    correct = predicted.eq(query_labels).sum().item()
    total = query_labels.size(0)
    return 100.0 * correct / total


def train_episode(
    model,
    support_data,
    query_data,
    dataset,
    prototype_loss_fn,
    optimizer,
    device,
    n_way,
    *,
    loss_mode: str,
    classifier_label_map: dict[int, int] | None = None,
    aam_loss_fn: Optional[AAMSoftmaxLoss] = None,
    hybrid_proto_weight: float = 1.0,
    hybrid_aam_weight: float = 1.0,
):
    """
    Train one episode of prototypical network

    Args:
        model: PrototypicalNetwork model
        support_data: list of support samples
        query_data: list of query samples
        dataset: SpeakerDataset for loading audio
        criterion: loss function
        optimizer: optimizer
        device: torch device
        n_way: number of classes in episode
    """
    # Forward pass
    optimizer.zero_grad()
    
    model.train()

    episode_output = _forward_episode(
        model=model,
        support_data=support_data,
        query_data=query_data,
        dataset=dataset,
        device=device,
        n_way=n_way,
    )
    prototype_scores = episode_output["prototype_scores"]
    query_labels = episode_output["query_episode_labels"]

    proto_loss, _ = prototype_loss_fn(prototype_scores)
    total_loss = proto_loss

    if loss_mode == "aam_softmax":
        if aam_loss_fn is None or classifier_label_map is None:
            raise ValueError("AAM-Softmax training requires classifier_label_map and aam_loss_fn.")
        classifier_embeddings = torch.cat(
            [episode_output["support_embeddings"], episode_output["query_embeddings"]],
            dim=0,
        )
        classifier_labels = _remap_classifier_labels(
            torch.cat(
                [episode_output["support_global_labels"], episode_output["query_global_labels"]],
                dim=0,
            ),
            classifier_label_map,
        )
        total_loss, _ = aam_loss_fn(classifier_embeddings, classifier_labels)
    elif loss_mode == "hybrid":
        if aam_loss_fn is None or classifier_label_map is None:
            raise ValueError("Hybrid loss requires classifier_label_map and aam_loss_fn.")
        classifier_embeddings = torch.cat(
            [episode_output["support_embeddings"], episode_output["query_embeddings"]],
            dim=0,
        )
        classifier_labels = _remap_classifier_labels(
            torch.cat(
                [episode_output["support_global_labels"], episode_output["query_global_labels"]],
                dim=0,
            ),
            classifier_label_map,
        )
        aam_loss, _ = aam_loss_fn(classifier_embeddings, classifier_labels)
        total_loss = (hybrid_proto_weight * proto_loss) + (hybrid_aam_weight * aam_loss)
    elif loss_mode != "angular_proto":
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    
    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    accuracy = _prototype_accuracy(prototype_scores, query_labels)

    return total_loss.item(), accuracy


def validate_episode(model, support_data, query_data, dataset, criterion, device, n_way):
    """Validate one episode"""
    model.eval()

    with torch.no_grad():
        episode_output = _forward_episode(
            model=model,
            support_data=support_data,
            query_data=query_data,
            dataset=dataset,
            device=device,
            n_way=n_way,
        )
        prototype_scores = episode_output["prototype_scores"]
        query_labels = episode_output["query_episode_labels"]

        # Compute loss and accuracy
        loss, _ = criterion(prototype_scores)
        accuracy = _prototype_accuracy(prototype_scores, query_labels)

    return loss.item(), accuracy


def evaluate_test_episodes(
    model,
    test_data,
    test_dataset,
    loss_fn,
    device,
    n_way,
    k_shot,
    n_query,
    n_val_episodes,
    n_test_episodes: int | None = None,
    model_path: Optional[str | Path] = None,
    eval_seed: Optional[int] = None,
    det_curve_path: Optional[str | Path] = None,
):
    """Evaluate the model on held-out test episodes."""
    model_path = Path(model_path) if model_path is not None else None
    if model_path is not None and model_path.exists():
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

    if eval_seed is not None:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)

    test_episode_count = n_val_episodes if n_test_episodes is None else n_test_episodes
    test_losses = []
    test_accs = []
    verification_scores = []
    verification_labels = []

    for _ in range(test_episode_count):
        try:
            test_support, test_query = sample_episode(test_data, n_way, k_shot, n_query)
            model.eval()
            with torch.no_grad():
                episode_output = _forward_episode(
                    model=model,
                    support_data=test_support,
                    query_data=test_query,
                    dataset=test_dataset,
                    device=device,
                    n_way=n_way,
                )
                test_logits = episode_output["prototype_scores"]
                test_query_labels = episode_output["query_episode_labels"]

            test_loss, _ = loss_fn(test_logits)
            test_acc = _prototype_accuracy(test_logits, test_query_labels)

            episode_scores, episode_labels = _extract_verification_scores(
                test_logits,
                test_query_labels,
            )
            test_losses.append(test_loss.item())
            test_accs.append(test_acc)
            verification_scores.append(episode_scores)
            verification_labels.append(episode_labels)
        except ValueError:
            continue

    if test_losses:
        avg_test_loss = np.mean(test_losses)
        avg_test_acc = np.mean(test_accs)
        print(f"\nTest Results ({len(test_losses)} episodes):")
        print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.2f}%")

        if verification_scores and verification_labels:
            all_scores = np.concatenate(verification_scores)
            all_labels = np.concatenate(verification_labels)
            eer, far, fnr, _, eer_threshold, eer_far, eer_fnr = _compute_eer(all_scores, all_labels)
            print(f"  Test EER: {eer * 100:.2f}%")
            print(
                f"  EER Point: threshold={eer_threshold:.4f}, "
                f"FAR={eer_far * 100:.2f}%, FNR={eer_fnr * 100:.2f}%"
            )

            if det_curve_path is not None:
                _plot_det_curve(far, fnr, eer, eer_far, eer_fnr, det_curve_path)

            return avg_test_loss, avg_test_acc, eer

        return avg_test_loss, avg_test_acc, None

    print("\nTest evaluation skipped: not enough data to sample episodes.")
    return None, None, None


def train_prototypical_network(
    dataset_list: List[dict],
    train_mode: bool = True,
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 5,
    n_episodes: int = 1000,
    n_val_episodes: int = 100,
    n_test_episodes: int | None = None,
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 512,
    hop_length: int = 256,
    vad_enabled: bool = False,
    vad_top_db: float = 30.0,
    vad_frame_length: int = 2048,
    vad_hop_length: int = 512,
    train_augment: bool = True,
    augmentation_probability: float = 0.8,
    augmentation_rir_dir: Optional[str | Path] = None,
    augmentation_kwargs: Optional[dict] = None,
    show_progress: bool = True,
    training_loss_mode: str = "aam_softmax",
    proto_scale: float = 30.0,
    proto_margin: float = 0.2,
    aam_scale: float = 30.0,
    aam_margin: float = 0.2,
    hybrid_proto_weight: float = 1.0,
    hybrid_aam_weight: float = 1.0,
    init_checkpoint_path: Optional[str | Path] = None,
    model_path: str | Path = "output/ECAPATDNN_protonet_model.pth",
    eval_seed: Optional[int] = None,
    plot_path: Optional[str | Path] = "ECAPATDNN_protonet_training_curves.png",
    det_curve_path: Optional[str | Path] = "output/ECAPATDNN_protonet_det_curve.png",
):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split data
    train_data = [item for item in dataset_list if item['split'] == 'train']
    val_data = [item for item in dataset_list if item['split'] == 'valid']
    test_data = [item for item in dataset_list if item['split'] == 'test']

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nEpisodic Learning Configuration:")
    print(f"  {n_way}-way {k_shot}-shot learning")
    print(f"  {n_query} query samples per class")
    print(f"  {n_episodes} training episodes")
    print(f"  {n_val_episodes} validation episodes")
    print(f"  Training loss mode: {training_loss_mode}")
    print(f"  Angular proto scale: {proto_scale}")
    print(f"  Angular proto margin: {proto_margin}")
    print(f"  AAM scale: {aam_scale}")
    print(f"  AAM margin: {aam_margin}")
    print(f"  VAD enabled: {vad_enabled}")
    if vad_enabled:
        print(f"  VAD top_db: {vad_top_db}")
        print(f"  VAD frame_length: {vad_frame_length}")
        print(f"  VAD hop_length: {vad_hop_length}")
    if training_loss_mode == "hybrid":
        print(f"  Hybrid weights: proto={hybrid_proto_weight}, aam={hybrid_aam_weight}")

    if train_mode:
        _validate_episode_configuration(
            split_name="Train",
            split_data=train_data,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
        )
        _validate_episode_configuration(
            split_name="Validation",
            split_data=val_data,
            n_way=n_way,
            k_shot=k_shot,
            n_query=n_query,
        )

    train_augmenter = None
    if train_augment:
        train_augmenter = DataAugmentation(
            sample_rate=sr,
            rir_dir=str(augmentation_rir_dir) if augmentation_rir_dir is not None else None,
            p=augmentation_probability,
            **(augmentation_kwargs or {}),
        )
        print("\nTraining augmentation:")
        print(f"  Enabled: {train_augment}")
        print(f"  Probability: {augmentation_probability}")
        print(f"  RIR directory: {augmentation_rir_dir or 'None'}")
    else:
        print("\nTraining augmentation:")
        print("  Enabled: False")

    dataset = SpeakerDataset(
        train_data,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        augment=train_augment,
        n_fft=n_fft,
        hop_length=hop_length,
        waveform_augmenter=train_augmenter,
        vad_enabled=vad_enabled,
        vad_top_db=vad_top_db,
        vad_frame_length=vad_frame_length,
        vad_hop_length=vad_hop_length,
    )

    val_dataset = SpeakerDataset(
        val_data,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        augment=False,
        n_fft=n_fft,
        hop_length=hop_length,
        vad_enabled=vad_enabled,
        vad_top_db=vad_top_db,
        vad_frame_length=vad_frame_length,
        vad_hop_length=vad_hop_length,
    )

    test_dataset = SpeakerDataset(
        test_data,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        augment=False,
        n_fft=n_fft,
        hop_length=hop_length,
        vad_enabled=vad_enabled,
        vad_top_db=vad_top_db,
        vad_frame_length=vad_frame_length,
        vad_hop_length=vad_hop_length,
    )

    # Initialize model
    model = ECAPATDNNBackbone(n_mels=n_mels, channels=512, emb_dim=192)
    model = model.to(device)

    if init_checkpoint_path is not None:
        init_checkpoint_path = Path(init_checkpoint_path)
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"Fine-tune checkpoint not found: {init_checkpoint_path}")
        checkpoint = torch.load(init_checkpoint_path, map_location=device)
        state_dict = _strip_module_prefix(_maybe_get_state_dict(checkpoint))
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded initial weights from: {init_checkpoint_path}")

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Loss and optimizer
    prototype_loss_fn = AngularPrototypicalLoss(
        n_classes=n_way,
        n_query=n_query,
        scale=proto_scale,
        margin=proto_margin,
    )

    train_classifier_map: dict[int, int] | None = None
    aam_loss_fn: Optional[AAMSoftmaxLoss] = None
    optimizer_params = list(model.parameters())

    if training_loss_mode in {"aam_softmax", "hybrid"}:
        train_speaker_ids = sorted({item["label_id"] for item in train_data})
        train_classifier_map = {label_id: idx for idx, label_id in enumerate(train_speaker_ids)}
        aam_loss_fn = AAMSoftmaxLoss(
            embedding_dim=192,
            num_classes=len(train_speaker_ids),
            scale=aam_scale,
            margin=aam_margin,
        ).to(device)
        optimizer_params.extend(list(aam_loss_fn.parameters()))
    elif training_loss_mode != "angular_proto":
        raise ValueError("training_loss_mode must be one of: angular_proto, aam_softmax, hybrid")

    optimizer = optim.AdamW(optimizer_params, lr=0.0001, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    tqdm = _get_tqdm()

    # Training episodes
    if train_mode is True:
        print("\nStarting training...")
        train_losses = []
        train_accs = []

        iterable = range(n_episodes)
        if show_progress and tqdm is not None:
            iterable = tqdm(range(n_episodes), desc="Training")

        for episode in iterable:
            try:
                # Sample episode
                support_data, query_data = sample_episode(train_data, n_way, k_shot, n_query)

                # Train on episode
                loss, acc = train_episode(
                    model, support_data, query_data, dataset,
                    prototype_loss_fn, optimizer, device, n_way,
                    loss_mode=training_loss_mode,
                    classifier_label_map=train_classifier_map,
                    aam_loss_fn=aam_loss_fn,
                    hybrid_proto_weight=hybrid_proto_weight,
                    hybrid_aam_weight=hybrid_aam_weight,
                )

                train_losses.append(loss)
                train_accs.append(acc)

                # Validate every 2 episodes
                if (episode + 1) % 2 == 0:
                    val_losses = []
                    val_accs = []

                    for _ in range(n_val_episodes):
                        try:
                            val_support, val_query = sample_episode(val_data, n_way, k_shot, n_query)
                            val_loss, val_acc = validate_episode(
                                model, val_support, val_query, val_dataset,
                                prototype_loss_fn, device, n_way
                            )
                            val_losses.append(val_loss)
                            val_accs.append(val_acc)
                        except ValueError:
                            continue

                    avg_train_loss = np.mean(train_losses[-100:])
                    avg_train_acc = np.mean(train_accs[-100:])
                    avg_val_loss = np.mean(val_losses)
                    avg_val_acc = np.mean(val_accs)

                    scheduler.step(avg_val_loss)

                    print(f"\nEpisode {episode+1}/{n_episodes}")
                    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")
                    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")

                    history["train_loss"].append(avg_train_loss)
                    history["train_acc"].append(avg_train_acc)
                    history["val_loss"].append(avg_val_loss)
                    history["val_acc"].append(avg_val_acc)

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), model_path)
                        print(f"  ✓ Saved best model (Val Loss: {avg_val_loss:.4f})")

            except ValueError as e:
                # Skip episodes where we can't sample enough data
                continue

        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"\nSkipping training. Evaluating checkpoint from: {model_path}")

    if plot_path and len(history["train_loss"]) > 0:
        _plot_training_curves(history, plot_path)
        print(f"Saved training curves to: {plot_path}")

    evaluate_test_episodes(
        model=model,
        test_data=test_data,
        test_dataset=test_dataset,
        loss_fn=prototype_loss_fn,
        device=device,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        n_val_episodes=n_val_episodes,
        n_test_episodes=n_test_episodes,
        model_path=model_path,
        eval_seed=eval_seed,
        det_curve_path=det_curve_path,
    )

    return model

def _plot_training_curves(history: dict, output_path: str | Path) -> None:
    """Plot training curves"""
    checkpoints = range(10, len(history["train_loss"]) * 10 + 1, 10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(checkpoints, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(checkpoints, history["val_loss"], label="Val Loss", marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss over Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(checkpoints, history["train_acc"], label="Train Acc", marker='o')
    plt.plot(checkpoints, history["val_acc"], label="Val Acc", marker='s')
    plt.xlabel("Episode")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"Training curves saved to {output_path}")
