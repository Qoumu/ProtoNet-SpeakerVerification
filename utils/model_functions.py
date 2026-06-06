from pathlib import Path

import numpy as np
import torch

from utils.data_preprocessing import audio_to_mel_spectrogram
from model.ECAPATDNN import ECAPATDNNBackbone

def _pad_or_crop_mel(
    mel: np.ndarray,
    *,
    target_frames: int,
    pad_value: float,
) -> np.ndarray:
    """Pad or center-crop mel spectrogram to a fixed number of frames."""
    current_frames = mel.shape[1]
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif current_frames < target_frames:
        pad_width = target_frames - current_frames
        mel = np.pad(
            mel,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=pad_value,
        )
    return mel

def preprocess_audio(
    audio_path: Path,
    *,
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """Load audio and return a normalized mel tensor shaped [1, n_mels, T]."""
    mel, _, sr_loaded = audio_to_mel_spectrogram(
        audio_path=audio_path,
        sr=sr,
        duration=None,  # load full audio first
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        to_db=True,
    )

    target_frames = int(duration * sr_loaded / hop_length)
    mel = _pad_or_crop_mel(mel, target_frames=target_frames, pad_value=mel.min())

    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)  # [1, n_mels, T]
    return mel_tensor

def load_model(
    model_path: Path,
    *,
    device: torch.device, n_mels: int = 80, channels: int = 512, emb_dim: int = 64,
) -> ECAPATDNNBackbone:
    """Instantiate backbone and load weights."""
    model = ECAPATDNNBackbone(n_mels=n_mels, channels=channels, emb_dim=emb_dim)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def extract_embedding(
    model: ECAPATDNNBackbone,
    audio_path: Path,
    *,
    device: torch.device,
    sr: int = 16000,
    n_mels: int = 80,
    duration: float = 3.0,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """Compute a single utterance embedding (shape [emb_dim])."""
    mel = preprocess_audio(
        audio_path,
        sr=sr,
        n_mels=n_mels,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
    ).to(device)

    with torch.no_grad():
        embedding = model(mel).squeeze(0)
    return embedding.cpu()
