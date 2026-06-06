import os
import numpy as np

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, List, Optional, Tuple

from utils.data_augmentation import DataAugmentation


def apply_vad(
    y: np.ndarray,
    *,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Remove low-energy regions and concatenate non-silent intervals.

    If no non-silent interval is found, return the original waveform so the
    caller can decide how to handle it.
    """
    if y.ndim != 1:
        raise ValueError("Waveform y must be 1D for VAD.")
    if y.size == 0:
        return y

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive for VAD.")

    if y.size < frame_length:
        padded = np.pad(y, (0, frame_length - y.size), mode="constant")
    else:
        padded = y

    frame_starts = np.arange(0, max(1, padded.size - frame_length + 1), hop_length, dtype=np.int64)
    if frame_starts.size == 0 or frame_starts[-1] + frame_length < padded.size:
        frame_starts = np.append(frame_starts, max(0, padded.size - frame_length))

    rms_values = []
    for start in frame_starts:
        frame = padded[start:start + frame_length]
        rms = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
        rms_values.append(rms)

    rms_values = np.asarray(rms_values, dtype=np.float64)
    ref_rms = float(np.max(rms_values))
    if ref_rms <= 1e-12:
        return y

    threshold = ref_rms * (10.0 ** (-top_db / 20.0))
    voiced_mask = rms_values >= threshold
    if not np.any(voiced_mask):
        return y

    intervals: list[tuple[int, int]] = []
    for start, keep in zip(frame_starts, voiced_mask):
        if not keep:
            continue
        end = min(start + frame_length, y.size)
        if end <= start:
            continue
        if intervals and start <= intervals[-1][1]:
            prev_start, prev_end = intervals[-1]
            intervals[-1] = (prev_start, max(prev_end, end))
        else:
            intervals.append((int(start), int(end)))

    if not intervals:
        return y

    voiced = [y[start:end] for start, end in intervals]
    merged = np.concatenate(voiced, axis=0)
    if merged.size == 0:
        return y
    return merged.astype(np.float32, copy=False)


def load_audio_waveform(
    audio_path: str | Path,
    *,
    sr: int = 16000,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Load a mono waveform at the target sample rate."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr_loaded = librosa.load(
        str(audio_path),
        sr=sr,
        mono=True,
        offset=offset,
        duration=duration,
    )
    if y.size == 0:
        raise ValueError("Loaded audio is empty. Check offset/duration and file content.")

    return y.astype(np.float32), sr_loaded

def audio_to_mel_spectrogram(
    audio_path: str | Path | None = None,
    *,
    y: Optional[np.ndarray] = None,
    sr: int = 16000,
    offset: float = 0.0,
    duration: Optional[float] = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    to_db: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load an audio file segment and convert it to a Mel spectrogram.

    Returns:
        mel: (n_mels, time) float32 ndarray (dB if to_db=True else power mel)
        y: waveform float32 (n_samples,)
        sr: sample rate (int)
    """
    if y is None:
        if audio_path is None:
            raise ValueError("Provide either audio_path or y.")
        y, sr_loaded = load_audio_waveform(
            audio_path,
            sr=sr,
            offset=offset,
            duration=duration,
        )
    else:
        if y.ndim != 1:
            raise ValueError("Waveform y must be 1D.")
        sr_loaded = sr

    if y.size == 0:
        raise ValueError("Loaded audio is empty. Check offset/duration and file content.")

    if win_length is None:
        win_length = n_fft
    if fmax is None:
        fmax = sr_loaded / 2.0

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr_loaded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )

    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)

    return mel.astype(np.float32), y.astype(np.float32), sr_loaded

class SpeakerDataset(Dataset):
    def __init__(
        self,
        dataset_list: List[dict],
        sr: int = 16000,
        n_mels: int = 80,
        duration: float = 3.0,
        augment: bool = False,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        waveform_augmenter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        vad_enabled: bool = False,
        vad_top_db: float = 30.0,
        vad_frame_length: int = 2048,
        vad_hop_length: int = 512,
    ):
        """
        Args:
            dataset_list: list of dicts with keys: 'audio_filepath', 'label', 'label_id', 'split'
            sr: audio sample rate
            n_mels: number of mel filterbanks
            duration: fixed duration in seconds (will pad/crop)
            augment: whether to apply augmentation
            n_fft: FFT window size
            hop_length: hop length for STFT
            win_length: window length (defaults to n_fft)
            fmin: minimum frequency
            fmax: maximum frequency (defaults to sr/2)
            waveform_augmenter: callable waveform augmenter used when augment=True
            vad_enabled: enable energy-based voice activity detection
            vad_top_db: silence threshold used by librosa.effects.split
            vad_frame_length: analysis frame length for VAD
            vad_hop_length: analysis hop length for VAD
        """
        self.dataset_list = dataset_list
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.augment = augment
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2.0
        self.waveform_augmenter = waveform_augmenter
        self.vad_enabled = vad_enabled
        self.vad_top_db = vad_top_db
        self.vad_frame_length = vad_frame_length
        self.vad_analysis_hop_length = vad_hop_length
        self.target_num_samples = max(1, int(round(self.duration * self.sr)))
        self.target_frames = int(self.duration * self.sr / self.hop_length)
        self._feature_transform_cache: dict[str, tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB]] = {}

        if self.augment and self.waveform_augmenter is None:
            self.waveform_augmenter = DataAugmentation(sample_rate=self.sr)

    def __len__(self):
        return len(self.dataset_list)

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        """Apply waveform augmentation during training only."""
        if not self.augment or self.waveform_augmenter is None:
            return y

        augmented = self.waveform_augmenter(y)
        return np.asarray(augmented, dtype=np.float32)

    def _load_waveform(self, audio_path: str | Path) -> np.ndarray:
        y, _ = load_audio_waveform(
            audio_path,
            sr=self.sr,
            duration=None,
        )
        return y

    def _apply_vad(self, y: np.ndarray) -> np.ndarray:
        if not self.vad_enabled:
            return y
        return apply_vad(
            y,
            top_db=self.vad_top_db,
            frame_length=self.vad_frame_length,
            hop_length=self.vad_analysis_hop_length,
        )

    def _pad_or_crop_waveform(self, y: np.ndarray) -> np.ndarray:
        """Pad or crop waveform to fixed duration before batching on device."""
        current_samples = y.shape[0]

        if current_samples > self.target_num_samples:
            if self.augment:
                start = np.random.randint(0, current_samples - self.target_num_samples + 1)
            else:
                start = (current_samples - self.target_num_samples) // 2
            y = y[start:start + self.target_num_samples]
        elif current_samples < self.target_num_samples:
            y = np.pad(y, (0, self.target_num_samples - current_samples), mode='constant')

        return y.astype(np.float32, copy=False)

    def _pad_or_crop_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Pad or crop mel spectrogram to fixed time dimension

        Args:
            mel: (n_mels, time) array

        Returns:
            mel: (n_mels, target_time) array
        """
        target_frames = int(self.duration * self.sr / self.hop_length)
        current_frames = mel.shape[1]

        if current_frames > target_frames:
            # Random crop for augmentation, center crop for validation
            if self.augment:
                start = np.random.randint(0, current_frames - target_frames + 1)
            else:
                start = (current_frames - target_frames) // 2
            mel = mel[:, start:start + target_frames]
        elif current_frames < target_frames:
            # Pad with minimum value (for dB scale)
            pad_width = target_frames - current_frames
            pad_value = mel.min()
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_value)

        return mel

    def _pad_or_crop_mel_tensor(self, mel: torch.Tensor) -> torch.Tensor:
        current_frames = mel.shape[-1]
        if current_frames > self.target_frames:
            if self.augment:
                start = int(torch.randint(0, current_frames - self.target_frames + 1, (1,), device=mel.device).item())
            else:
                start = (current_frames - self.target_frames) // 2
            mel = mel[:, start:start + self.target_frames]
        elif current_frames < self.target_frames:
            mel = F.pad(mel, (0, self.target_frames - current_frames), value=float(mel.min().item()))
        return mel

    def _get_feature_transforms(
        self,
        device: torch.device,
    ) -> tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB]:
        cache_key = str(device)
        transforms = self._feature_transform_cache.get(cache_key)
        if transforms is None:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                power=2.0,
            ).to(device)
            db_transform = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)
            transforms = (mel_transform, db_transform)
            self._feature_transform_cache[cache_key] = transforms
        return transforms

    def load_waveform_tensor(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset_list[idx]
        label_id = item['label_id']

        y = self._load_waveform(item['audio_filepath'])
        y = self._apply_vad(y)
        if self.augment:
            y = self._augment_waveform(y)
        y = self._pad_or_crop_waveform(y)

        waveform_tensor = torch.from_numpy(y).float()
        return waveform_tensor, label_id

    def waveforms_to_mels(self, waveforms: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Convert a batch of waveforms into normalized mel tensors on the target device."""
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)

        waveforms = waveforms.to(device)
        mel_transform, db_transform = self._get_feature_transforms(device)

        mels = mel_transform(waveforms)
        mels = db_transform(mels)
        mels = torch.stack([self._pad_or_crop_mel_tensor(mel) for mel in mels], dim=0)

        mel_mean = mels.mean(dim=(1, 2), keepdim=True)
        mel_std = mels.std(dim=(1, 2), keepdim=True)
        mels = (mels - mel_mean) / (mel_std + 1e-8)
        return mels

    def __getitem__(self, idx):
        item = self.dataset_list[idx]
        audio_path = item['audio_filepath']
        label_id = item['label_id']

        # Load audio and convert to mel spectrogram using your function
        _, y, sr = audio_to_mel_spectrogram(
            audio_path=audio_path,
            sr=self.sr,
            duration=None,  # Load entire file first
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            to_db=True,
        )
        y = self._apply_vad(y)

        # Apply waveform augmentation and recompute mel if needed
        if self.augment:
            y = self._augment_waveform(y)

        # Recompute mel spectrogram with filtered/augmented waveform
        mel, _, _ = audio_to_mel_spectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            to_db=True,
        )

        # Pad or crop to fixed duration
        mel = self._pad_or_crop_mel(mel)

        # Normalize (per-sample normalization)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Convert to torch tensor: (n_mels, time)
        mel_tensor = torch.from_numpy(mel).float()

        return mel_tensor, label_id
    
def _pad_or_crop_mel_chunk(
    mel: np.ndarray, *, target_frames: int, pad_value: float
) -> np.ndarray:
    """Pad or center-crop mel spectrogram to the target number of frames."""
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


def audio_chunking(
    y: np.ndarray,
    sr: int,
    chunk_duration: float = 10.0,
    overlap_duration: float = 0.5,
    *,
    return_mels: bool = False,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    target_duration: float | None = None,
) -> List[np.ndarray | torch.Tensor]:
    """
    Split audio waveform into overlapping chunks. Optionally convert each chunk
    into a model-ready mel tensor shaped [1, n_mels, T].

    Args:
        y: 1D waveform (n_samples,)
        sr: sample rate
        chunk_duration: duration of each chunk in seconds
        overlap_duration: overlap between chunks in seconds
        return_mels: when True, return normalized mel tensors instead of waveforms
        n_mels, n_fft, hop_length: mel/STFT parameters (used when return_mels=True)
        target_duration: duration (sec) to pad/crop mel chunks to. Defaults to
            chunk_duration when None.

    Returns:
        List of waveform arrays or mel tensors (each [1, n_mels, T]).
    """
    if y.size == 0:
        return []

    chunk_size = int(chunk_duration * sr)
    overlap_size = int(overlap_duration * sr)
    step_size = max(1, chunk_size - overlap_size)

    wave_chunks: list[np.ndarray] = []
    for start in range(0, len(y), step_size):
        end = start + chunk_size
        chunk = y[start:end]
        if len(chunk) < chunk_size:
            pad_width = chunk_size - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode="constant", constant_values=0)
        wave_chunks.append(chunk)
        if end >= len(y):
            break

    if not return_mels:
        return wave_chunks

    mels: list[torch.Tensor] = []
    target_dur = target_duration if target_duration is not None else chunk_duration
    target_frames = int(target_dur * sr / hop_length)

    for chunk in wave_chunks:
        mel, _, _ = audio_to_mel_spectrogram(
            y=chunk,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            to_db=True,
        )
        mel = _pad_or_crop_mel_chunk(mel, target_frames=target_frames, pad_value=mel.min())
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        mels.append(torch.from_numpy(mel).float().unsqueeze(0))  # [1, n_mels, T]

    return mels
