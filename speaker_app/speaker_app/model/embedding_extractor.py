from __future__ import annotations

from pathlib import Path
import logging
import time
from typing import Protocol

import numpy as np
import soundfile as sf

from speaker_app.config import AppConfig
from speaker_app.services.embedding_math import l2_normalize


LOGGER = logging.getLogger(__name__)


class EmbeddingExtractor(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

    def extract(self, audio_path: Path) -> np.ndarray: ...


class _AudioPreprocessor:
    def __init__(self, config: AppConfig) -> None:
        import torch
        import torchaudio.transforms as transforms

        self.torch = torch
        self.config = config
        self.target_samples = int(config.sample_rate * config.model_audio_duration_seconds)
        self.target_frames = int(
            config.model_audio_duration_seconds * config.sample_rate / config.hop_length
        )
        self.mel = transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=0.0,
            f_max=config.sample_rate / 2,
            power=2.0,
        )
        self.to_db = transforms.AmplitudeToDB(stype="power")

    def __call__(self, audio_path: Path):
        import torch.nn.functional as functional
        import torchaudio.functional as audio_functional

        try:
            waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot read audio file: {audio_path}") from exc
        mono = np.asarray(waveform.mean(axis=1), dtype=np.float32)
        if mono.size == 0 or not np.all(np.isfinite(mono)):
            raise ValueError(f"Audio file is empty or invalid: {audio_path}")
        tensor = self.torch.from_numpy(mono)
        if sample_rate != self.config.sample_rate:
            tensor = audio_functional.resample(tensor, sample_rate, self.config.sample_rate)

        if tensor.shape[0] > self.target_samples:
            start = (tensor.shape[0] - self.target_samples) // 2
            tensor = tensor[start : start + self.target_samples]
        elif tensor.shape[0] < self.target_samples:
            tensor = functional.pad(tensor, (0, self.target_samples - tensor.shape[0]))

        mel = self.to_db(self.mel(tensor.float()))
        if mel.shape[1] > self.target_frames:
            start = (mel.shape[1] - self.target_frames) // 2
            mel = mel[:, start : start + self.target_frames]
        elif mel.shape[1] < self.target_frames:
            mel = functional.pad(
                mel, (0, self.target_frames - mel.shape[1]), value=float(mel.min().item())
            )
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel.unsqueeze(0).float()


class TorchEmbeddingExtractor:
    def __init__(self, config: AppConfig) -> None:
        import torch

        from speaker_app.model.ecapa_tdnn import ECAPATDNNBackbone

        if not config.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {config.model_path}")
        provider = config.inference_provider.lower()
        if provider == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
        self.device = torch.device("cuda" if provider == "cuda" else "cpu")
        LOGGER.info(
            "Loading PyTorch embedding model (device=%s, model=%s)",
            self.device,
            config.model_path.name,
        )
        started = time.perf_counter()
        self._model_version = config.model_version
        self._embedding_dimension = config.embedding_dimension
        self.preprocess = _AudioPreprocessor(config)
        self.model = ECAPATDNNBackbone(
            n_mels=config.n_mels, embedding_dimension=config.embedding_dimension
        )
        try:
            payload = torch.load(config.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(config.model_path, map_location="cpu")
        if isinstance(payload, dict) and not all(torch.is_tensor(value) for value in payload.values()):
            payload = next(
                (payload[key] for key in ("state_dict", "model_state_dict", "model") if key in payload),
                payload,
            )
        if payload and all(key.startswith("module.") for key in payload):
            payload = {key.removeprefix("module."): value for key, value in payload.items()}
        self.model.load_state_dict(payload, strict=True)
        self.model.to(self.device).eval()
        LOGGER.info("PyTorch embedding model loaded (%.3fs)", time.perf_counter() - started)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def extract(self, audio_path: Path) -> np.ndarray:
        import torch

        started = time.perf_counter()
        LOGGER.debug("PyTorch embedding extraction started")
        mel = self.preprocess(Path(audio_path)).to(self.device)
        with torch.inference_mode():
            embedding = self.model(mel).squeeze(0).cpu().numpy()
        result = l2_normalize(embedding)
        LOGGER.debug(
            "PyTorch embedding extraction finished (dimension=%d, %.3fs)",
            result.size,
            time.perf_counter() - started,
        )
        return result


class OnnxEmbeddingExtractor:
    def __init__(self, config: AppConfig) -> None:
        import onnxruntime as ort

        if not config.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {config.model_path}")
        requested = config.inference_provider
        provider = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }.get(requested.lower(), requested)
        if not provider.endswith("ExecutionProvider"):
            raise ValueError(f"Invalid ONNX Runtime provider: {requested}")
        LOGGER.info(
            "Loading ONNX embedding model (provider=%s, model=%s)",
            provider,
            config.model_path.name,
        )
        started = time.perf_counter()
        self.session = ort.InferenceSession(str(config.model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self._model_version = config.model_version
        self._embedding_dimension = config.embedding_dimension
        self.preprocess = _AudioPreprocessor(config)
        LOGGER.info("ONNX embedding model loaded (%.3fs)", time.perf_counter() - started)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def extract(self, audio_path: Path) -> np.ndarray:
        started = time.perf_counter()
        LOGGER.debug("ONNX embedding extraction started")
        model_input = self.preprocess(Path(audio_path)).numpy()
        embedding = self.session.run(None, {self.input_name: model_input})[0]
        result = l2_normalize(np.asarray(embedding, dtype=np.float32))
        if result.size != self.embedding_dimension:
            raise ValueError(
                f"ONNX model returned {result.size} values; expected {self.embedding_dimension}"
            )
        LOGGER.debug(
            "ONNX embedding extraction finished (dimension=%d, %.3fs)",
            result.size,
            time.perf_counter() - started,
        )
        return result


def load_embedding_extractor(config: AppConfig) -> EmbeddingExtractor:
    if config.model_path.suffix.lower() == ".onnx":
        return OnnxEmbeddingExtractor(config)
    if config.model_path.suffix.lower() in {".pt", ".pth"}:
        return TorchEmbeddingExtractor(config)
    raise ValueError("APP_MODEL_PATH must point to an ONNX or PyTorch model")
