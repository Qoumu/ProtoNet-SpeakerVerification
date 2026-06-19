from __future__ import annotations

import threading
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from speaker_app.domain import AudioValidationResult, RecordingResult


LOGGER = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    """Raised when gain or denoising cannot process captured samples."""


class AudioService(Protocol):
    def status(self) -> tuple[bool, str]: ...

    def record_clip(self, output_path: Path, duration_seconds: float) -> RecordingResult: ...

    def stop_recording(self) -> bool: ...

    def validate_audio(self, audio_path: Path) -> AudioValidationResult: ...


class AudioValidationMixin:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        min_duration_seconds: float = 2.0,
        min_rms: float = 0.008,
        clipping_ratio_limit: float = 0.02,
        gain_db: float = 20.0,
        denoise_enabled: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_duration_seconds = min_duration_seconds
        self.min_rms = min_rms
        self.clipping_ratio_limit = clipping_ratio_limit
        self.gain_db = gain_db
        self.denoise_enabled = denoise_enabled

    def processing_status(self) -> tuple[bool, str]:
        if not self.denoise_enabled:
            return True, "Denoising disabled"
        try:
            import noisereduce  # noqa: F401
        except ImportError:
            return (
                False,
                "Denoising requires noisereduce. Launch with ~/Projects/.venv/bin/python.",
            )
        return True, "Gain and denoising ready"

    def process_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gain and denoising without trimming or changing sample count."""
        processed = np.asarray(waveform, dtype=np.float32).reshape(-1)
        original_samples = processed.size
        if original_samples == 0:
            raise ValueError("Cannot process empty audio")

        gain = 10.0 ** (self.gain_db / 20.0)
        processed = processed * gain
        peak = float(np.max(np.abs(processed)))
        if peak > 0.98:
            LOGGER.debug("Gain limiter applied (pre_limit_peak=%.4f)", peak)
            processed = processed * (0.98 / peak)

        if self.denoise_enabled:
            try:
                import noisereduce as noise_reduction
            except ImportError as exc:
                raise AudioProcessingError(
                    "Audio denoising requires the noisereduce package"
                ) from exc
            LOGGER.debug("Non-stationary audio denoising started")
            try:
                processed = np.asarray(
                    noise_reduction.reduce_noise(
                        y=processed,
                        sr=sample_rate,
                        stationary=False,
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception as exc:
                raise AudioProcessingError("Audio denoising failed") from exc
            LOGGER.debug("Non-stationary audio denoising finished")

        if processed.size != original_samples:
            raise AudioProcessingError("Audio processing unexpectedly changed clip length")
        if not np.all(np.isfinite(processed)):
            raise AudioProcessingError("Audio processing produced invalid samples")
        return np.clip(processed, -0.98, 0.98).astype(np.float32, copy=False)

    def validate_audio(self, audio_path: Path) -> AudioValidationResult:
        LOGGER.debug("Validating recorded audio")
        try:
            waveform, sample_rate = sf.read(
                str(audio_path), dtype="float32", always_2d=True
            )
        except (OSError, RuntimeError) as exc:
            return AudioValidationResult(False, 0.0, 0.0, 0.0, f"Audio cannot be read: {exc}")

        if waveform.size == 0 or sample_rate <= 0:
            return AudioValidationResult(False, 0.0, 0.0, 0.0, "Audio is empty")
        mono = waveform.mean(axis=1)
        duration = mono.size / sample_rate
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))

        clipping_ratio = float(np.mean(np.abs(mono) >= 0.99))
        LOGGER.debug(
            "Audio metrics (no VAD): duration=%.3fs rms=%.6f peak=%.4f clipping=%.4f",
            duration,
            rms,
            peak,
            clipping_ratio,
        )

        if duration < self.min_duration_seconds:
            message = f"Audio is too short ({duration:.1f}s)"
        elif rms < self.min_rms:
            message = "Audio is too quiet"
        elif clipping_ratio > self.clipping_ratio_limit:
            message = "Audio is heavily clipped; move farther from the microphone"
        else:
            LOGGER.info(
                "Audio accepted without VAD (duration=%.2fs, peak=%.3f)",
                duration,
                peak,
            )
            return AudioValidationResult(True, duration, 0.0, peak, "Audio accepted")
        LOGGER.warning("Audio rejected: %s", message)
        return AudioValidationResult(False, duration, 0.0, peak, message)


class SoundDeviceAudioService(AudioValidationMixin):
    """Live mono WAV capture through python-sounddevice."""

    def __init__(self, *, audio_device: str | int | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.audio_device = audio_device
        self._stop_event = threading.Event()
        self._recording_lock = threading.Lock()
        self._recording_active = False

    @staticmethod
    def _sounddevice():
        try:
            import sounddevice
        except ImportError as exc:
            raise RuntimeError("Microphone support requires the sounddevice package") from exc
        return sounddevice

    def status(self) -> tuple[bool, str]:
        processing_ready, processing_message = self.processing_status()
        if not processing_ready:
            return False, processing_message
        try:
            sd = self._sounddevice()
            device = sd.query_devices(self.audio_device, "input")
            if int(device.get("max_input_channels", 0)) < 1:
                return False, "No microphone input channels"
            LOGGER.debug(
                "Selected input device: name=%s channels=%s default_rate=%s",
                device.get("name", "unknown"),
                device.get("max_input_channels", "unknown"),
                device.get("default_samplerate", "unknown"),
            )
            return True, str(device.get("name", "Ready"))
        except Exception as exc:
            return False, str(exc)

    def record_clip(self, output_path: Path, duration_seconds: float) -> RecordingResult:
        output_path = Path(output_path)
        chunks: list[np.ndarray] = []
        stream = None

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            del frames, time_info
            if status:
                LOGGER.warning("Microphone stream status: %s", status)
            chunks.append(indata.copy())

        try:
            sd = self._sounddevice()
            LOGGER.info(
                "Recording started (maximum=%.1fs, sample_rate=%d, device=%s)",
                duration_seconds,
                self.sample_rate,
                self.audio_device if self.audio_device is not None else "default",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with self._recording_lock:
                self._recording_active = True
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.audio_device,
                callback=callback,
            )
            stream.start()
            self._stop_event.wait(timeout=max(0.1, duration_seconds))
            # A graceful PortAudio stop can block on some USB devices. Input has
            # no pending output to drain, so abort is the correct bounded exit.
            stream.abort(ignore_errors=True)
            stream.close(ignore_errors=True)
            stream = None

            if not chunks:
                raise RuntimeError("The microphone returned no audio")
            recording = np.concatenate(chunks, axis=0)
            normalized = recording.astype(np.float32).reshape(-1) / 32768.0
            processed = self.process_audio(normalized, self.sample_rate)
            sf.write(str(output_path), processed, self.sample_rate, subtype="PCM_16")
            recorded_duration = recording.shape[0] / self.sample_rate
            LOGGER.info("Recording stopped (duration=%.3fs)", recorded_duration)
            return RecordingResult(
                True,
                output_path,
                recorded_duration,
                self.sample_rate,
                "Recording complete",
            )
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            if isinstance(exc, AudioProcessingError):
                LOGGER.exception("Captured audio processing failed")
                message = f"Audio processing failed: {exc}"
            else:
                LOGGER.exception("Microphone recording failed")
                message = f"Microphone unavailable: {exc}"
            return RecordingResult(False, None, 0.0, self.sample_rate, message)
        finally:
            if stream is not None:
                stream.abort(ignore_errors=True)
                stream.close(ignore_errors=True)
            with self._recording_lock:
                self._recording_active = False
            self._stop_event.clear()

    def stop_recording(self) -> bool:
        """Request that the current recording stream close as soon as possible."""
        with self._recording_lock:
            was_active = self._recording_active
        self._stop_event.set()
        LOGGER.info("Recording stop requested (active=%s)", was_active)
        return was_active


class FileAudioService(AudioValidationMixin):
    """Copy a configured WAV file in place of recording for local testing."""

    def __init__(self, source_path: Path, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.source_path = Path(source_path)

    def status(self) -> tuple[bool, str]:
        processing_ready, processing_message = self.processing_status()
        if not processing_ready:
            return False, processing_message
        if not self.source_path.is_file():
            return False, f"Test audio not found: {self.source_path}"
        LOGGER.debug("File-backed audio service ready: %s", self.source_path.name)
        return True, f"Test file: {self.source_path.name}"

    def record_clip(self, output_path: Path, duration_seconds: float) -> RecordingResult:
        del duration_seconds
        if not self.source_path.is_file():
            return RecordingResult(False, None, 0.0, self.sample_rate, "Test audio file is unavailable")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        waveform, sample_rate = sf.read(
            str(self.source_path), dtype="float32", always_2d=True
        )
        mono = waveform.mean(axis=1)
        processed = self.process_audio(mono, sample_rate)
        sf.write(str(output_path), processed, sample_rate, subtype="PCM_16")
        info = sf.info(str(output_path))
        LOGGER.info("Loaded test audio (duration=%.3fs, sample_rate=%d)", info.duration, info.samplerate)
        return RecordingResult(
            True,
            output_path,
            info.duration,
            info.samplerate,
            "Test audio loaded",
        )

    def stop_recording(self) -> bool:
        return False
