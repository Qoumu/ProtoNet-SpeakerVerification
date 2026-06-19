import numpy as np
import soundfile as sf
import time
from concurrent.futures import ThreadPoolExecutor

from speaker_app.services.audio_service import FileAudioService, SoundDeviceAudioService


def write_tone(path, *, seconds=3.0, amplitude=0.2):
    sample_rate = 16000
    times = np.arange(int(seconds * sample_rate)) / sample_rate
    sf.write(path, amplitude * np.sin(2 * np.pi * 220 * times), sample_rate)


def test_file_audio_service_records_and_validates(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    write_tone(source)
    service = FileAudioService(source)

    assert service.status()[0]
    recording = service.record_clip(output, 5.0)
    assert recording.success
    assert service.validate_audio(output).accepted


def test_audio_validation_rejects_silence(tmp_path):
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros(16000 * 3), 16000)
    result = FileAudioService(source).validate_audio(source)
    assert not result.accepted
    assert "quiet" in result.message.lower()


def test_live_recording_can_be_stopped(monkeypatch, tmp_path):
    service = SoundDeviceAudioService(denoise_enabled=False)

    class FakeInputStream:
        def __init__(self, *, callback, **kwargs):
            del kwargs
            self.callback = callback

        def start(self):
            samples = np.ones((1600, 1), dtype=np.int16)
            self.callback(samples, len(samples), None, None)

        def abort(self, ignore_errors=True):
            del ignore_errors

        def close(self, ignore_errors=True):
            del ignore_errors

    class FakeSoundDevice:
        InputStream = FakeInputStream

    monkeypatch.setattr(service, "_sounddevice", lambda: FakeSoundDevice())
    output = tmp_path / "stopped.wav"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.record_clip, output, 10.0)
        for _ in range(100):
            if service._recording_active:
                break
            time.sleep(0.001)
        assert service.stop_recording()
        result = future.result(timeout=1.0)

    assert result.success
    assert result.duration_seconds == 0.1
    assert output.exists()


def test_processing_preserves_full_clip_without_vad(tmp_path):
    source = tmp_path / "speech_with_silence.wav"
    output = tmp_path / "processed.wav"
    sample_rate = 16000
    waveform = np.zeros(sample_rate * 3, dtype=np.float32)
    times = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform[sample_rate : sample_rate * 2] = 0.1 * np.sin(2 * np.pi * 220 * times)
    sf.write(source, waveform, sample_rate)
    service = FileAudioService(source, gain_db=0.0, denoise_enabled=False)

    result = service.record_clip(output, 5.0)
    processed, _ = sf.read(output, dtype="float32")

    assert result.success
    assert processed.shape == waveform.shape
    assert np.max(np.abs(processed[:sample_rate])) == 0.0
    assert np.max(np.abs(processed[sample_rate * 2 :])) == 0.0


def test_gain_increases_level_without_changing_length(tmp_path):
    source = tmp_path / "quiet.wav"
    output = tmp_path / "gained.wav"
    sample_rate = 16000
    times = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = 0.01 * np.sin(2 * np.pi * 220 * times)
    sf.write(source, waveform, sample_rate)
    service = FileAudioService(source, gain_db=6.0, denoise_enabled=False)

    service.record_clip(output, 5.0)
    processed, _ = sf.read(output, dtype="float32")

    assert processed.shape == waveform.shape
    assert np.sqrt(np.mean(processed**2)) > 1.9 * np.sqrt(np.mean(waveform**2))
