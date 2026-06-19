from speaker_app.config import load_config


def test_enrollment_audio_retention_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("APP_RETAIN_ENROLLMENT_AUDIO", raising=False)
    assert load_config("development").retain_enrollment_audio
