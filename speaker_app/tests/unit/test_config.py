from pathlib import Path

import pytest

from speaker_app.config import PACKAGE_ROOT, PROJECT_ROOT, load_config


@pytest.fixture(autouse=True)
def clear_password_hash_env(monkeypatch):
    monkeypatch.delenv("ENROLLMENT_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("ENROLLMENT_PASSWORD_HASH_FILE", raising=False)


def test_enrollment_audio_retention_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("APP_RETAIN_ENROLLMENT_AUDIO", raising=False)
    assert load_config("development").retain_enrollment_audio


def test_raspberry_pi_profile_uses_project_paths_when_run_natively(monkeypatch):
    for name in ("APP_DATA_DIR", "APP_MODEL_DIR", "APP_MODEL_PATH", "APP_LOG_DIR"):
        monkeypatch.delenv(name, raising=False)

    config = load_config("raspberry-pi")

    assert config.database_path == PACKAGE_ROOT / "data" / "speakers.db"
    assert config.model_path == PROJECT_ROOT / "output" / "ecapa_tdnn_protonet_model.pth"
    assert config.log_dir == PACKAGE_ROOT / "logs"


def test_raspberry_pi_profile_honors_path_overrides(monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", "/tmp/protonet/data")
    monkeypatch.setenv("APP_MODEL_DIR", "/tmp/protonet/models")
    monkeypatch.setenv("APP_LOG_DIR", "/tmp/protonet/logs")
    monkeypatch.delenv("APP_MODEL_PATH", raising=False)

    config = load_config("raspberry-pi")

    assert config.database_path == Path("/tmp/protonet/data/speakers.db")
    assert config.model_path == Path("/tmp/protonet/models/ecapa_tdnn_protonet_model.pth")
    assert config.log_dir == Path("/tmp/protonet/logs")


def test_legacy_container_paths_fall_back_to_project_paths(monkeypatch):
    monkeypatch.setenv("APP_DATA_DIR", "/app/data")
    monkeypatch.setenv("APP_MODEL_DIR", "/app/models")
    monkeypatch.setenv("APP_MODEL_PATH", "/app/models/ecapa_tdnn_protonet_model.pth")
    monkeypatch.setenv("APP_LOG_DIR", "/app/logs")

    config = load_config("raspberry-pi")

    assert config.database_path == PACKAGE_ROOT / "data" / "speakers.db"
    assert config.model_path == PROJECT_ROOT / "output" / "ecapa_tdnn_protonet_model.pth"
    assert config.log_dir == PACKAGE_ROOT / "logs"


def test_password_hash_defaults_to_data_secret_file(monkeypatch, tmp_path):
    for name in (
        "APP_MODEL_PATH",
        "ENROLLMENT_PASSWORD_HASH",
        "ENROLLMENT_PASSWORD_HASH_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

    (tmp_path / "enrollment_password_hash").write_text("scrypt$stored\n", encoding="utf-8")

    config = load_config("raspberry-pi")

    assert config.password_hash == "scrypt$stored"


def test_password_hash_env_takes_precedence_over_data_secret_file(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ENROLLMENT_PASSWORD_HASH", "scrypt$env")
    monkeypatch.delenv("ENROLLMENT_PASSWORD_HASH_FILE", raising=False)
    (tmp_path / "enrollment_password_hash").write_text("scrypt$stored\n", encoding="utf-8")

    config = load_config("raspberry-pi")

    assert config.password_hash == "scrypt$env"


def test_missing_password_hash_file_env_falls_back_to_data_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(
        "ENROLLMENT_PASSWORD_HASH_FILE",
        str(tmp_path / "missing" / "enrollment_password_hash"),
    )
    (tmp_path / "enrollment_password_hash").write_text("scrypt$stored\n", encoding="utf-8")

    config = load_config("raspberry-pi")

    assert config.password_hash == "scrypt$stored"
