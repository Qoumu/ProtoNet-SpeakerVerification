from stat import S_IMODE

import pytest

from speaker_app.password_hash import main as password_hash_main
from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService


def test_authorization_accepts_password_and_expires(monkeypatch):
    password_hash = EnrollmentAuthorizationService.create_password_hash("correct")
    service = EnrollmentAuthorizationService(password_hash, timeout_seconds=10)
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])

    assert service.verify_password("correct").accepted
    assert service.authorized
    now[0] = 111.0
    assert not service.authorized


def test_authorization_limits_failed_attempts():
    password_hash = EnrollmentAuthorizationService.create_password_hash("correct")
    service = EnrollmentAuthorizationService(password_hash, max_attempts=2)

    assert service.verify_password("wrong").remaining_attempts == 1
    result = service.verify_password("still-wrong")
    assert not result.accepted
    assert result.remaining_attempts == 0
    assert not service.verify_password("correct").accepted


def test_missing_hash_fails_closed():
    result = EnrollmentAuthorizationService(None).verify_password("anything")
    assert not result.accepted
    assert "not configured" in result.message


def test_password_hash_command_writes_persistent_secret(monkeypatch, tmp_path):
    prompts = iter(("secret", "secret"))
    monkeypatch.setattr("getpass.getpass", lambda _: next(prompts))
    secret_file = tmp_path / "enrollment_password_hash"

    assert password_hash_main(["--output", str(secret_file)]) == 0

    stored_hash = secret_file.read_text(encoding="utf-8").strip()
    assert stored_hash.startswith("scrypt$")
    assert S_IMODE(secret_file.stat().st_mode) == 0o600
    assert EnrollmentAuthorizationService(stored_hash).verify_password("secret").accepted


def test_password_hash_command_uses_env_file_data_dir(monkeypatch, tmp_path):
    prompts = iter(("secret", "secret"))
    monkeypatch.setattr("getpass.getpass", lambda _: next(prompts))
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.delenv("ENROLLMENT_PASSWORD_HASH_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("APP_DATA_DIR=custom_data\n", encoding="utf-8")

    assert password_hash_main(["--env-file", str(env_file)]) == 0

    assert (tmp_path / "custom_data" / "enrollment_password_hash").exists()


def test_password_hash_command_ignores_runtime_secret_file_env(monkeypatch, tmp_path):
    prompts = iter(("secret", "secret"))
    monkeypatch.setattr("getpass.getpass", lambda _: next(prompts))
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ENROLLMENT_PASSWORD_HASH_FILE", "/run/secrets/enrollment_password_hash")

    assert password_hash_main(["--env-file", str(tmp_path / "missing.env")]) == 0

    assert (tmp_path / "data" / "enrollment_password_hash").exists()


def test_password_hash_command_ignores_legacy_container_data_dir(monkeypatch, tmp_path):
    prompts = iter(("secret", "secret"))
    monkeypatch.setattr("getpass.getpass", lambda _: next(prompts))
    monkeypatch.setattr("speaker_app.config.PACKAGE_ROOT", tmp_path)
    monkeypatch.setenv("APP_DATA_DIR", "/app/data")
    monkeypatch.delenv("ENROLLMENT_PASSWORD_HASH_FILE", raising=False)

    assert password_hash_main(["--env-file", str(tmp_path / "missing.env")]) == 0

    assert (tmp_path / "data" / "enrollment_password_hash").exists()


def test_password_hash_command_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    prompts = iter(("secret", "secret"))
    monkeypatch.setattr("getpass.getpass", lambda _: next(prompts))
    secret_file = tmp_path / "enrollment_password_hash"
    secret_file.write_text("old\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        password_hash_main(["--output", str(secret_file)])

    assert secret_file.read_text(encoding="utf-8") == "old\n"
