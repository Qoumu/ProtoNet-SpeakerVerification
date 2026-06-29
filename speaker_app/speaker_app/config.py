from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
PASSWORD_HASH_FILE_NAME = "enrollment_password_hash"
VALID_PROFILES = {"development", "raspberry-pi"}
LEGACY_CONTAINER_ROOT = Path("/app")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _is_legacy_container_path(path: Path) -> bool:
    return path == LEGACY_CONTAINER_ROOT or LEGACY_CONTAINER_ROOT in path.parents


def _path_env(name: str, default: Path, *, ignore_legacy_container: bool = False) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if ignore_legacy_container and _is_legacy_container_path(path):
        return default
    return path


def _validate_profile(profile: str) -> str:
    if profile not in VALID_PROFILES:
        raise ValueError("profile must be 'development' or 'raspberry-pi'")
    return profile


def default_password_hash_path(
    profile: str | None = None, *, honor_secret_file_env: bool = True
) -> Path:
    _validate_profile(profile or os.getenv("APP_PROFILE", "development"))
    if honor_secret_file_env and os.getenv("ENROLLMENT_PASSWORD_HASH_FILE"):
        return Path(os.environ["ENROLLMENT_PASSWORD_HASH_FILE"]).expanduser()
    return (
        _path_env(
            "APP_DATA_DIR",
            PACKAGE_ROOT / "data",
            ignore_legacy_container=True,
        )
        / PASSWORD_HASH_FILE_NAME
    )


def _read_password_hash_file(path: Path, *, required: bool) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        if required:
            raise ValueError(f"Cannot read ENROLLMENT_PASSWORD_HASH_FILE: {path}") from None
        return None
    except OSError as exc:
        raise ValueError(f"Cannot read enrollment password hash file: {path}") from exc


def _password_hash(default_secret_path: Path) -> str | None:
    direct = os.getenv("ENROLLMENT_PASSWORD_HASH")
    if direct:
        return direct.strip()
    secret_path = os.getenv("ENROLLMENT_PASSWORD_HASH_FILE")
    if secret_path:
        configured_hash = _read_password_hash_file(Path(secret_path).expanduser(), required=False)
        if configured_hash:
            return configured_hash
    return _read_password_hash_file(default_secret_path, required=False)


@dataclass(frozen=True)
class AppConfig:
    profile: str
    windowed: bool
    enrollment_clip_count: int
    enrollment_clip_duration_seconds: float
    recognition_clip_duration_seconds: float
    recognition_threshold: float
    max_enrollment_password_attempts: int
    enrollment_authorization_timeout_seconds: int
    sample_rate: int
    audio_channels: int
    audio_dtype: str
    audio_device: str | int | None
    audio_backend: str
    test_audio_file: Path | None
    database_path: Path
    temporary_audio_dir: Path
    enrollment_audio_dir: Path
    retain_enrollment_audio: bool
    model_path: Path
    model_version: str
    inference_provider: str
    log_dir: Path
    log_level: str
    password_hash: str | None
    min_audio_duration_seconds: float
    min_rms: float
    clipping_ratio_limit: float
    audio_gain_db: float
    audio_denoise_enabled: bool
    n_mels: int = 80
    n_fft: int = 512
    hop_length: int = 256
    model_audio_duration_seconds: float = 3.0
    embedding_dimension: int = 192

    def create_directories(self) -> None:
        for path in (
            self.database_path.parent,
            self.temporary_audio_dir,
            self.enrollment_audio_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(profile: str | None = None, *, windowed: bool | None = None) -> AppConfig:
    selected_profile = _validate_profile(profile or os.getenv("APP_PROFILE", "development"))

    rpi = selected_profile == "raspberry-pi"
    # Project-relative defaults keep both profiles writable for native launches.
    # Deployments can still override any path with the matching APP_* variable.
    default_data = PACKAGE_ROOT / "data"
    default_models = PROJECT_ROOT / "output"
    default_logs = PACKAGE_ROOT / "logs"
    data_dir = _path_env("APP_DATA_DIR", default_data, ignore_legacy_container=True)
    model_dir = _path_env("APP_MODEL_DIR", default_models, ignore_legacy_container=True)
    configured_windowed = _bool_env("APP_WINDOWED", not rpi) if windowed is None else windowed

    device_value = os.getenv("APP_AUDIO_DEVICE")
    audio_device: str | int | None = None
    if device_value:
        audio_device = int(device_value) if device_value.isdigit() else device_value

    test_audio = os.getenv("APP_TEST_AUDIO_FILE")
    model_path = _path_env(
        "APP_MODEL_PATH",
        model_dir / "ecapa_tdnn_protonet_model.pth",
        ignore_legacy_container=True,
    )
    return AppConfig(
        profile=selected_profile,
        windowed=configured_windowed,
        enrollment_clip_count=int(os.getenv("APP_ENROLLMENT_CLIP_COUNT", "5")),
        enrollment_clip_duration_seconds=float(os.getenv("APP_ENROLLMENT_CLIP_DURATION", "5")),
        recognition_clip_duration_seconds=float(os.getenv("APP_RECOGNITION_CLIP_DURATION", "5")),
        recognition_threshold=float(os.getenv("APP_RECOGNITION_THRESHOLD", "0.55")),
        max_enrollment_password_attempts=int(os.getenv("APP_MAX_PASSWORD_ATTEMPTS", "3")),
        enrollment_authorization_timeout_seconds=int(os.getenv("APP_AUTH_TIMEOUT_SECONDS", "120")),
        sample_rate=int(os.getenv("APP_SAMPLE_RATE", "16000")),
        audio_channels=1,
        audio_dtype="int16",
        audio_device=audio_device,
        audio_backend=os.getenv("APP_AUDIO_BACKEND", "sounddevice"),
        test_audio_file=Path(test_audio).expanduser() if test_audio else None,
        database_path=_path_env(
            "APP_DATABASE_PATH",
            data_dir / "speakers.db",
            ignore_legacy_container=True,
        ),
        temporary_audio_dir=_path_env(
            "APP_TEMP_AUDIO_DIR",
            data_dir / "temporary_audio",
            ignore_legacy_container=True,
        ),
        enrollment_audio_dir=_path_env(
            "APP_ENROLLMENT_AUDIO_DIR",
            data_dir / "enrollment_audio",
            ignore_legacy_container=True,
        ),
        retain_enrollment_audio=_bool_env("APP_RETAIN_ENROLLMENT_AUDIO", True),
        model_path=model_path,
        model_version=os.getenv("APP_MODEL_VERSION", f"ecapa-tdnn:{model_path.name}"),
        inference_provider=os.getenv("APP_INFERENCE_PROVIDER", "cpu"),
        log_dir=_path_env("APP_LOG_DIR", default_logs, ignore_legacy_container=True),
        log_level=os.getenv("APP_LOG_LEVEL", "INFO" if rpi else "DEBUG"),
        password_hash=_password_hash(data_dir / PASSWORD_HASH_FILE_NAME),
        min_audio_duration_seconds=float(os.getenv("APP_MIN_AUDIO_DURATION", "2.0")),
        min_rms=float(os.getenv("APP_MIN_RMS", "0.008")),
        clipping_ratio_limit=float(os.getenv("APP_CLIPPING_RATIO_LIMIT", "0.02")),
        audio_gain_db=float(os.getenv("APP_AUDIO_GAIN_DB", "20.0")),
        audio_denoise_enabled=_bool_env("APP_AUDIO_DENOISE_ENABLED", True),
        n_mels=int(os.getenv("APP_N_MELS", "80")),
        n_fft=int(os.getenv("APP_N_FFT", "512")),
        hop_length=int(os.getenv("APP_HOP_LENGTH", "256")),
        model_audio_duration_seconds=float(os.getenv("APP_MODEL_AUDIO_DURATION", "3.0")),
        embedding_dimension=int(os.getenv("APP_EMBEDDING_DIMENSION", "192")),
    )
