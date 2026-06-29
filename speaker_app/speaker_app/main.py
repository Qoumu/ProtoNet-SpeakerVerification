from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from speaker_app.config import AppConfig, load_config


LOGGER = logging.getLogger(__name__)


class UnavailableExtractor:
    def __init__(self, config: AppConfig, reason: str) -> None:
        self.model_version = config.model_version
        self.embedding_dimension = config.embedding_dimension
        self.reason = reason

    def extract(self, audio_path: Path):
        del audio_path
        raise RuntimeError(self.reason)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speaker recognition touchscreen application")
    parser.add_argument(
        "--profile", choices=("development", "raspberry-pi"), default=None
    )
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--test-audio", type=Path, help="Use a WAV file instead of a microphone")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Override APP_LOG_LEVEL for terminal and file logging",
    )
    return parser.parse_args(argv)


def configure_logging(config: AppConfig) -> None:
    config.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=(
            logging.FileHandler(config.log_dir / "speaker_app.log"),
            logging.StreamHandler(sys.stdout),
        ),
        force=True,
    )
    # Keep development DEBUG useful by suppressing verbose dependency probes.
    for logger_name in ("torio", "matplotlib", "PIL", "numba"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import os

    if args.test_audio:
        os.environ["APP_AUDIO_BACKEND"] = "file"
        os.environ["APP_TEST_AUDIO_FILE"] = str(args.test_audio)
    if args.log_level:
        os.environ["APP_LOG_LEVEL"] = args.log_level
    config = load_config(args.profile, windowed=True if args.windowed else None)
    config.create_directories()
    configure_logging(config)
    LOGGER.info(
        "Starting speaker application (profile=%s, windowed=%s, python=%s)",
        config.profile,
        config.windowed,
        sys.executable,
    )
    LOGGER.debug(
        "Runtime configuration: audio_backend=%s sample_rate=%d model=%s provider=%s "
        "database=%s threshold=%.3f clips=%d gain_db=%.1f denoise=%s vad=false",
        config.audio_backend,
        config.sample_rate,
        config.model_path,
        config.inference_provider,
        config.database_path,
        config.recognition_threshold,
        config.enrollment_clip_count,
        config.audio_gain_db,
        config.audio_denoise_enabled,
    )

    from PySide6.QtWidgets import QApplication

    from speaker_app.app_controller import AppController
    from speaker_app.model import load_embedding_extractor
    from speaker_app.services.audio_service import FileAudioService, SoundDeviceAudioService
    from speaker_app.services.enrollment_authorization_service import (
        EnrollmentAuthorizationService,
    )
    from speaker_app.services.enrollment_service import EnrollmentService
    from speaker_app.services.recognition_service import RecognitionService
    from speaker_app.services.speaker_repository import SpeakerRepository
    from speaker_app.ui.main_window import MainWindow

    repository = SpeakerRepository(config.database_path)
    database_status = (True, str(config.database_path))
    try:
        repository.initialize()
        LOGGER.info("Database ready (%d enrolled speakers)", repository.count())
    except Exception as exc:
        logging.exception("Database initialization failed")
        database_status = (False, str(exc))

    audio_kwargs = dict(
        sample_rate=config.sample_rate,
        min_duration_seconds=config.min_audio_duration_seconds,
        min_rms=config.min_rms,
        clipping_ratio_limit=config.clipping_ratio_limit,
        gain_db=config.audio_gain_db,
        denoise_enabled=config.audio_denoise_enabled,
    )
    if config.audio_backend == "file":
        if config.test_audio_file is None:
            raise SystemExit("APP_TEST_AUDIO_FILE is required when APP_AUDIO_BACKEND=file")
        audio = FileAudioService(config.test_audio_file, **audio_kwargs)
    elif config.audio_backend == "sounddevice":
        audio = SoundDeviceAudioService(audio_device=config.audio_device, **audio_kwargs)
    else:
        raise SystemExit("APP_AUDIO_BACKEND must be 'sounddevice' or 'file'")
    microphone_status = audio.status()
    LOGGER.log(
        logging.INFO if microphone_status[0] else logging.ERROR,
        "Microphone status: %s",
        microphone_status[1],
    )

    try:
        extractor = load_embedding_extractor(config)
        model_status = (True, config.model_path.name)
        LOGGER.info(
            "Embedding model ready (version=%s, dimension=%d)",
            extractor.model_version,
            extractor.embedding_dimension,
        )
    except Exception as exc:
        logging.exception("Model initialization failed")
        extractor = UnavailableExtractor(config, str(exc))
        model_status = (False, str(exc))

    authorization = EnrollmentAuthorizationService(
        config.password_hash,
        config.max_enrollment_password_attempts,
        config.enrollment_authorization_timeout_seconds,
    )
    authorization_status = (
        authorization.configured,
        "Configured"
        if authorization.configured
        else "Enrollment password is missing. Run python -m speaker_app.password_hash.",
    )
    LOGGER.log(
        logging.INFO if authorization.configured else logging.WARNING,
        "Enrollment authorization: %s",
        authorization_status[1],
    )
    enrollment = EnrollmentService(repository, extractor, config.enrollment_clip_count)
    recognition = RecognitionService(repository, extractor, config.recognition_threshold)

    qt_app = QApplication([sys.argv[0], *(argv or sys.argv[1:])])
    window = MainWindow()
    window.pages["home"].set_system_status(
        microphone=microphone_status,
        model=model_status,
        database=database_status,
        authorization=authorization_status,
    )
    controller = AppController(
        window, config, audio, authorization, repository, enrollment, recognition
    )
    window.controller = controller
    if config.windowed:
        window.resize(800, 480)
        window.show()
    else:
        window.showFullScreen()
    LOGGER.info("GUI ready")
    exit_code = qt_app.exec()
    LOGGER.info("Speaker application stopped (exit_code=%d)", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
