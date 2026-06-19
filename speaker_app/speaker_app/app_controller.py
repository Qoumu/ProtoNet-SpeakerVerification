from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool

from speaker_app.config import AppConfig
from speaker_app.domain import RecognitionResult
from speaker_app.services.audio_service import AudioService
from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService
from speaker_app.services.enrollment_service import EnrollmentService
from speaker_app.services.recognition_service import RecognitionService
from speaker_app.services.speaker_repository import SpeakerRepository
from speaker_app.ui.main_window import MainWindow
from speaker_app.workers.backend_worker import BackendWorker


LOGGER = logging.getLogger(__name__)


class AppController:
    """Own navigation and serialize all audio/model/database operations."""

    def __init__(
        self,
        window: MainWindow,
        config: AppConfig,
        audio: AudioService,
        authorization: EnrollmentAuthorizationService,
        repository: SpeakerRepository,
        enrollment: EnrollmentService,
        recognition: RecognitionService,
    ) -> None:
        self.window = window
        self.config = config
        self.audio = audio
        self.authorization = authorization
        self.repository = repository
        self.enrollment = enrollment
        self.recognition = recognition
        self.thread_pool = QThreadPool.globalInstance()
        self.busy = False
        self._operation_id = 0
        self._active_workers: dict[int, tuple[object, ...]] = {}
        self.pending_speaker_id = ""
        self.pending_display_name = ""
        self.accepted_clips: list[Path] = []
        self.current_clip: Path | None = None
        self.enrollment_started = False
        self.cancel_requested = False
        self._connect_pages()
        LOGGER.debug("Application controller initialized")

    def _connect_pages(self) -> None:
        home = self.window.pages["home"]
        info = self.window.pages["enroll_info"]
        auth = self.window.pages["authorization"]
        recording = self.window.pages["enroll_recording"]
        success = self.window.pages["enroll_success"]
        recognition = self.window.pages["recognition"]
        result = self.window.pages["recognition_result"]

        home.enroll_requested.connect(lambda: self.window.show_page("enroll_info"))
        home.recognize_requested.connect(self.open_recognition)
        home.remove_speaker_requested.connect(self.remove_speaker)
        info.continue_requested.connect(self.continue_enrollment)
        info.back_requested.connect(self.cancel_enrollment)
        auth.confirm_requested.connect(self.verify_authorization)
        auth.back_requested.connect(self.back_to_enrollment_info)
        recording.record_requested.connect(self.record_enrollment_clip)
        recording.next_requested.connect(self.accept_current_clip)
        recording.cancel_requested.connect(self.cancel_enrollment)
        success.home_requested.connect(self.finish_enrollment)
        recognition.start_requested.connect(self.start_recognition)
        recognition.back_requested.connect(lambda: self.window.show_page("home"))
        result.again_requested.connect(self.open_recognition)
        result.home_requested.connect(lambda: self.window.show_page("home"))

    def remove_speaker(self) -> None:
        if self.busy:
            self.window.show_error("Wait for the current operation to finish.")
            return

        def profiles_loaded(payload: object) -> None:
            profiles = list(payload)
            if not profiles:
                self.window.show_info("No speaker profiles are enrolled.")
                return
            speaker_id = self.window.choose_speaker(profiles)
            if speaker_id is None:
                return
            profile = next(item for item in profiles if item.speaker_id == speaker_id)
            password = self.window.request_enrollment_password(
                f"Enter the enrollment password to remove {speaker_id}"
            )
            if password is None:
                return
            authorization = self.authorization.verify_password(password)
            password = ""
            if not authorization.accepted:
                suffix = (
                    f" ({authorization.remaining_attempts} attempts remaining)"
                    if authorization.remaining_attempts
                    else ""
                )
                self.window.show_error(authorization.message + suffix)
                return
            if not self.window.confirm_speaker_delete(
                profile.speaker_id, profile.display_name
            ):
                self.authorization.reset()
                return

            LOGGER.warning("Authorized speaker removal requested (speaker_id=%s)", speaker_id)
            self.authorization.reset()

            def completed(deleted: object) -> None:
                if not bool(deleted):
                    self.window.show_error("Speaker profile was not found.")
                    return
                remaining = max(0, len(profiles) - 1)
                home = self.window.pages["home"]
                home.database_status.setText(
                    f"Database: Ready - {remaining} enrolled speaker"
                    f"{'s' if remaining != 1 else ''}"
                )
                self.window.show_info(
                    f"Removed speaker profile {speaker_id}.\n"
                    "Enrollment audio files were preserved."
                )

            self._run(lambda: self.repository.delete(speaker_id), completed)

        self._run(self.repository.get_all, profiles_loaded)

    def _run(
        self,
        function: Callable[[], Any],
        on_result: Callable[[Any], None],
        on_error: Callable[[], None] | None = None,
    ) -> None:
        if self.busy:
            LOGGER.warning("Ignored duplicate backend operation while another task is active")
            return
        self.busy = True
        self._operation_id += 1
        operation_id = self._operation_id
        LOGGER.debug("Submitting backend operation")
        worker = BackendWorker(function)

        def handle_result(result: object) -> None:
            if operation_id != self._operation_id:
                LOGGER.warning("Ignored stale backend result (operation=%d)", operation_id)
                return
            self.busy = False
            LOGGER.debug("Backend result received; controls unlocked (operation=%d)", operation_id)
            on_result(result)

        def handle_error(message: str) -> None:
            if operation_id != self._operation_id:
                LOGGER.warning("Ignored stale backend error (operation=%d)", operation_id)
                return
            self.busy = False
            LOGGER.error("Backend operation reported an error: %s", message)
            self.window.show_error(message)
            if on_error:
                on_error()

        def handle_finished() -> None:
            if operation_id == self._operation_id and self.busy:
                LOGGER.warning(
                    "Backend operation finished without result; forcing controls unlocked "
                    "(operation=%d)",
                    operation_id,
                )
                self.busy = False
            self._active_workers.pop(operation_id, None)

        worker.signals.result.connect(handle_result)
        worker.signals.error.connect(handle_error)
        worker.signals.finished.connect(handle_finished)
        # PySide does not keep strong references to plain Python slot closures.
        # Retain the runnable and handlers until its final queued signal arrives.
        self._active_workers[operation_id] = (
            worker,
            handle_result,
            handle_error,
            handle_finished,
        )
        self.thread_pool.start(worker)

    def continue_enrollment(self, speaker_id: str, display_name: str) -> None:
        speaker_id = speaker_id.strip()
        LOGGER.info("Enrollment requested (speaker_id=%s)", speaker_id or "<empty>")
        error = self.enrollment.validate_speaker_id(speaker_id)
        info_page = self.window.pages["enroll_info"]
        if not error and self.repository.exists(speaker_id):
            error = "Speaker ID already exists"
        if error:
            LOGGER.warning("Enrollment information rejected: %s", error)
            info_page.error.setText(error)
            return
        info_page.error.clear()
        self.pending_speaker_id = speaker_id
        self.pending_display_name = display_name.strip()
        auth_page = self.window.pages["authorization"]
        auth_page.speaker_label.setText(f"Speaker: {speaker_id}")
        auth_page.password.clear()
        auth_page.error.clear()
        self.window.show_page("authorization")
        LOGGER.debug("Navigation: enrollment authorization")

    def back_to_enrollment_info(self) -> None:
        auth_page = self.window.pages["authorization"]
        auth_page.password.clear()
        auth_page.error.clear()
        self.window.show_page("enroll_info")
        LOGGER.debug("Navigation: enrollment information")

    def verify_authorization(self, password: str) -> None:
        result = self.authorization.verify_password(password)
        auth_page = self.window.pages["authorization"]
        auth_page.password.clear()
        if not result.accepted:
            suffix = f" ({result.remaining_attempts} attempts remaining)" if result.remaining_attempts else ""
            auth_page.error.setText(result.message + suffix)
            LOGGER.warning(
                "Enrollment authorization rejected (remaining_attempts=%d)",
                result.remaining_attempts,
            )
            return
        LOGGER.info("Enrollment authorization accepted")
        self.accepted_clips.clear()
        self.current_clip = None
        self.enrollment_started = False
        self.cancel_requested = False
        recording = self.window.pages["enroll_recording"]
        recording.update_clip(1, self.config.enrollment_clip_count)
        self.window.show_page("enroll_recording")
        LOGGER.debug("Navigation: enrollment recording")

    def _record_and_validate(self, path: Path, duration: float):
        recording = self.audio.record_clip(path, duration)
        if not recording.success or recording.audio_path is None:
            return recording, None
        return recording, self.audio.validate_audio(recording.audio_path)

    def record_enrollment_clip(self) -> None:
        page = self.window.pages["enroll_recording"]
        if self.busy:
            LOGGER.info("User requested recording stop")
            self.audio.stop_recording()
            page.record_button.setEnabled(False)
            page.status.setText("Stopping recording, applying gain and denoising...")
            return
        if not self.enrollment_started and not self.authorization.authorized:
            self.window.show_error("Enrollment authorization expired. Enter the password again.")
            self.back_to_enrollment_info()
            return
        self.enrollment_started = True
        clip_number = len(self.accepted_clips) + 1
        LOGGER.info(
            "Starting enrollment clip %d of %d",
            clip_number,
            self.config.enrollment_clip_count,
        )
        speaker_dir = self.config.enrollment_audio_dir / self.pending_speaker_id
        path = speaker_dir / f"{self.pending_speaker_id}_clip_{clip_number:02d}.wav"
        page.status.setText(f"Recording clip {clip_number}...")
        page.record_button.setText("Stop Recording")
        page.record_button.setEnabled(True)
        page.next_button.setEnabled(False)

        def completed(payload: object) -> None:
            recording, validation = payload
            if self.cancel_requested:
                LOGGER.info("Recording completed after cancellation request; discarding clip")
                path.unlink(missing_ok=True)
                self._complete_enrollment_cancel()
                return
            if not recording.success or validation is None:
                LOGGER.warning("Enrollment clip recording failed: %s", recording.message)
                page.status.setText(recording.message)
                page.record_button.setEnabled(True)
                page.record_button.setText("Record Again")
                return
            if not validation.accepted:
                LOGGER.warning("Enrollment clip %d rejected: %s", clip_number, validation.message)
                path.unlink(missing_ok=True)
                page.status.setText(validation.message + ". Please record again.")
                page.record_button.setEnabled(True)
                page.record_button.setText("Record Again")
                return
            self.current_clip = path
            LOGGER.info("Enrollment clip %d accepted", clip_number)
            page.status.setText(
                f"Audio accepted: {validation.duration_seconds:.1f}s raw-duration clip."
            )
            page.record_button.setEnabled(True)
            page.record_button.setText("Record Again")
            page.next_button.setText(
                "Process Enrollment"
                if clip_number == self.config.enrollment_clip_count
                else "Next Clip"
            )
            page.next_button.setEnabled(True)

        def failed() -> None:
            if self.cancel_requested:
                path.unlink(missing_ok=True)
                self._complete_enrollment_cancel()
                return
            page.record_button.setEnabled(True)
            page.record_button.setText("Record Again")

        self._run(
            lambda: self._record_and_validate(path, self.config.enrollment_clip_duration_seconds),
            completed,
            failed,
        )

    def accept_current_clip(self) -> None:
        if self.busy or self.current_clip is None:
            return
        self.accepted_clips.append(self.current_clip)
        LOGGER.info(
            "Enrollment clip confirmed (%d/%d)",
            len(self.accepted_clips),
            self.config.enrollment_clip_count,
        )
        self.current_clip = None
        if len(self.accepted_clips) < self.config.enrollment_clip_count:
            self.window.pages["enroll_recording"].update_clip(
                len(self.accepted_clips) + 1, self.config.enrollment_clip_count
            )
            return

        self.window.show_page("processing")
        LOGGER.info("Processing enrollment profile (clips=%d)", len(self.accepted_clips))

        def completed(result: object) -> None:
            if not result.success:
                LOGGER.warning("Enrollment processing rejected: %s", result.message)
                self.window.show_error(result.message)
                self._prepare_final_clip_retry()
                return
            success_page = self.window.pages["enroll_success"]
            LOGGER.info("Enrollment completed (speaker_id=%s)", result.speaker_id)
            success_page.details.setText(
                f"Speaker ID: {result.speaker_id}\nAccepted clips: {result.accepted_clip_count}"
            )
            if not self.config.retain_enrollment_audio:
                self._delete_clips()
            self.window.show_page("enroll_success")

        def failed() -> None:
            self._prepare_final_clip_retry()

        self._run(
            lambda: self.enrollment.enroll(
                self.pending_speaker_id, self.pending_display_name, list(self.accepted_clips)
            ),
            completed,
            failed,
        )

    def _prepare_final_clip_retry(self) -> None:
        page = self.window.pages["enroll_recording"]
        if self.current_clip is None and self.accepted_clips:
            self.current_clip = self.accepted_clips.pop()
        page.status.setText("Processing failed. Retry processing or record the final clip again.")
        page.record_button.setText("Record Again")
        page.record_button.setEnabled(True)
        page.next_button.setText("Retry Processing")
        page.next_button.setEnabled(self.current_clip is not None)
        self.window.show_page("enroll_recording")

    def _delete_clips(self) -> None:
        for path in set(self.accepted_clips + ([self.current_clip] if self.current_clip else [])):
            path.unlink(missing_ok=True)
        if self.pending_speaker_id:
            directory = self.config.enrollment_audio_dir / self.pending_speaker_id
            try:
                directory.rmdir()
            except OSError:
                pass

    def cancel_enrollment(self) -> None:
        if self.busy:
            LOGGER.info("Enrollment cancellation requested during recording")
            self.cancel_requested = True
            self.audio.stop_recording()
            page = self.window.pages["enroll_recording"]
            page.status.setText("Cancelling enrollment...")
            page.record_button.setEnabled(False)
            page.next_button.setEnabled(False)
            page.cancel_button.setEnabled(False)
            return
        self._complete_enrollment_cancel()

    def _complete_enrollment_cancel(self) -> None:
        LOGGER.info("Enrollment cancelled; cleaning temporary state")
        self._delete_clips()
        self.authorization.reset()
        self.pending_speaker_id = ""
        self.pending_display_name = ""
        self.accepted_clips.clear()
        self.current_clip = None
        self.enrollment_started = False
        self.cancel_requested = False
        info = self.window.pages["enroll_info"]
        info.speaker_id.clear()
        info.display_name.clear()
        info.error.clear()
        self.window.show_page("home")

    def finish_enrollment(self) -> None:
        LOGGER.debug("Enrollment success acknowledged; returning home")
        self.authorization.reset()
        self.pending_speaker_id = ""
        self.pending_display_name = ""
        self.accepted_clips.clear()
        self.current_clip = None
        self.enrollment_started = False
        info = self.window.pages["enroll_info"]
        info.speaker_id.clear()
        info.display_name.clear()
        info.error.clear()
        self.window.show_page("home")

    def open_recognition(self) -> None:
        LOGGER.debug("Navigation: recognition")
        page = self.window.pages["recognition"]
        page.status.setText("Press Start Recording and speak naturally.")
        page.start_button.setText("Start Recording")
        page.start_button.setEnabled(True)
        self.window.show_page("recognition")

    def _recognition_job(self, path: Path) -> tuple[str, object]:
        try:
            recording, validation = self._record_and_validate(
                path, self.config.recognition_clip_duration_seconds
            )
            if not recording.success:
                return "error", recording.message
            if validation is None or not validation.accepted:
                return "error", validation.message if validation else "Recording failed"
            return "result", self.recognition.recognize(path)
        finally:
            path.unlink(missing_ok=True)

    def start_recognition(self) -> None:
        page = self.window.pages["recognition"]
        if self.busy:
            LOGGER.info("User requested recognition recording stop")
            self.audio.stop_recording()
            page.start_button.setEnabled(False)
            page.status.setText("Stopping recording and recognizing speaker...")
            return
        page.status.setText("Recording, validating, and comparing speakers...")
        LOGGER.info("Recognition recording started")
        page.start_button.setText("Stop Recording")
        page.start_button.setEnabled(True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.config.temporary_audio_dir / f"recognition_{timestamp}.wav"

        def completed(payload: tuple[str, object]) -> None:
            kind, value = payload
            if kind == "error":
                LOGGER.warning("Recognition could not run: %s", value)
                page.status.setText(str(value))
                page.start_button.setText("Start Recording")
                page.start_button.setEnabled(True)
                return
            result: RecognitionResult = value
            result_page = self.window.pages["recognition_result"]
            if result.accepted:
                LOGGER.info(
                    "Recognition accepted (speaker_id=%s, similarity=%.4f)",
                    result.speaker_id,
                    result.similarity,
                )
                result_page.heading.setText("Recognized Speaker")
                identity = result.speaker_id or ""
                if result.display_name:
                    identity += f"\n{result.display_name}"
                result_page.identity.setText(identity)
                result_page.details.setText(f"Similarity Score: {result.similarity:.3f}")
            else:
                LOGGER.info("Recognition rejected (similarity=%.4f)", result.similarity)
                result_page.heading.setText("Unknown Speaker")
                result_page.identity.clear()
                score = "" if result.similarity < -0.999 else f"\nSimilarity Score: {result.similarity:.3f}"
                result_page.details.setText(result.message + score)
            self.window.show_page("recognition_result")

        self._run(
            lambda: self._recognition_job(path),
            completed,
            lambda: (
                page.start_button.setText("Start Recording"),
                page.start_button.setEnabled(True),
            ),
        )
