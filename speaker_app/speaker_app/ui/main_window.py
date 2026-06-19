from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QInputDialog, QLineEdit, QMainWindow, QMessageBox, QStackedWidget

from speaker_app.domain import SpeakerProfile
from speaker_app.ui.pages import (
    AuthorizationPage,
    EnrollInfoPage,
    EnrollmentRecordingPage,
    EnrollmentSuccessPage,
    HomePage,
    ProcessingPage,
    RecognitionPage,
    RecognitionResultPage,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Speaker Recognition")
        self.setMinimumSize(640, 400)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.pages = {
            "home": HomePage(),
            "enroll_info": EnrollInfoPage(),
            "authorization": AuthorizationPage(),
            "enroll_recording": EnrollmentRecordingPage(),
            "processing": ProcessingPage(),
            "enroll_success": EnrollmentSuccessPage(),
            "recognition": RecognitionPage(),
            "recognition_result": RecognitionResultPage(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)
        stylesheet = Path(__file__).with_name("styles.qss")
        self.setStyleSheet(stylesheet.read_text(encoding="utf-8"))
        self.show_page("home")

    def show_page(self, name: str) -> None:
        self.stack.setCurrentWidget(self.pages[name])

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Speaker Recognition", message)

    def show_info(self, message: str) -> None:
        QMessageBox.information(self, "Speaker Recognition", message)

    def request_enrollment_password(self, instruction: str) -> str | None:
        password, accepted = QInputDialog.getText(
            self,
            "Enrollment Authorization",
            instruction,
            QLineEdit.EchoMode.Password,
        )
        return password if accepted else None

    def choose_speaker(self, profiles: list[SpeakerProfile]) -> str | None:
        labels = [
            f"{profile.speaker_id} - {profile.display_name}"
            if profile.display_name
            else profile.speaker_id
            for profile in profiles
        ]
        selected, accepted = QInputDialog.getItem(
            self,
            "Remove Enrolled Speaker",
            "Select the speaker profile to remove",
            labels,
            0,
            False,
        )
        if not accepted:
            return None
        return profiles[labels.index(selected)].speaker_id

    def confirm_speaker_delete(self, speaker_id: str, display_name: str | None) -> bool:
        identity = f"{speaker_id} ({display_name})" if display_name else speaker_id
        answer = QMessageBox.question(
            self,
            "Remove Enrolled Speaker",
            f"Delete speaker profile {identity}?\n\n"
            "Saved enrollment WAV files will be preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes
