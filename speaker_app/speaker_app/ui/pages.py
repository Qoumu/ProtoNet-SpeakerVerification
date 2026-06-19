from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("pageTitle")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def action_button(text: str, *, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setMinimumHeight(64)
    if primary:
        button.setObjectName("primaryButton")
    return button


class HomePage(QWidget):
    enroll_requested = Signal()
    recognize_requested = Signal()
    remove_speaker_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title("SPEAKER RECOGNITION SYSTEM"))
        layout.addSpacing(24)
        self.enroll_button = action_button("ENROLL SPEAKER", primary=True)
        self.recognize_button = action_button("RECOGNIZE SPEAKER", primary=True)
        self.remove_speaker_button = action_button("REMOVE ENROLLED SPEAKER")
        self.remove_speaker_button.setObjectName("dangerButton")
        self.enroll_button.clicked.connect(self.enroll_requested.emit)
        self.recognize_button.clicked.connect(self.recognize_requested.emit)
        self.remove_speaker_button.clicked.connect(self.remove_speaker_requested.emit)
        layout.addWidget(self.enroll_button)
        layout.addWidget(self.recognize_button)
        layout.addWidget(self.remove_speaker_button)
        layout.addSpacing(20)
        self.microphone_status = QLabel()
        self.model_status = QLabel()
        self.database_status = QLabel()
        self.authorization_status = QLabel()
        for label in (
            self.microphone_status,
            self.model_status,
            self.database_status,
            self.authorization_status,
        ):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
        layout.addStretch()

    def set_system_status(
        self,
        *,
        microphone: tuple[bool, str],
        model: tuple[bool, str],
        database: tuple[bool, str],
        authorization: tuple[bool, str],
    ) -> None:
        def status_text(name: str, value: tuple[bool, str]) -> str:
            return f"{name}: {'Ready' if value[0] else 'Unavailable'} - {value[1]}"

        self.microphone_status.setText(status_text("Microphone", microphone))
        self.model_status.setText(status_text("Model", model))
        self.database_status.setText(status_text("Database", database))
        self.authorization_status.setText(status_text("Enrollment authorization", authorization))
        self.enroll_button.setEnabled(microphone[0] and model[0] and database[0] and authorization[0])
        self.recognize_button.setEnabled(microphone[0] and model[0] and database[0])
        self.remove_speaker_button.setEnabled(database[0] and authorization[0])


class EnrollInfoPage(QWidget):
    continue_requested = Signal(str, str)
    back_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(title("Enroll Speaker"))
        form = QFormLayout()
        self.speaker_id = QLineEdit()
        self.speaker_id.setPlaceholderText("Example: USER_001")
        self.display_name = QLineEdit()
        self.display_name.setPlaceholderText("Optional")
        form.addRow("Speaker ID", self.speaker_id)
        form.addRow("Display name", self.display_name)
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setObjectName("errorLabel")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        back = action_button("Back")
        proceed = action_button("Continue", primary=True)
        back.clicked.connect(self.back_requested.emit)
        proceed.clicked.connect(
            lambda: self.continue_requested.emit(self.speaker_id.text(), self.display_name.text())
        )
        buttons.addWidget(back)
        buttons.addWidget(proceed)
        layout.addStretch()
        layout.addLayout(buttons)


class AuthorizationPage(QWidget):
    confirm_requested = Signal(str)
    back_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(title("Enrollment Authorization"))
        self.speaker_label = QLabel()
        self.speaker_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.speaker_label)
        layout.addWidget(QLabel("Enter the enrollment password to continue"))
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.returnPressed.connect(lambda: self.confirm_requested.emit(self.password.text()))
        layout.addWidget(self.password)
        show_password = QCheckBox("Show password")
        show_password.toggled.connect(
            lambda checked: self.password.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        layout.addWidget(show_password)
        self.error = QLabel()
        self.error.setObjectName("errorLabel")
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        back = action_button("Back")
        confirm = action_button("Confirm", primary=True)
        back.clicked.connect(self.back_requested.emit)
        confirm.clicked.connect(lambda: self.confirm_requested.emit(self.password.text()))
        buttons.addWidget(back)
        buttons.addWidget(confirm)
        layout.addStretch()
        layout.addLayout(buttons)


class EnrollmentRecordingPage(QWidget):
    record_requested = Signal()
    next_requested = Signal()
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(title("Record Enrollment Clips"))
        self.clip_label = QLabel()
        self.clip_label.setObjectName("largeStatus")
        self.clip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.clip_label)
        self.status = QLabel("Press Record Clip and speak naturally.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.record_button = action_button("Record Clip", primary=True)
        self.next_button = action_button("Next Clip")
        self.cancel_button = action_button("Cancel Enrollment")
        self.record_button.clicked.connect(self.record_requested.emit)
        self.next_button.clicked.connect(self.next_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        layout.addStretch()
        layout.addWidget(self.record_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.cancel_button)
        self.next_button.setEnabled(False)

    def update_clip(self, index: int, count: int) -> None:
        self.clip_label.setText(f"Clip {index} of {count}")
        self.progress.setRange(0, count)
        self.progress.setValue(index - 1)
        self.status.setText("Press Record Clip and speak naturally.")
        self.record_button.setText("Record Clip")
        self.record_button.setEnabled(True)
        self.next_button.setEnabled(False)
        self.cancel_button.setEnabled(True)


class ProcessingPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title("Processing Enrollment"))
        self.status = QLabel("Extracting embeddings and saving speaker profile...")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        progress = QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(progress)
        layout.addStretch()


class EnrollmentSuccessPage(QWidget):
    home_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(title("Speaker Enrolled Successfully"))
        self.details = QLabel()
        self.details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.details)
        home = action_button("Return Home", primary=True)
        home.clicked.connect(self.home_requested.emit)
        layout.addWidget(home)
        layout.addStretch()


class RecognitionPage(QWidget):
    start_requested = Signal()
    back_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(title("Recognize Speaker"))
        self.status = QLabel("Press Start Recording and speak naturally.")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)
        layout.addStretch()
        self.start_button = action_button("Start Recording", primary=True)
        back = action_button("Back")
        self.start_button.clicked.connect(self.start_requested.emit)
        back.clicked.connect(self.back_requested.emit)
        layout.addWidget(self.start_button)
        layout.addWidget(back)


class RecognitionResultPage(QWidget):
    again_requested = Signal()
    home_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch()
        self.heading = title("")
        self.identity = QLabel()
        self.identity.setObjectName("resultIdentity")
        self.identity.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.heading)
        layout.addWidget(self.identity)
        layout.addWidget(self.details)
        again = action_button("Recognize Again", primary=True)
        home = action_button("Return Home")
        again.clicked.connect(self.again_requested.emit)
        home.clicked.connect(self.home_requested.emit)
        layout.addWidget(again)
        layout.addWidget(home)
        layout.addStretch()
