import numpy as np

from speaker_app.app_controller import AppController
from speaker_app.domain import SpeakerProfile
from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService
from speaker_app.services.speaker_repository import SpeakerRepository


class LabelStub:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class HomeStub:
    def __init__(self):
        self.database_status = LabelStub()


class WindowStub:
    def __init__(self):
        self.pages = {"home": HomeStub()}
        self.information = []
        self.errors = []

    def choose_speaker(self, profiles):
        assert [profile.speaker_id for profile in profiles] == ["alice", "bob"]
        return "bob"

    def request_enrollment_password(self, instruction):
        assert "bob" in instruction
        return "secret"

    def confirm_speaker_delete(self, speaker_id, display_name):
        return speaker_id == "bob" and display_name == "Bob"

    def show_info(self, message):
        self.information.append(message)

    def show_error(self, message):
        self.errors.append(message)


def test_remove_speaker_deletes_only_selected_profile(tmp_path):
    repository = SpeakerRepository(tmp_path / "speakers.db")
    repository.initialize()
    repository.save(SpeakerProfile("alice", "Alice", np.array([1.0, 0.0]), "v1", 6))
    repository.save(SpeakerProfile("bob", "Bob", np.array([0.0, 1.0]), "v1", 6))
    password_hash = EnrollmentAuthorizationService.create_password_hash("secret")

    controller = AppController.__new__(AppController)
    controller.busy = False
    controller.repository = repository
    controller.authorization = EnrollmentAuthorizationService(password_hash)
    controller.window = WindowStub()
    controller._run = lambda function, callback, on_error=None: callback(function())

    controller.remove_speaker()

    assert repository.exists("alice")
    assert not repository.exists("bob")
    assert repository.count() == 1
    assert not controller.window.errors
    assert "preserved" in controller.window.information[0]
