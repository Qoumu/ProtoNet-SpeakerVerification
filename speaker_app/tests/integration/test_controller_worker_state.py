import time

import pytest


pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QThreadPool

from speaker_app.app_controller import AppController


class WindowStub:
    def show_error(self, message):
        raise AssertionError(message)


def test_backend_result_unlocks_controller_before_callback():
    app = QCoreApplication.instance() or QCoreApplication([])
    controller = AppController.__new__(AppController)
    controller.window = WindowStub()
    controller.busy = False
    controller._operation_id = 0
    controller._active_workers = {}
    controller.thread_pool = QThreadPool()
    callback_states = []

    controller._run(lambda: "accepted", lambda result: callback_states.append((result, controller.busy)))

    deadline = time.monotonic() + 2.0
    while not callback_states and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.001)
    controller.thread_pool.waitForDone(1000)
    app.processEvents()

    assert callback_states == [("accepted", False)]
    assert not controller.busy
