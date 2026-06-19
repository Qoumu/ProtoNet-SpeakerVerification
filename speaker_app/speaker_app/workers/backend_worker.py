from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


LOGGER = logging.getLogger(__name__)


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class BackendWorker(QRunnable):
    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        operation = getattr(self.function, "__name__", type(self.function).__name__)
        started = time.perf_counter()
        LOGGER.debug("Background operation started: %s", operation)
        try:
            self.signals.result.emit(self.function(*self.args, **self.kwargs))
        except Exception:
            LOGGER.exception("Background operation failed")
            self.signals.error.emit("The operation failed. Check the application log for details.")
        finally:
            LOGGER.debug(
                "Background operation finished: %s (%.3fs)",
                operation,
                time.perf_counter() - started,
            )
            self.signals.finished.emit()
