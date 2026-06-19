"""Source-tree shim for running the nested application package from the repo root."""

from pathlib import Path


_PACKAGE_DIRECTORY = Path(__file__).with_name("speaker_app")
if str(_PACKAGE_DIRECTORY) not in __path__:
    __path__.append(str(_PACKAGE_DIRECTORY))
