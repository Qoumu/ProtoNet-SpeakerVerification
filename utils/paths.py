from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_default_librispeech_root() -> Path:
    env_value = os.environ.get("LIBRISPEECH_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    candidates = (
        PROJECT_ROOT / "data" / "speakerdataset" / "LibriSpeech" ,
        PROJECT_ROOT.parent / "data" / "speakerdataset" / "LibriSpeech" ,
        PROJECT_ROOT.parent / "Nemo_SR" / "data" / "speakerdataset" / "LibriSpeech" ,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[1].resolve()
