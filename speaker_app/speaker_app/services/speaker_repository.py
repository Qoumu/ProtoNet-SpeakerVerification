from __future__ import annotations

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from speaker_app.domain import SpeakerProfile


LOGGER = logging.getLogger(__name__)


class SpeakerRepository:
    """Transactional SQLite storage for speaker profiles."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        LOGGER.debug("Initializing SQLite speaker repository: %s", self.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS speakers (
                    speaker_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    embedding BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    number_of_samples INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
        LOGGER.info("SQLite speaker repository ready")

    def exists(self, speaker_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM speakers WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        return row is not None

    def save(self, profile: SpeakerProfile, *, overwrite: bool = False) -> None:
        embedding = np.asarray(profile.embedding, dtype=np.float32).reshape(-1)
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding must contain finite values")
        now = datetime.now(timezone.utc).isoformat()
        LOGGER.info(
            "Saving speaker profile (speaker_id=%s, dimension=%d, samples=%d, overwrite=%s)",
            profile.speaker_id,
            embedding.size,
            profile.number_of_samples,
            overwrite,
        )
        with self._connect() as connection:
            if overwrite:
                connection.execute(
                    """
                    INSERT INTO speakers (
                        speaker_id, display_name, embedding, embedding_dimension,
                        model_version, number_of_samples, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(speaker_id) DO UPDATE SET
                        display_name=excluded.display_name,
                        embedding=excluded.embedding,
                        embedding_dimension=excluded.embedding_dimension,
                        model_version=excluded.model_version,
                        number_of_samples=excluded.number_of_samples,
                        updated_at=excluded.updated_at
                    """,
                    (
                        profile.speaker_id,
                        profile.display_name,
                        embedding.tobytes(),
                        int(embedding.size),
                        profile.model_version,
                        profile.number_of_samples,
                        profile.created_at or now,
                        now,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO speakers VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.speaker_id,
                        profile.display_name,
                        embedding.tobytes(),
                        int(embedding.size),
                        profile.model_version,
                        profile.number_of_samples,
                        profile.created_at or now,
                        now,
                    ),
                )
        LOGGER.info("Speaker profile saved (speaker_id=%s)", profile.speaker_id)

    @staticmethod
    def _profile_from_row(row: sqlite3.Row) -> SpeakerProfile:
        embedding = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        if embedding.size != row["embedding_dimension"]:
            raise ValueError(f"Corrupt embedding for speaker {row['speaker_id']}")
        return SpeakerProfile(
            speaker_id=row["speaker_id"],
            display_name=row["display_name"],
            embedding=embedding,
            model_version=row["model_version"],
            number_of_samples=row["number_of_samples"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, speaker_id: str) -> SpeakerProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM speakers WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        return self._profile_from_row(row) if row else None

    def get_all(self) -> list[SpeakerProfile]:
        """Return every enrolled speaker, including profiles from older models."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM speakers ORDER BY speaker_id"
            ).fetchall()
        LOGGER.debug("Loaded all speaker profiles (count=%d)", len(rows))
        return [self._profile_from_row(row) for row in rows]

    def get_all_compatible(
        self, model_version: str, embedding_dimension: int
    ) -> list[SpeakerProfile]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM speakers
                WHERE model_version = ? AND embedding_dimension = ?
                ORDER BY speaker_id
                """,
                (model_version, embedding_dimension),
            ).fetchall()
        LOGGER.debug(
            "Loaded compatible profiles (model_version=%s, dimension=%d, count=%d)",
            model_version,
            embedding_dimension,
            len(rows),
        )
        return [self._profile_from_row(row) for row in rows]

    def delete(self, speaker_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,)
            )
        deleted = cursor.rowcount > 0
        LOGGER.info("Speaker profile delete (speaker_id=%s, deleted=%s)", speaker_id, deleted)
        return deleted

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM speakers").fetchone()[0])
