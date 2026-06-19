from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time

from speaker_app.domain import AuthorizationResult


LOGGER = logging.getLogger(__name__)


class EnrollmentAuthorizationService:
    """Verify an application-wide enrollment password using stdlib scrypt."""

    PREFIX = "scrypt"

    def __init__(
        self,
        password_hash: str | None,
        max_attempts: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        self._password_hash = password_hash
        self._max_attempts = max(1, max_attempts)
        self._timeout_seconds = max(1, timeout_seconds)
        self._failed_attempts = 0
        self._authorized_at: float | None = None
        LOGGER.debug(
            "Authorization service initialized (configured=%s, max_attempts=%d, timeout=%ds)",
            bool(password_hash),
            self._max_attempts,
            self._timeout_seconds,
        )

    @classmethod
    def create_password_hash(
        cls, password: str, *, n: int = 2**14, r: int = 8, p: int = 1
    ) -> str:
        if not password:
            raise ValueError("Password cannot be empty")
        salt = secrets.token_bytes(16)
        derived = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p)
        return "$".join(
            (
                cls.PREFIX,
                str(n),
                str(r),
                str(p),
                base64.urlsafe_b64encode(salt).decode(),
                base64.urlsafe_b64encode(derived).decode(),
            )
        )

    @property
    def configured(self) -> bool:
        return bool(self._password_hash)

    @property
    def authorized(self) -> bool:
        return self._authorized_at is not None and (
            time.monotonic() - self._authorized_at <= self._timeout_seconds
        )

    def reset(self) -> None:
        self._failed_attempts = 0
        self._authorized_at = None
        LOGGER.debug("Authorization session reset")

    def verify_password(self, password: str) -> AuthorizationResult:
        remaining = max(0, self._max_attempts - self._failed_attempts)
        if not self._password_hash:
            LOGGER.error("Authorization attempted without a configured password hash")
            return AuthorizationResult(False, remaining, "Enrollment password is not configured")
        if self._failed_attempts >= self._max_attempts:
            LOGGER.warning("Authorization blocked after maximum failed attempts")
            return AuthorizationResult(False, 0, "Maximum authorization attempts reached")
        if not password:
            LOGGER.warning("Authorization attempted with an empty password")
            return AuthorizationResult(False, remaining, "Enter the enrollment password")

        try:
            prefix, n, r, p, salt_text, expected_text = self._password_hash.split("$", 5)
            if prefix != self.PREFIX:
                raise ValueError("unsupported hash")
            salt = base64.urlsafe_b64decode(salt_text)
            expected = base64.urlsafe_b64decode(expected_text)
            actual = hashlib.scrypt(
                password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
            )
        except (ValueError, TypeError):
            LOGGER.exception("Enrollment password hash is invalid")
            return AuthorizationResult(False, remaining, "Enrollment password configuration is invalid")

        if hmac.compare_digest(actual, expected):
            self._failed_attempts = 0
            self._authorized_at = time.monotonic()
            LOGGER.info("Enrollment authorization succeeded")
            return AuthorizationResult(True, self._max_attempts, "Enrollment authorized")

        self._failed_attempts += 1
        remaining = max(0, self._max_attempts - self._failed_attempts)
        message = "Incorrect enrollment password"
        if remaining == 0:
            message = "Maximum authorization attempts reached"
        LOGGER.warning("Enrollment authorization failed (remaining_attempts=%d)", remaining)
        return AuthorizationResult(False, remaining, message)
