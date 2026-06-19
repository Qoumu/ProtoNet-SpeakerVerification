from __future__ import annotations

import getpass

from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService


def main() -> int:
    password = getpass.getpass("Enrollment password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    print(EnrollmentAuthorizationService.create_password_hash(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
