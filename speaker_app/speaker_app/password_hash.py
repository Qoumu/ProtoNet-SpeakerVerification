from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from speaker_app.config import default_password_hash_path
from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up the enrollment password hash")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Environment file to read before choosing the default output path",
    )
    parser.add_argument(
        "--profile",
        choices=("development", "raspberry-pi"),
        default=None,
        help="Profile used to choose the default secret file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Secret file to write. Defaults to APP_DATA_DIR/enrollment_password_hash "
            "and ignores ENROLLMENT_PASSWORD_HASH_FILE."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing password hash file",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the hash instead of saving it",
    )
    return parser.parse_args(argv)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _write_secret_file(path: Path, password_hash: str, *, force: bool) -> None:
    path = path.expanduser()
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists. Use --force to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as handle:
        handle.write(password_hash)
        handle.write("\n")
    os.chmod(path, 0o600)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_env_file(args.env_file.expanduser())
    password = getpass.getpass("Enrollment password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")
    password_hash = EnrollmentAuthorizationService.create_password_hash(password)
    if args.print_only:
        print(password_hash)
        return 0

    output = args.output or default_password_hash_path(
        args.profile, honor_secret_file_env=False
    )
    _write_secret_file(output, password_hash, force=args.force)
    print(f"Saved enrollment password hash to {output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
