from speaker_app.services.enrollment_authorization_service import EnrollmentAuthorizationService


def test_authorization_accepts_password_and_expires(monkeypatch):
    password_hash = EnrollmentAuthorizationService.create_password_hash("correct")
    service = EnrollmentAuthorizationService(password_hash, timeout_seconds=10)
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])

    assert service.verify_password("correct").accepted
    assert service.authorized
    now[0] = 111.0
    assert not service.authorized


def test_authorization_limits_failed_attempts():
    password_hash = EnrollmentAuthorizationService.create_password_hash("correct")
    service = EnrollmentAuthorizationService(password_hash, max_attempts=2)

    assert service.verify_password("wrong").remaining_attempts == 1
    result = service.verify_password("still-wrong")
    assert not result.accepted
    assert result.remaining_attempts == 0
    assert not service.verify_password("correct").accepted


def test_missing_hash_fails_closed():
    result = EnrollmentAuthorizationService(None).verify_password("anything")
    assert not result.accepted
    assert "not configured" in result.message
