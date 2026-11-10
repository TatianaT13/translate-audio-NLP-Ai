"""Tests — confirmation de l'adresse email à l'inscription.

Avec REQUIRE_EMAIL_VERIFICATION, les crédits offerts ne sont accordés qu'après
le clic sur le lien reçu par email : un script ne peut plus créer des comptes
en boucle pour consommer le budget API.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from gateway import emails, main, models
from gateway.database import SessionLocal

client = TestClient(main.app)
PASSWORD = "Motdepasse2026!"


@pytest.fixture
def outbox(monkeypatch):
    """Emails interceptés : (type, destinataire, lien éventuel)."""
    sent = []
    monkeypatch.setattr(emails, "send_verification", lambda to, url: sent.append(("verification", to, url)))
    monkeypatch.setattr(emails, "send_welcome", lambda to: sent.append(("welcome", to, None)))
    return sent


@pytest.fixture
def verification_required(monkeypatch):
    monkeypatch.setattr(main, "REQUIRE_EMAIL_VERIFICATION", True)
    monkeypatch.setattr(main, "DEV_MODE", False)


def _register() -> tuple[str, dict]:
    email = f"{uuid.uuid4().hex}@example.com"
    r = client.post("/auth/register", json={"email": email, "password": PASSWORD})
    assert r.status_code == 201, r.text
    return email, r.json()


def _login(email: str) -> dict:
    token = client.post("/auth/login", json={"email": email, "password": PASSWORD}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _balance(headers: dict) -> int:
    return client.get("/billing/credits", headers=headers).json()["processing"]["balance"]


def _token(url: str) -> str:
    return url.split("token=")[1]


class TestVerificationOff:
    """Comportement par défaut, inchangé tant que l'envoi d'emails n'est pas prêt."""

    def test_trial_credits_are_granted_at_signup(self, outbox):
        email, body = _register()
        headers = _login(email)
        assert body["verification_required"] is False
        assert _balance(headers) == 10
        assert client.get("/auth/me", headers=headers).json()["email_verified"] is True
        assert [kind for kind, *_ in outbox] == ["welcome"]


class TestVerificationOn:
    def test_no_credits_until_the_address_is_confirmed(self, verification_required, outbox):
        email, body = _register()
        headers = _login(email)
        assert body["verification_required"] is True
        assert "verify_url" not in body                      # jamais le lien dans la réponse en prod
        assert _balance(headers) == 0
        assert client.get("/auth/me", headers=headers).json()["email_verified"] is False
        assert [(kind, to) for kind, to, _ in outbox] == [("verification", email)]

    def test_confirming_grants_the_trial_credits(self, verification_required, outbox):
        email, _ = _register()
        headers = _login(email)
        r = client.post("/auth/verify-email", json={"token": _token(outbox[0][2])})
        assert r.status_code == 200
        assert _balance(headers) == 10
        assert client.get("/auth/me", headers=headers).json()["email_verified"] is True
        assert outbox[-1] == ("welcome", email, None)

    def test_a_link_works_only_once(self, verification_required, outbox):
        email, _ = _register()
        token = _token(outbox[0][2])
        assert client.post("/auth/verify-email", json={"token": token}).status_code == 200
        assert client.post("/auth/verify-email", json={"token": token}).status_code == 400
        assert _balance(_login(email)) == 10                 # pas de second lot de crédits

    def test_unknown_link_is_refused(self, verification_required):
        assert client.post("/auth/verify-email", json={"token": "nimportequoi"}).status_code == 400

    def test_expired_link_is_refused(self, verification_required, outbox):
        email, _ = _register()
        db = SessionLocal()
        user = db.query(models.User).filter_by(email=email).one()
        db.query(models.EmailVerificationToken).filter_by(user_id=user.id).update(
            {"expires_at": datetime.now(timezone.utc) - timedelta(minutes=1)})
        db.commit()
        db.close()
        assert client.post("/auth/verify-email", json={"token": _token(outbox[0][2])}).status_code == 400
        assert _balance(_login(email)) == 0

    def test_resend_replaces_the_previous_link(self, verification_required, outbox):
        email, _ = _register()
        headers = _login(email)
        assert client.post("/auth/resend-verification", headers=headers).status_code == 200
        first, second = _token(outbox[0][2]), _token(outbox[1][2])
        assert client.post("/auth/verify-email", json={"token": first}).status_code == 400
        assert client.post("/auth/verify-email", json={"token": second}).status_code == 200
        assert _balance(headers) == 10

    def test_resend_does_nothing_once_confirmed(self, verification_required, outbox):
        email, _ = _register()
        headers = _login(email)
        client.post("/auth/verify-email", json={"token": _token(outbox[0][2])})
        sent_before = len(outbox)
        assert client.post("/auth/resend-verification", headers=headers).status_code == 200
        assert len(outbox) == sent_before

    def test_resend_requires_login(self, verification_required):
        assert client.post("/auth/resend-verification").status_code == 401

    def test_dev_mode_returns_the_link_for_local_testing(self, verification_required, outbox, monkeypatch):
        monkeypatch.setattr(main, "DEV_MODE", True)
        _, body = _register()
        assert body["verify_url"] == outbox[0][2]


def test_migration_marks_existing_accounts_as_verified(tmp_path, monkeypatch):
    """Base de production d'avant la fonctionnalité : les comptes existants ne
    doivent pas se retrouver « non confirmés »."""
    engine = create_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR, hashed_password VARCHAR, "
                          "is_active BOOLEAN, is_admin BOOLEAN, stripe_customer_id VARCHAR, "
                          "subscription_tier VARCHAR DEFAULT 'free' NOT NULL, subscription_status VARCHAR, "
                          "current_period_end TIMESTAMP)"))
        conn.execute(text("INSERT INTO users (email, hashed_password) VALUES ('ancien@example.com', 'x')"))
    monkeypatch.setattr(main, "engine", engine)

    main._run_lightweight_migrations()
    main._run_lightweight_migrations()      # idempotent

    assert "email_verified" in {c["name"] for c in inspect(engine).get_columns("users")}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT email_verified FROM users")).scalar() == 1
