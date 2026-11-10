"""Tests — résiliation de l'abonnement Stripe.

- à tout moment depuis l'application (effet à la fin de la période payée) ;
- immédiatement à la suppression du compte, sinon le client resterait prélevé.
Stripe est simulé : aucun appel réseau.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import stripe
from fastapi.testclient import TestClient

from gateway import auth, billing, emails, main, models
from gateway.database import SessionLocal

client = TestClient(main.app)
PERIOD_END = int(datetime(2026, 12, 12, tzinfo=timezone.utc).timestamp())


class FakeStripe:
    """Abonnements d'un client, avec les trois appels utilisés par l'application."""

    def __init__(self, monkeypatch, subs: list[dict], fail: bool = False):
        self.subs, self.fail, self.calls = subs, fail, []
        monkeypatch.setattr(stripe, "api_key", "sk_test_simule")
        monkeypatch.setattr(stripe.Subscription, "list", self.list)
        monkeypatch.setattr(stripe.Subscription, "modify", self.modify)
        monkeypatch.setattr(stripe.Subscription, "cancel", self.cancel)

    def _check(self):
        if self.fail:
            raise stripe.APIConnectionError("stripe injoignable")

    def list(self, **kwargs):
        self._check()
        self.calls.append(("list", kwargs["customer"]))
        return SimpleNamespace(data=self.subs)

    def modify(self, sub_id, **kwargs):
        self._check()
        self.calls.append(("modify", sub_id, kwargs))
        sub = next(s for s in self.subs if s["id"] == sub_id)
        sub.update(kwargs)
        return sub

    def cancel(self, sub_id):
        self._check()
        self.calls.append(("cancel", sub_id))
        next(s for s in self.subs if s["id"] == sub_id)["status"] = "canceled"


def _active_sub(sub_id: str = "sub_1") -> dict:
    return {"id": sub_id, "status": "active", "cancel_at_period_end": False, "current_period_end": PERIOD_END}


def _subscriber(is_admin: bool = False) -> tuple[int, dict]:
    db = SessionLocal()
    user = models.User(
        email=f"{uuid.uuid4().hex}@example.com", hashed_password="x", is_admin=is_admin,
        stripe_customer_id=f"cus_{uuid.uuid4().hex[:12]}", subscription_tier="pro", subscription_status="active",
    )
    db.add(user)
    db.commit()
    uid, email = user.id, user.email
    db.close()
    return uid, {"Authorization": f"Bearer {auth.create_access_token(uid, email, is_admin)}"}


@pytest.fixture
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(emails, "send_cancellation_confirmation", lambda to, end: sent.append((to, end)))
    return sent


class TestCancelAnytime:
    def test_cancel_keeps_access_until_the_end_of_the_paid_period(self, monkeypatch, outbox):
        fake = FakeStripe(monkeypatch, [_active_sub()])
        _, headers = _subscriber()

        r = client.post("/billing/cancel", headers=headers)

        assert r.status_code == 200
        body = r.json()
        assert body["cancel_at_period_end"] is True
        assert body["tier"] == "pro"                             # toujours Pro jusqu'à l'échéance
        assert body["current_period_end"].startswith("2026-12-12")
        assert ("modify", "sub_1", {"cancel_at_period_end": True}) in fake.calls
        assert not any(c[0] == "cancel" for c in fake.calls)     # pas de coupure immédiate

    def test_cancellation_is_confirmed_by_email_once(self, monkeypatch, outbox):
        FakeStripe(monkeypatch, [_active_sub()])
        _, headers = _subscriber()
        client.post("/billing/cancel", headers=headers)
        client.post("/billing/cancel", headers=headers)          # double clic
        assert len(outbox) == 1 and outbox[0][1] == "12/12/2026"

    def test_subscription_state_shows_the_cancellation(self, monkeypatch, outbox):
        FakeStripe(monkeypatch, [_active_sub()])
        _, headers = _subscriber()
        assert client.get("/billing/subscription", headers=headers).json()["cancel_at_period_end"] is False
        client.post("/billing/cancel", headers=headers)
        assert client.get("/billing/subscription", headers=headers).json()["cancel_at_period_end"] is True

    def test_resume_undoes_the_cancellation(self, monkeypatch, outbox):
        fake = FakeStripe(monkeypatch, [_active_sub()])
        _, headers = _subscriber()
        client.post("/billing/cancel", headers=headers)
        r = client.post("/billing/resume", headers=headers)
        assert r.json()["cancel_at_period_end"] is False
        assert fake.calls[-1] == ("modify", "sub_1", {"cancel_at_period_end": False})

    def test_no_subscription_to_cancel(self, monkeypatch, outbox):
        FakeStripe(monkeypatch, [])
        _, headers = _subscriber()
        assert client.post("/billing/cancel", headers=headers).status_code == 400
        assert outbox == []

    def test_stripe_down_is_reported_not_silently_ignored(self, monkeypatch, outbox):
        FakeStripe(monkeypatch, [_active_sub()], fail=True)
        _, headers = _subscriber()
        assert client.post("/billing/cancel", headers=headers).status_code == 502
        assert client.get("/billing/subscription", headers=headers).json()["cancel_at_period_end"] is False
        assert outbox == []

    def test_login_is_required(self):
        assert client.post("/billing/cancel").status_code == 401
        assert client.post("/billing/resume").status_code == 401


class TestAccountDeletion:
    def test_deleting_the_account_cancels_the_subscription_immediately(self, monkeypatch):
        fake = FakeStripe(monkeypatch, [_active_sub("sub_a"), {**_active_sub("sub_old"), "status": "canceled"}])
        uid, headers = _subscriber()

        assert client.delete("/auth/account", headers=headers).status_code == 200

        assert ("cancel", "sub_a") in fake.calls
        assert ("cancel", "sub_old") not in fake.calls           # déjà terminé
        db = SessionLocal()
        assert db.query(models.User).filter_by(id=uid).count() == 0
        db.close()

    def test_account_is_kept_if_the_subscription_cannot_be_cancelled(self, monkeypatch):
        """Sinon : abonnement actif, prélèvements, et plus de compte pour le résilier."""
        FakeStripe(monkeypatch, [_active_sub()], fail=True)
        uid, headers = _subscriber()

        assert client.delete("/auth/account", headers=headers).status_code == 502

        db = SessionLocal()
        assert db.query(models.User).filter_by(id=uid).count() == 1
        db.close()

    def test_admin_deleting_an_account_cancels_its_subscription(self, monkeypatch):
        fake = FakeStripe(monkeypatch, [_active_sub("sub_b")])
        uid, _ = _subscriber()
        _, admin = _subscriber(is_admin=True)
        assert client.delete(f"/admin/users/{uid}", headers=admin).status_code == 200
        assert ("cancel", "sub_b") in fake.calls

    def test_account_without_stripe_customer_is_deleted_without_calling_stripe(self, monkeypatch):
        fake = FakeStripe(monkeypatch, [_active_sub()])
        db = SessionLocal()
        user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x")
        db.add(user)
        db.commit()
        headers = {"Authorization": f"Bearer {auth.create_access_token(user.id, user.email, False)}"}
        db.close()
        assert client.delete("/auth/account", headers=headers).status_code == 200
        assert fake.calls == []


class TestSubscriptionSync:
    def test_period_end_is_read_from_items_on_recent_stripe_api_versions(self):
        sub = {"status": "active", "items": {"data": [{"current_period_end": PERIOD_END}]}}
        assert billing._period_end_ts(sub) == PERIOD_END

    def test_period_end_on_older_api_versions(self):
        assert billing._period_end_ts({"current_period_end": PERIOD_END}) == PERIOD_END

    def test_migration_adds_the_column_to_an_existing_database(self, tmp_path, monkeypatch):
        from sqlalchemy import create_engine, inspect, text
        engine = create_engine(f"sqlite:///{tmp_path}/old.db")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR, hashed_password VARCHAR, "
                              "is_active BOOLEAN, is_admin BOOLEAN)"))
            conn.execute(text("INSERT INTO users (email, hashed_password) VALUES ('ancien@example.com', 'x')"))
        monkeypatch.setattr(main, "engine", engine)
        main._run_lightweight_migrations()
        main._run_lightweight_migrations()
        assert "cancel_at_period_end" in {c["name"] for c in inspect(engine).get_columns("users")}
        with engine.connect() as conn:
            assert conn.execute(text("SELECT cancel_at_period_end FROM users")).scalar() == 0
