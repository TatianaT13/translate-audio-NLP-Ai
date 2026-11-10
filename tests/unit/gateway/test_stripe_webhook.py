"""Tests — événements Stripe : allocation des crédits, remboursement, avis de renouvellement.

Stripe est simulé (signature du webhook court-circuitée, aucun appel réseau).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import stripe
from fastapi.testclient import TestClient

from gateway import billing, credits, emails, main, models
from gateway.database import SessionLocal

client = TestClient(main.app)
NOW = datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def fake_stripe(monkeypatch):
    """Webhook sans signature + abonnements Stripe simulés."""
    state = SimpleNamespace(subs=[], cancelled=[])
    monkeypatch.setattr(stripe, "api_key", "sk_test_simule")
    monkeypatch.setattr(stripe.Webhook, "construct_event", lambda payload, sig, secret: json.loads(payload))
    monkeypatch.setattr(stripe.Subscription, "list", lambda **k: SimpleNamespace(data=state.subs))
    monkeypatch.setattr(stripe.Subscription, "cancel", lambda sub_id: state.cancelled.append(sub_id))
    monkeypatch.setattr(emails, "send_payment_confirmation", lambda *a: None)
    return state


@pytest.fixture
def customer() -> tuple[int, str]:
    """Compte gratuit avec ses 10 crédits d'essai et un client Stripe."""
    db = SessionLocal()
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x",
                       stripe_customer_id=f"cus_{uuid.uuid4().hex[:12]}")
    db.add(user)
    db.commit()
    credits.grant_trial(db, user.id)
    uid, cus = user.id, user.stripe_customer_id
    db.close()
    return uid, cus


def _event(kind: str, obj: dict) -> None:
    r = client.post("/billing/webhook", content=json.dumps({"type": kind, "data": {"object": obj}}),
                    headers={"stripe-signature": "simulee"})
    assert r.status_code == 200, r.text


def _sub(cus: str, *, status="active", end=None, interval="month", cancel=False) -> dict:
    end = end or NOW + timedelta(days=30)
    return {"id": "sub_1", "customer": cus, "status": status, "cancel_at_period_end": cancel,
            "current_period_end": int(end.timestamp()),
            "items": {"data": [{"price": {"recurring": {"interval": interval}}}]}}


def _state(uid: int) -> SimpleNamespace:
    db = SessionLocal()
    user = db.query(models.User).filter_by(id=uid).one()
    row = credits._get_or_create(db, uid)
    out = SimpleNamespace(balance=row.processing_balance, resets_at=credits._aware(row.period_resets_at),
                          tier=user.subscription_tier, interval=user.subscription_interval,
                          cancel=user.cancel_at_period_end)
    db.close()
    return out


class TestCreditAllocation:
    def test_subscribing_credits_the_month_once(self, customer):
        """Stripe envoie plusieurs événements à la souscription."""
        uid, cus = customer
        _event("customer.subscription.created", _sub(cus, status="incomplete"))
        assert _state(uid).balance == 10
        _event("customer.subscription.updated", _sub(cus))
        _event("customer.subscription.updated", _sub(cus))        # ex. carte mise à jour
        assert _state(uid).balance == 210
        assert _state(uid).tier == "pro"

    def test_cancelling_and_resuming_gives_no_credits(self, customer):
        """Sinon : résilier / reprendre en boucle créditerait 200 crédits à chaque clic."""
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus))
        for cancel in (True, False, True, False):
            _event("customer.subscription.updated", _sub(cus, cancel=cancel))
        assert _state(uid).balance == 210
        assert _state(uid).cancel is False

    def test_monthly_renewal_credits_the_new_month(self, customer):
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus, end=NOW + timedelta(hours=1)))   # fin de mois
        _event("customer.subscription.updated", _sub(cus, end=NOW + timedelta(days=30)))   # renouvellement
        assert _state(uid).balance == 410

    def test_yearly_plan_is_credited_every_month_not_once_a_year(self, customer):
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus, end=NOW + timedelta(days=365), interval="year"))
        state = _state(uid)
        assert state.balance == 210 and state.interval == "year"
        assert timedelta(days=29) < state.resets_at - NOW < timedelta(days=31)   # pas dans un an

        db = SessionLocal()                       # un mois plus tard
        credits._get_or_create(db, uid).period_resets_at = NOW - timedelta(minutes=1)
        db.commit()
        credits.refill_if_due(db, uid, "pro")
        db.close()
        assert _state(uid).balance == 410

    def test_subscription_end_keeps_the_remaining_credits(self, customer):
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus))
        _event("customer.subscription.deleted", _sub(cus, status="canceled"))
        state = _state(uid)
        assert (state.tier, state.balance, state.resets_at) == ("free", 210, None)


class TestRefund:
    def test_full_refund_cancels_the_subscription_and_takes_its_credits_back(self, customer, fake_stripe):
        uid, cus = customer
        fake_stripe.subs = [{"id": "sub_1", "status": "active"}]
        _event("customer.subscription.updated", _sub(cus))

        _event("charge.refunded", {"customer": cus, "refunded": True, "amount_refunded": 990})

        state = _state(uid)
        assert fake_stripe.cancelled == ["sub_1"]
        assert (state.tier, state.balance) == ("free", 10)          # l'essai reste, l'abonnement repart

    def test_replayed_refund_event_changes_nothing_more(self, customer, fake_stripe):
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus))
        refund = {"customer": cus, "refunded": True}
        _event("charge.refunded", refund)
        _event("charge.refunded", refund)
        assert _state(uid).balance == 10

    def test_partial_refund_is_ignored(self, customer, fake_stripe):
        """Geste commercial : l'abonnement continue."""
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus))
        _event("charge.refunded", {"customer": cus, "refunded": False, "amount_refunded": 300})
        assert (_state(uid).tier, _state(uid).balance) == ("pro", 210)
        assert fake_stripe.cancelled == []

    def test_refund_never_creates_a_debt(self, customer):
        uid, cus = customer
        _event("customer.subscription.updated", _sub(cus))
        db = SessionLocal()
        credits.consume(db, uid, "processing", 150, "pipeline_process:4500s:audio_translated")
        db.close()
        _event("charge.refunded", {"customer": cus, "refunded": True})
        assert _state(uid).balance == 0


def _yearly(end: datetime, **fields) -> int:
    db = SessionLocal()
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x", subscription_tier="pro",
                       subscription_interval="year", current_period_end=end, **fields)
    db.add(user)
    db.commit()
    uid = user.id
    db.close()
    return uid


class TestRenewalNotice:
    @pytest.fixture
    def outbox(self, monkeypatch):
        sent = []
        monkeypatch.setattr(emails, "send_renewal_notice", lambda to, date: sent.append(date))
        return sent

    def _run(self) -> int:
        with SessionLocal() as db:
            return billing.send_due_renewal_notices(db)

    def test_yearly_subscriber_is_warned_once_before_renewal(self, outbox):
        end = NOW + timedelta(days=60)
        _yearly(end)
        before = len(outbox)
        self._run()
        assert outbox[before:] == [end.strftime("%d/%m/%Y")]
        self._run()                                # le lendemain : pas de second avis
        assert len(outbox) == before + 1

    def test_no_notice_far_from_the_renewal(self, outbox):
        uid = _yearly(NOW + timedelta(days=200))
        self._run()
        with SessionLocal() as db:
            assert db.query(models.User).filter_by(id=uid).one().renewal_notice_for is None

    def test_next_year_is_announced_again(self, outbox):
        uid = _yearly(NOW + timedelta(days=60))
        self._run()
        with SessionLocal() as db:               # un an plus tard : nouvelle échéance
            db.query(models.User).filter_by(id=uid).update({"current_period_end": NOW + timedelta(days=50)})
            db.commit()
        before = len(outbox)
        self._run()
        assert len(outbox) == before + 1

    @pytest.mark.parametrize("fields", [
        {"subscription_interval": "month"},      # sans engagement, résiliable chaque mois
        {"cancel_at_period_end": True},          # déjà résilié : rien ne sera reconduit
        {"subscription_tier": "free"},
    ])
    def test_others_are_not_notified(self, outbox, fields):
        db = SessionLocal()
        user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x",
                           **{"subscription_tier": "pro", "subscription_interval": "year",
                              "current_period_end": NOW + timedelta(days=60), **fields})
        db.add(user)
        db.commit()
        uid = user.id
        db.close()
        self._run()
        with SessionLocal() as db:
            assert db.query(models.User).filter_by(id=uid).one().renewal_notice_for is None

    def test_notice_shows_the_deadline_in_a_box(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(emails, "_send", lambda to, subject, html, text: captured.update(html=html, text=text))
        emails.send_renewal_notice("client@example.com", "12/12/2026")
        assert "Date limite pour ne pas renouveler : 12/12/2026" in captured["html"]
        assert "border:2px solid" in captured["html"]
        assert "/consommation" in captured["text"]
