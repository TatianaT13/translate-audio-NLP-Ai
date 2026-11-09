"""Tests des endpoints service-to-service de facturation (Pipeline → Gateway).

/billing/reserve, /settle, /release : jeton de service obligatoire, puis le
cycle complet d'un job pipeline.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from gateway import credits, models
from gateway.database import SessionLocal
from gateway.main import app

HEADERS = {"x-service-token": "test-service-token"}

client = TestClient(app)


@pytest.fixture
def user_id() -> int:
    """Un user neuf avec ses 10 credits d'essai."""
    db = SessionLocal()
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    credits.grant_trial(db, user.id)
    uid = user.id
    db.close()
    return uid


def _reserve(user_id: int, amount: int) -> dict:
    r = client.post("/billing/reserve", headers=HEADERS,
                    json={"user_id": user_id, "kind": "processing", "amount": amount})
    assert r.status_code == 200
    return r.json()


@pytest.mark.parametrize("path, body", [
    ("/billing/reserve", {"user_id": 1, "kind": "processing", "amount": 1}),
    ("/billing/settle",  {"hold_id": "x", "amount": 1, "reason": "job"}),
    ("/billing/release", {"hold_id": "x"}),
])
@pytest.mark.parametrize("headers", [{}, {"x-service-token": "mauvais"}])
def test_service_token_is_required(path, body, headers):
    assert client.post(path, json=body, headers=headers).status_code == 401


def test_job_lifecycle_reserve_then_settle(user_id):
    reserved = _reserve(user_id, 8)
    assert reserved["ok"] is True and reserved["hold_id"]

    r = client.post("/billing/settle", headers=HEADERS,
                    json={"hold_id": reserved["hold_id"], "amount": 3, "reason": "pipeline_process:90s:audio_translated"})
    assert r.json() == {"ok": True, "charged": 3, "balance": 7}


def test_second_parallel_job_is_refused(user_id):
    assert _reserve(user_id, 8)["ok"] is True
    second = _reserve(user_id, 8)
    assert second["ok"] is False and second["hold_id"] is None
    assert second["balance"] == 10 and second["needed"] == 8


def test_settle_replayed_charges_once(user_id):
    hold_id = _reserve(user_id, 8)["hold_id"]
    body = {"hold_id": hold_id, "amount": 3, "reason": "job"}
    client.post("/billing/settle", headers=HEADERS, json=body)
    replay = client.post("/billing/settle", headers=HEADERS, json=body).json()
    assert replay == {"ok": True, "charged": 0, "already_settled": True}


def test_release_makes_credits_available_again(user_id):
    hold_id = _reserve(user_id, 10)["hold_id"]
    assert client.post("/billing/release", headers=HEADERS, json={"hold_id": hold_id}).json()["released"] is True
    assert _reserve(user_id, 10)["ok"] is True


def test_pre_authorize_sees_holds(user_id):
    """Le STT (compte-rendu) ne doit pas engager des credits reserves par un job."""
    _reserve(user_id, 10)
    r = client.post("/billing/pre-authorize", headers=HEADERS,
                    json={"user_id": user_id, "kind": "processing", "amount": 1})
    assert r.json()["ok"] is False


def test_simultaneous_reserves_grant_only_one(user_id):
    """10 envois lances en meme temps avec un solde pour un seul job."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: _reserve(user_id, 8), range(10)))
    assert sum(r["ok"] for r in results) == 1


@pytest.mark.parametrize("amount", [0, -5])
def test_reserve_rejects_non_positive_amount(user_id, amount):
    r = client.post("/billing/reserve", headers=HEADERS,
                    json={"user_id": user_id, "kind": "processing", "amount": amount})
    assert r.status_code == 400


def test_reserve_unknown_user(user_id):
    r = client.post("/billing/reserve", headers=HEADERS,
                    json={"user_id": 10_000_000, "kind": "processing", "amount": 1})
    assert r.status_code == 404
