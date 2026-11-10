"""Tests unitaires — le pipeline et le STT acceptent le cookie de session du navigateur."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from pipeline import main
from stt import auth as stt_auth

SECRET = "s" * 32
TOKEN = jwt.encode({"sub": "7", "type": "access"}, SECRET, algorithm="HS256")


@pytest.fixture(autouse=True)
def secret(monkeypatch):
    monkeypatch.setattr(main, "JWT_SECRET", SECRET)
    monkeypatch.setattr(stt_auth, "JWT_SECRET", SECRET)


def test_pipeline_authenticates_from_the_cookie():
    assert main._authenticate(None, None, TOKEN) == 7


def test_authorization_header_still_works():
    assert main._authenticate(f"Bearer {TOKEN}", None, None) == 7


@pytest.mark.parametrize("cookie", [None, "", "pas-un-jwt", jwt.encode({"sub": "7"}, "autre-secret", algorithm="HS256")])
def test_missing_or_forged_cookie_is_refused(cookie):
    with pytest.raises(HTTPException) as exc:
        main._authenticate(None, None, cookie)
    assert exc.value.status_code == 401


def test_job_status_is_reachable_with_the_cookie_only():
    client = TestClient(main.app)
    client.cookies.set("access_token", TOKEN)
    assert client.get("/jobs/inconnu").status_code == 404        # authentifié : job introuvable, pas 401
    assert TestClient(main.app).get("/jobs/inconnu").status_code == 401


def test_stt_accepts_cookie_or_header():
    assert stt_auth.require_caller(authorization=None, x_service_token=None, access_token=TOKEN) == 7
    assert stt_auth.require_caller(authorization=f"Bearer {TOKEN}", x_service_token=None, access_token=None) == 7
    with pytest.raises(HTTPException):
        stt_auth.require_caller(authorization=None, x_service_token=None, access_token="faux")
