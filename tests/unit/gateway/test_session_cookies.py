"""Tests — session du navigateur par cookies HttpOnly.

Les jetons ne sont plus lisibles par le JavaScript de la page : un script
injecté (XSS) ne peut plus emporter la session.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from gateway import auth, main, models
from gateway.database import SessionLocal

PASSWORD = "Motdepasse2026!"


@pytest.fixture
def account() -> str:
    db = SessionLocal()
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password=auth.hash_password(PASSWORD))
    db.add(user)
    db.commit()
    email = user.email
    db.close()
    return email


@pytest.fixture
def browser() -> TestClient:
    """Un navigateur neuf : son propre pot de cookies."""
    return TestClient(main.app)


def _login(browser: TestClient, email: str):
    r = browser.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200
    return r


def _set_cookie_headers(response) -> dict[str, str]:
    return {h.split("=", 1)[0]: h.lower() for h in response.headers.get_list("set-cookie")}


class TestLogin:
    def test_tokens_are_set_as_httponly_cookies(self, browser, account):
        cookies = _set_cookie_headers(_login(browser, account))
        assert "httponly" in cookies["access_token"] and "samesite=lax" in cookies["access_token"]
        assert "httponly" in cookies["refresh_token"] and "samesite=lax" in cookies["refresh_token"]

    def test_session_marker_is_readable_by_the_site_and_holds_no_secret(self, browser, account):
        cookies = _set_cookie_headers(_login(browser, account))
        assert cookies["session"].startswith("session=1;") and "httponly" not in cookies["session"]

    def test_cookies_are_secure_on_https(self, browser, account, monkeypatch):
        monkeypatch.setattr(main, "_COOKIE_SECURE", True)
        assert all("secure" in c for c in _set_cookie_headers(_login(browser, account)).values())

    def test_session_cookie_alone_authenticates(self, browser, account):
        _login(browser, account)
        assert browser.get("/auth/me").json()["email"] == account      # aucun en-tête Authorization

    def test_no_cookie_no_header_is_401(self, browser):
        assert browser.get("/auth/me").status_code == 401

    def test_api_clients_can_still_use_the_authorization_header(self, browser, account):
        token = _login(browser, account).json()["access_token"]
        fresh = TestClient(main.app)
        assert fresh.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["email"] == account


class TestRefresh:
    def test_refresh_from_cookie_rotates_without_exposing_tokens(self, browser, account):
        _login(browser, account)
        old = browser.cookies.get("refresh_token")

        r = browser.post("/auth/refresh")

        assert r.status_code == 200 and r.json() == {"ok": True}       # pas de jeton lisible par un script
        assert browser.cookies.get("refresh_token") not in (None, old)
        assert browser.get("/auth/me").status_code == 200

    def test_a_rotated_refresh_token_cannot_be_replayed(self, browser, account):
        _login(browser, account)
        old = browser.cookies.get("refresh_token")
        browser.post("/auth/refresh")
        thief = TestClient(main.app)
        thief.cookies.set("refresh_token", old)
        assert thief.post("/auth/refresh").status_code == 401

    def test_refresh_without_any_token_is_401(self, browser):
        assert browser.post("/auth/refresh").status_code == 401

    def test_api_clients_refresh_with_the_token_in_the_body(self, browser, account):
        refresh = _login(browser, account).json()["refresh_token"]
        fresh = TestClient(main.app)
        r = fresh.post("/auth/refresh", json={"refresh_token": refresh})
        assert r.status_code == 200 and r.json()["access_token"]
        assert r.headers.get_list("set-cookie") == []


class TestLogout:
    def test_logout_clears_cookies_and_revokes_the_session(self, browser, account):
        _login(browser, account)
        refresh = browser.cookies.get("refresh_token")

        assert browser.post("/auth/logout").status_code == 200

        assert browser.cookies.get("access_token") is None and browser.cookies.get("session") is None
        assert browser.get("/auth/me").status_code == 401
        thief = TestClient(main.app)
        thief.cookies.set("refresh_token", refresh)
        assert thief.post("/auth/refresh").status_code == 401

    def test_logout_works_with_an_expired_access_token(self, browser, account):
        _login(browser, account)
        browser.cookies.delete("access_token")                          # expiré : le navigateur ne l'envoie plus
        assert browser.post("/auth/logout").status_code == 200
        assert browser.post("/auth/refresh").status_code == 401

    def test_logout_without_session_is_harmless(self, browser):
        assert browser.post("/auth/logout").status_code == 200

    def test_changing_the_password_ends_the_session(self, browser, account):
        _login(browser, account)
        r = browser.post("/auth/change-password", json={"old_password": PASSWORD, "new_password": "Nouveau2026!!"})
        assert r.status_code == 200
        assert browser.cookies.get("access_token") is None
        assert browser.post("/auth/refresh").status_code == 401         # tous les appareils déconnectés
