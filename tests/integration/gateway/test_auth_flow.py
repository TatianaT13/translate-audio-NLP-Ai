"""Test d'intégration — flow d'authentification complet.

Couvre : register → login → /auth/me → refresh (rotation) → logout.
Prérequis : gateway service up sur 127.0.0.1:8004.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


class TestAuthFlow:
    """Le flow d'auth JWT complet fonctionne bout-en-bout."""

    def test_register_creates_account(self, client, gateway_url, test_user_credentials):
        r = client.post(f"{gateway_url}/auth/register", json=test_user_credentials)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["email"] == test_user_credentials["email"]

    def test_register_duplicate_email_returns_409(self, client, gateway_url, test_user_credentials):
        client.post(f"{gateway_url}/auth/register", json=test_user_credentials)
        r = client.post(f"{gateway_url}/auth/register", json=test_user_credentials)
        assert r.status_code == 409

    def test_login_returns_tokens(self, client, gateway_url, test_user_credentials):
        client.post(f"{gateway_url}/auth/register", json=test_user_credentials)
        r = client.post(f"{gateway_url}/auth/login", json=test_user_credentials)
        assert r.status_code == 200
        tokens = r.json()
        assert tokens["access_token"]
        assert tokens["refresh_token"]
        assert tokens["access_token"] != tokens["refresh_token"]

    def test_login_wrong_password_returns_401(self, client, gateway_url, test_user_credentials):
        client.post(f"{gateway_url}/auth/register", json=test_user_credentials)
        bad = {**test_user_credentials, "password": "WrongPassword!123"}
        r = client.post(f"{gateway_url}/auth/login", json=bad)
        assert r.status_code == 401

    def test_me_returns_user_info(self, client, gateway_url, registered_user, auth_headers):
        r = client.get(f"{gateway_url}/auth/me", headers=auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == registered_user["email"]
        assert body["is_admin"] is False
        assert body["id"] > 0
        assert body["created_at"]  # ISO string non vide

    def test_me_without_token_returns_401_or_403(self, client, gateway_url):
        r = client.get(f"{gateway_url}/auth/me")
        assert r.status_code in (401, 403)

    def test_me_with_invalid_token_returns_401(self, client, gateway_url):
        r = client.get(f"{gateway_url}/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert r.status_code == 401

    def test_refresh_rotates_tokens(self, client, gateway_url, registered_user):
        """Le refresh renouvelle les tokens ET révoque l'ancien refresh (rotation)."""
        old_refresh = registered_user["refresh_token"]

        r = client.post(f"{gateway_url}/auth/refresh", json={"refresh_token": old_refresh})
        assert r.status_code == 200
        new_tokens = r.json()
        assert new_tokens["refresh_token"] != old_refresh

        # L'ancien refresh doit être révoqué → refuse la 2ème utilisation
        r2 = client.post(f"{gateway_url}/auth/refresh", json={"refresh_token": old_refresh})
        assert r2.status_code == 401

    def test_logout_revokes_refresh_token(self, client, gateway_url, registered_user, auth_headers):
        r = client.post(
            f"{gateway_url}/auth/logout",
            headers=auth_headers,
            json={"refresh_token": registered_user["refresh_token"]},
        )
        assert r.status_code == 200

        # Le refresh révoqué ne peut plus être utilisé
        r2 = client.post(
            f"{gateway_url}/auth/refresh",
            json={"refresh_token": registered_user["refresh_token"]},
        )
        assert r2.status_code == 401

    def test_admin_endpoint_forbidden_for_regular_user(self, client, gateway_url, auth_headers):
        r = client.get(f"{gateway_url}/admin/stats", headers=auth_headers)
        assert r.status_code == 403
