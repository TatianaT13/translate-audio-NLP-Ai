"""Fixtures pour les tests d'intégration — appels HTTP réels aux services.

Prérequis : `docker compose up -d` (services accessibles sur 127.0.0.1).
Skip élégant si un service ne répond pas.
"""

from __future__ import annotations

import os
import secrets
from typing import Iterator

import httpx
import pytest


# ── URLs — override par variable d'env si besoin ─────────────────────────────

GATEWAY_URL  = os.getenv("TEST_GATEWAY_URL",  "http://127.0.0.1:8004")
PIPELINE_URL = os.getenv("TEST_PIPELINE_URL", "http://127.0.0.1:8000")
STT_URL      = os.getenv("TEST_STT_URL",      "http://127.0.0.1:8001")
LLM_URL      = os.getenv("TEST_LLM_URL",      "http://127.0.0.1:8002")
TTS_URL      = os.getenv("TEST_TTS_URL",      "http://127.0.0.1:8003")


def _service_up(url: str, timeout: float = 2.0) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


@pytest.fixture(scope="session")
def gateway_url() -> str:
    if not _service_up(GATEWAY_URL):
        pytest.skip(f"Gateway indisponible sur {GATEWAY_URL} (docker compose up ?)")
    return GATEWAY_URL


@pytest.fixture(scope="session")
def pipeline_url() -> str:
    if not _service_up(PIPELINE_URL):
        pytest.skip(f"Pipeline indisponible sur {PIPELINE_URL} (docker compose up ?)")
    return PIPELINE_URL


@pytest.fixture(scope="session")
def stt_url() -> str:
    if not _service_up(STT_URL):
        pytest.skip(f"STT indisponible sur {STT_URL}")
    return STT_URL


@pytest.fixture(scope="session")
def llm_url() -> str:
    if not _service_up(LLM_URL):
        pytest.skip(f"LLM indisponible sur {LLM_URL}")
    return LLM_URL


@pytest.fixture(scope="session")
def tts_url() -> str:
    if not _service_up(TTS_URL):
        pytest.skip(f"TTS indisponible sur {TTS_URL}")
    return TTS_URL


# ── Client HTTP partagé ──────────────────────────────────────────────────────

@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=30.0) as c:
        yield c


# ── User de test jetable + token JWT ─────────────────────────────────────────

@pytest.fixture
def test_user_credentials() -> dict:
    """Génère un couple email/password unique par test — évite les collisions
    entre runs successifs sans avoir à nettoyer la DB.
    Utilise example.com (RFC 2606 réservé aux tests) — accepté par Pydantic EmailStr."""
    suffix = secrets.token_hex(6)
    return {
        "email":    f"pytest_{suffix}@example.com",
        "password": f"Pytest!{suffix}_2026",
    }


@pytest.fixture
def registered_user(client: httpx.Client, gateway_url: str, test_user_credentials: dict) -> dict:
    """Crée un user via /auth/register puis login pour récupérer les tokens.
    Renvoie {email, password, access_token, refresh_token}."""
    r = client.post(f"{gateway_url}/auth/register", json=test_user_credentials)
    assert r.status_code in (200, 201), f"register failed: {r.status_code} {r.text}"

    r = client.post(f"{gateway_url}/auth/login", json=test_user_credentials)
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    tokens = r.json()

    return {
        **test_user_credentials,
        "access_token":  tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
    }


@pytest.fixture
def auth_headers(registered_user: dict) -> dict:
    """Headers Authorization prêts à l'emploi."""
    return {"Authorization": f"Bearer {registered_user['access_token']}"}
