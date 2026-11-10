"""Tests unitaires — facturation des appels directs au STT (comptes-rendus)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from stt import billing


def _gateway(monkeypatch, answer):
    async def fake(user_id):
        return answer
    monkeypatch.setattr(billing, "_pre_authorize", fake)


def test_rate_for_plain_transcription(monkeypatch):
    _gateway(monkeypatch, {"ok": True, "balance": 10, "tier": "free"})
    assert asyncio.run(billing.rate_for(1, diarize=False)) == billing.TRANSCRIPTION_RATE


def test_no_credits_is_402(monkeypatch):
    _gateway(monkeypatch, {"ok": False, "balance": 0, "tier": "free"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(billing.rate_for(1, diarize=False))
    assert exc.value.status_code == 402


def test_gateway_unreachable_blocks_the_transcription(monkeypatch):
    """Avant : taux 0, donc transcription gratuite tant que la Gateway ne répondait pas."""
    _gateway(monkeypatch, None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(billing.rate_for(1, diarize=False))
    assert exc.value.status_code == 503
