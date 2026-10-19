"""Facturation des appels directs du navigateur (page Compte-rendu).

Les appels internes (jeton de service) ne passent pas ici : le pipeline facture
lui-meme ses traitements. Gateway injoignable : on laisse passer, comme le pipeline.
"""
import os

import httpx
from fastapi import HTTPException

GATEWAY_URL   = os.getenv("GATEWAY_URL", "http://gateway:8004")
SERVICE_TOKEN = os.getenv("GATEWAY_SERVICE_TOKEN", "")

# Bareme affiche sur /tarifs, identique pour toutes les offres.
TRANSCRIPTION_RATE = 0.5
DIARIZATION_RATE   = 0.5


async def _pre_authorize(user_id: int) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"{GATEWAY_URL}/billing/pre-authorize",
                json={"user_id": user_id, "kind": "processing", "amount": 1},
                headers={"x-service-token": SERVICE_TOKEN},
            )
        if r.status_code != 200:
            print(f"[billing] pre-authorize HTTP {r.status_code} (non bloquant)", flush=True)
            return None
        return r.json()
    except Exception as e:
        print(f"[billing] pre-authorize injoignable (non bloquant): {e}", flush=True)
        return None


def _insufficient(info: dict) -> HTTPException:
    return HTTPException(status_code=402, detail={
        "reason":  "credits_insufficient",
        "balance": info.get("balance"),
        "tier":    info.get("tier"),
    })


async def rate_for(user_id: int, diarize: bool) -> float:
    """Taux en credit/min pour cet appel ; 402 si le solde est vide."""
    info = await _pre_authorize(user_id)
    if info is None:
        return 0.0
    rate = DIARIZATION_RATE if diarize else TRANSCRIPTION_RATE
    if not info.get("ok"):
        raise _insufficient(info)
    return rate


async def charge(user_id: int, seconds: float, rate: float, reason: str) -> None:
    if rate <= 0 or seconds <= 0:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{GATEWAY_URL}/billing/consume-seconds",
                json={"user_id": user_id, "seconds": seconds, "rate_per_min": rate, "reason": reason},
                headers={"x-service-token": SERVICE_TOKEN},
            )
    except Exception as e:
        print(f"[billing] consume-seconds echoue pour user {user_id}: {e}", flush=True)
