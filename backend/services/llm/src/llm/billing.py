"""Facturation des appels directs du navigateur (page Compte-rendu).

Les appels internes (jeton de service) ne passent pas ici : le pipeline facture
lui-meme ses traitements. Gateway injoignable : on laisse passer, comme le pipeline.
"""
import os

import httpx
from fastapi import HTTPException

GATEWAY_URL   = os.getenv("GATEWAY_URL", "http://gateway:8004")
SERVICE_TOKEN = os.getenv("GATEWAY_SERVICE_TOKEN", "")


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


async def ensure_can_summarize(user_id: int) -> None:
    """Compte-rendu inclus : toujours en Pro, en gratuit tant qu'il reste des credits."""
    info = await _pre_authorize(user_id)
    if info is None:
        return
    if info.get("tier", "free") == "free" and not info.get("ok"):
        raise _insufficient(info)
