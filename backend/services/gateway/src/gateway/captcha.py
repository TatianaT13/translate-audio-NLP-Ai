"""Verification Cloudflare Turnstile (anti-robot a l'inscription).

Inactive tant que TURNSTILE_SECRET_KEY est vide. Une fois active, elle echoue
en mode ferme : si Cloudflare ne repond pas, l'inscription est refusee plutot
que de laisser passer les robots pendant la panne.
"""
import os

import httpx
from fastapi import HTTPException

TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")
SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def verify_turnstile(token: str | None, remote_ip: str | None) -> None:
    if not TURNSTILE_SECRET_KEY:
        return
    if not token:
        raise HTTPException(status_code=400, detail="Vérification anti-robot manquante.")
    data = {"secret": TURNSTILE_SECRET_KEY, "response": token}
    if remote_ip:
        data["remoteip"] = remote_ip
    try:
        r = httpx.post(SITEVERIFY_URL, data=data, timeout=5.0)
        ok = r.status_code == 200 and r.json().get("success") is True
    except Exception as e:
        print(f"[captcha] siteverify injoignable: {e}", flush=True)
        raise HTTPException(status_code=503, detail="Vérification anti-robot indisponible, réessayez dans un instant.")
    if not ok:
        raise HTTPException(status_code=400, detail="Vérification anti-robot échouée, réessayez.")
