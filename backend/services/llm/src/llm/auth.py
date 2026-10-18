"""Acces reserve : JWT utilisateur (navigateur) ou jeton de service (pipeline, watcher)."""
import hmac
import os

from fastapi import Header, HTTPException
from jose import JWTError, jwt

JWT_SECRET    = os.getenv("JWT_SECRET", "")
SERVICE_TOKEN = os.getenv("GATEWAY_SERVICE_TOKEN", "")


def require_caller(
    authorization: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
) -> None:
    # La valeur par defaut du docker-compose est publique (dans le repo) : jamais acceptee.
    if (SERVICE_TOKEN and not SERVICE_TOKEN.startswith("change-me") and x_service_token
            and hmac.compare_digest(x_service_token, SERVICE_TOKEN)):
        return
    if JWT_SECRET and authorization and authorization.startswith("Bearer "):
        try:
            jwt.decode(authorization.removeprefix("Bearer ").strip(), JWT_SECRET, algorithms=["HS256"])
            return
        except JWTError:
            pass
    raise HTTPException(status_code=401, detail="Authentification requise")
