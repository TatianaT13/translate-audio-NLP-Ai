"""Acces reserve : JWT utilisateur (navigateur) ou jeton de service (appels internes)."""
import hmac
import os

from fastapi import Cookie, Header, HTTPException
from jose import JWTError, jwt

JWT_SECRET    = os.getenv("JWT_SECRET", "")
SERVICE_TOKEN = os.getenv("GATEWAY_SERVICE_TOKEN", "")


def _is_service(x_service_token: str | None) -> bool:
    # La valeur par defaut du docker-compose est publique (dans le repo) : jamais acceptee.
    return bool(SERVICE_TOKEN and not SERVICE_TOKEN.startswith("change-me") and x_service_token
                and hmac.compare_digest(x_service_token, SERVICE_TOKEN))


def require_caller(
    authorization: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
    access_token: str | None = Cookie(default=None),
) -> int | None:
    """user_id pour un JWT valide, None pour un appel interne ; sinon 401.

    Le JWT vient de l'en-tete Authorization (clients d'API) ou du cookie de
    session pose par la Gateway (navigateur).
    """
    if _is_service(x_service_token):
        return None
    tokens = [access_token]
    if authorization and authorization.startswith("Bearer "):
        tokens.insert(0, authorization.removeprefix("Bearer ").strip())
    for token in tokens:
        if not (JWT_SECRET and token):
            continue
        try:
            return int(jwt.decode(token, JWT_SECRET, algorithms=["HS256"])["sub"])
        except (JWTError, KeyError, ValueError):
            pass
    raise HTTPException(status_code=401, detail="Authentification requise")


def require_service(x_service_token: str | None = Header(default=None)) -> None:
    """Endpoints internes uniquement (jamais appeles par le navigateur)."""
    if not _is_service(x_service_token):
        raise HTTPException(status_code=401, detail="Authentification requise")
