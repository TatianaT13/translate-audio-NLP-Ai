"""Limitation de debit par IP, en memoire (gateway mono-worker).

La requete traverse nginx puis le proxy Next.js : l'IP de connexion est celle
du conteneur frontend, commune a tous les clients. Seul X-Real-IP, pose par
nginx a partir de la vraie connexion, identifie le client. Sans cet en-tete,
on desactive la limite plutot que de bloquer tout le monde dans un seul compteur.
"""
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException, Request

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()
_warned_missing_ip = False
_MAX_KEYS = 50_000


def client_ip(request: Request) -> str | None:
    return request.headers.get("x-real-ip")


def rate_limit(scope: str, limit: int, window_s: int):
    def dependency(request: Request) -> None:
        global _warned_missing_ip
        ip = client_ip(request)
        if not ip:
            if not _warned_missing_ip:
                print("[ratelimit] en-tete X-Real-IP absent : limitation desactivee "
                      "(ajouter proxy_set_header X-Real-IP $remote_addr dans nginx)", flush=True)
                _warned_missing_ip = True
            return
        now = time.monotonic()
        key = f"{scope}:{ip}"
        with _lock:
            if len(_hits) > _MAX_KEYS:
                for k in [k for k, q in _hits.items() if not q or now - q[-1] > 3600]:
                    del _hits[k]
            q = _hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                retry_after = int(window_s - (now - q[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Trop de tentatives. Réessayez dans quelques minutes.",
                    headers={"Retry-After": str(retry_after)},
                )
            q.append(now)
    return dependency
