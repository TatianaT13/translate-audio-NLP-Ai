"""
Gateway Service — Port 8004
Auth complète : register / login / logout / refresh / me /
                change-password / forgot-password / reset-password / delete-account
"""

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

import auth as auth_utils
import models
import schemas
from database import Base, engine, get_db

# ── Init ──────────────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

DEV_MODE      = os.getenv("DEV_MODE", "false").lower() == "true"
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")

LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

EXPERIMENTS_CSV = Path(os.getenv("EXPERIMENTS_CSV", "/app/experiments/results.csv"))

PIPELINE_URL = os.getenv("PIPELINE_URL", "http://pipeline:8000")
STT_URL      = os.getenv("STT_URL",      "http://stt:8001")
LLM_URL      = os.getenv("LLM_URL",      "http://llm:8002")
TTS_URL      = os.getenv("TTS_URL",      "http://tts:8003")
WATCHER_URL  = os.getenv("WATCHER_URL",  "http://watcher:8005")

app = FastAPI(title="Gateway — Auth Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)


# ── Auth dependency ────────────────────────────────────────────────────────────

def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
    db: Session = Depends(get_db),
) -> models.User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié")

    payload = auth_utils.decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide ou expiré")

    user = db.query(models.User).filter(models.User.id == int(payload["sub"])).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Compte introuvable")

    return user


def get_admin_user(current_user: models.User = Depends(get_current_user)) -> models.User:
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Accès administrateur requis")
    return current_user


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(body: schemas.RegisterRequest, db: Session = Depends(get_db)):
    """Créer un nouveau compte."""
    if db.query(models.User).filter(models.User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Cet email est déjà utilisé")

    user = models.User(
        email           = body.email,
        hashed_password = auth_utils.hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "Compte créé avec succès", "email": user.email}


@app.post("/auth/login", response_model=schemas.TokenResponse)
def login(body: schemas.LoginRequest, db: Session = Depends(get_db)):
    """Connexion — retourne access_token + refresh_token."""
    user = db.query(models.User).filter(models.User.email == body.email).first()

    if not user or not auth_utils.verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Compte désactivé")

    access_token          = auth_utils.create_access_token(user.id, user.email, user.is_admin)
    raw_refresh, hash_ref = auth_utils.make_token_pair()

    expires = datetime.now(timezone.utc) + timedelta(days=auth_utils.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(models.RefreshToken(user_id=user.id, token_hash=hash_ref, expires_at=expires))
    db.commit()

    return schemas.TokenResponse(access_token=access_token, refresh_token=raw_refresh)


@app.post("/auth/logout")
def logout(
    body: schemas.RefreshRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Révoquer le refresh token actuel."""
    token_hash = auth_utils.hash_token(body.refresh_token)
    rt = db.query(models.RefreshToken).filter(
        models.RefreshToken.token_hash == token_hash,
        models.RefreshToken.user_id   == current_user.id,
    ).first()
    if rt:
        rt.revoked = True
        db.commit()
    return {"message": "Déconnecté"}


@app.post("/auth/refresh", response_model=schemas.TokenResponse)
def refresh(body: schemas.RefreshRequest, db: Session = Depends(get_db)):
    """Renouveler l'access token via refresh token."""
    token_hash = auth_utils.hash_token(body.refresh_token)
    rt = db.query(models.RefreshToken).filter(
        models.RefreshToken.token_hash == token_hash,
        models.RefreshToken.revoked    == False,  # noqa: E712
    ).first()

    if not rt:
        raise HTTPException(status_code=401, detail="Refresh token invalide")

    if rt.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expiré")

    user = db.query(models.User).filter(models.User.id == rt.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Compte introuvable")

    # Rotate: revoke old, issue new
    rt.revoked = True
    access_token          = auth_utils.create_access_token(user.id, user.email, user.is_admin)
    raw_refresh, hash_ref = auth_utils.make_token_pair()
    expires = datetime.now(timezone.utc) + timedelta(days=auth_utils.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(models.RefreshToken(user_id=user.id, token_hash=hash_ref, expires_at=expires))
    db.commit()

    return schemas.TokenResponse(access_token=access_token, refresh_token=raw_refresh)


@app.get("/auth/me")
def me(current_user: models.User = Depends(get_current_user)):
    """Informations du compte connecté."""
    return {
        "id":         current_user.id,
        "email":      current_user.email,
        "is_admin":   current_user.is_admin,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


@app.post("/auth/change-password")
def change_password(
    body: schemas.ChangePasswordRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Changer le mot de passe (requiert l'ancien)."""
    if not auth_utils.verify_password(body.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Ancien mot de passe incorrect")

    current_user.hashed_password = auth_utils.hash_password(body.new_password)

    # Révoquer tous les refresh tokens (forcer reconnexion sur tous les appareils)
    db.query(models.RefreshToken).filter(
        models.RefreshToken.user_id == current_user.id
    ).update({"revoked": True})

    db.commit()
    return {"message": "Mot de passe modifié. Reconnectez-vous."}


@app.post("/auth/forgot-password")
def forgot_password(body: schemas.ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Demander un lien de réinitialisation.
    En DEV_MODE=true : retourne le token directement.
    En production    : envoyer par email (SMTP à configurer).
    """
    user = db.query(models.User).filter(models.User.email == body.email).first()

    # Toujours répondre OK pour ne pas révéler si l'email existe
    if not user:
        return {"message": "Si cet email existe, un lien a été envoyé."}

    # Invalider les anciens tokens non utilisés
    db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.user_id == user.id,
        models.PasswordResetToken.used    == False,  # noqa: E712
    ).update({"used": True})

    raw_token, hash_tok = auth_utils.make_token_pair()
    expires = datetime.now(timezone.utc) + timedelta(hours=auth_utils.RESET_TOKEN_EXPIRE_HOURS)
    db.add(models.PasswordResetToken(user_id=user.id, token_hash=hash_tok, expires_at=expires))
    db.commit()

    reset_url = f"http://localhost:3000/reset-password?token={raw_token}"

    if DEV_MODE:
        # En développement : retourner le lien directement
        return {"message": "Lien de réinitialisation (DEV uniquement)", "reset_url": reset_url}

    # TODO production : envoyer l'email via SMTP
    # send_reset_email(user.email, reset_url)
    return {"message": "Si cet email existe, un lien a été envoyé."}


@app.post("/auth/reset-password")
def reset_password(body: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    """Réinitialiser le mot de passe avec le token reçu."""
    token_hash = auth_utils.hash_token(body.token)
    rt = db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.token_hash == token_hash,
        models.PasswordResetToken.used       == False,  # noqa: E712
    ).first()

    if not rt:
        raise HTTPException(status_code=400, detail="Lien invalide ou déjà utilisé")

    if rt.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Lien expiré (valide 1h)")

    user = db.query(models.User).filter(models.User.id == rt.user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="Compte introuvable")

    user.hashed_password = auth_utils.hash_password(body.new_password)
    rt.used = True

    # Révoquer tous les refresh tokens
    db.query(models.RefreshToken).filter(
        models.RefreshToken.user_id == user.id
    ).update({"revoked": True})

    db.commit()
    return {"message": "Mot de passe réinitialisé avec succès. Connectez-vous."}


@app.delete("/auth/account")
def delete_account(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Supprimer définitivement le compte et toutes ses données."""
    db.query(models.RefreshToken).filter(
        models.RefreshToken.user_id == current_user.id
    ).delete()
    db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.user_id == current_user.id
    ).delete()
    db.delete(current_user)
    db.commit()
    return {"message": "Compte supprimé définitivement"}


# ── Admin endpoints ────────────────────────────────────────────────────────────

@app.post("/admin/seed", status_code=status.HTTP_201_CREATED)
def admin_seed(db: Session = Depends(get_db)):
    """
    DEV_MODE only — promote or create the first admin account.
    Call once after first registration.
    """
    if not DEV_MODE:
        raise HTTPException(status_code=403, detail="Disponible uniquement en DEV_MODE")
    user = db.query(models.User).first()
    if not user:
        raise HTTPException(status_code=404, detail="Aucun utilisateur enregistré")
    user.is_admin = True
    db.commit()
    return {"message": f"{user.email} est maintenant administrateur"}


@app.get("/admin/stats", response_model=schemas.AdminStatsResponse)
def admin_stats(
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Statistiques globales des utilisateurs."""
    total  = db.query(models.User).count()
    active = db.query(models.User).filter(models.User.is_active == True).count()   # noqa: E712
    admins = db.query(models.User).filter(models.User.is_admin  == True).count()   # noqa: E712
    return schemas.AdminStatsResponse(total_users=total, active_users=active, admin_users=admins)


@app.get("/admin/users", response_model=list[schemas.AdminUserResponse])
def admin_list_users(
    _: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Liste complète des utilisateurs."""
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    return [
        schemas.AdminUserResponse(
            id=u.id,
            email=u.email,
            is_active=u.is_active,
            is_admin=u.is_admin,
            created_at=u.created_at.isoformat() if u.created_at else None,
        )
        for u in users
    ]


@app.patch("/admin/users/{user_id}")
def admin_update_user(
    user_id: int,
    body: schemas.AdminUserUpdate,
    current_admin: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Modifier is_active ou is_admin d'un utilisateur."""
    if user_id == current_admin.id:
        raise HTTPException(status_code=400, detail="Impossible de modifier son propre compte")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    db.commit()
    return {"message": "Utilisateur mis à jour"}


@app.delete("/admin/users/{user_id}")
def admin_delete_user(
    user_id: int,
    current_admin: models.User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Supprimer un utilisateur (admin uniquement)."""
    if user_id == current_admin.id:
        raise HTTPException(status_code=400, detail="Impossible de supprimer son propre compte ici")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    db.query(models.RefreshToken).filter(models.RefreshToken.user_id == user_id).delete()
    db.query(models.PasswordResetToken).filter(models.PasswordResetToken.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"message": "Utilisateur supprimé"}


@app.get("/admin/langfuse/metrics", response_model=schemas.LangfuseMetricsResponse)
async def admin_langfuse_metrics(_: models.User = Depends(get_admin_user)):
    """Métriques agrégées depuis Langfuse (scores)."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return schemas.LangfuseMetricsResponse(
            connected=False,
            error="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY non configurées",
        )

    try:
        all_scores: list[dict] = []
        page = 1
        async with httpx.AsyncClient(timeout=15.0) as client:
            while True:
                r = await client.get(
                    f"{LANGFUSE_HOST}/api/public/scores",
                    auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
                    params={"limit": 100, "page": page},
                )
                if not r.is_success:
                    return schemas.LangfuseMetricsResponse(
                        connected=False,
                        error=f"Langfuse HTTP {r.status_code}",
                    )
                data   = r.json()
                batch  = data.get("data", [])
                all_scores.extend(batch)
                if len(batch) < 100:
                    break
                page += 1

        # ── Aggregate by metric name ──
        by_name: dict[str, list[float]] = {}
        for s in all_scores:
            by_name.setdefault(s["name"], []).append(s["value"])

        def avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        # ── Model comparison from comments ──
        # comment format: "{audio} | {whisper} | {llm} | {prompt_version}"
        model_map: dict[str, dict] = {}
        for s in all_scores:
            if s["name"] != "latency_total_ms":
                continue
            comment = s.get("comment") or ""
            parts   = [p.strip() for p in comment.split("|")]
            if len(parts) < 4:
                continue
            whisper, llm, pv = parts[1], parts[2], parts[3]
            key = f"{whisper}|{llm}|{pv}"
            if key not in model_map:
                model_map[key] = {
                    "whisper": whisper, "llm": llm, "prompt_version": pv,
                    "count": 0, "totals": [], "stts": [], "llms_lat": [], "bleus": [],
                }
            model_map[key]["count"]    += 1
            model_map[key]["totals"].append(s["value"])

        # Fill STT / LLM latency and BLEU per key
        for s in all_scores:
            comment = s.get("comment") or ""
            parts   = [p.strip() for p in comment.split("|")]
            if len(parts) < 4:
                continue
            key = f"{parts[1]}|{parts[2]}|{parts[3]}"
            if key not in model_map:
                continue
            if s["name"] == "latency_stt_ms":
                model_map[key]["stts"].append(s["value"])
            elif s["name"] == "latency_llm_ms":
                model_map[key]["llms_lat"].append(s["value"])
            elif s["name"] == "bleu":
                model_map[key]["bleus"].append(s["value"])
            elif s["name"] == "meteor":
                model_map[key].setdefault("meteors", []).append(s["value"])
            elif s["name"] == "wer":
                model_map[key].setdefault("wers", []).append(s["value"])

        model_stats = [
            schemas.LangfuseModelStat(
                whisper=v["whisper"],
                llm=v["llm"],
                prompt_version=v["prompt_version"],
                count=v["count"],
                avg_total_ms=round(avg(v["totals"]), 1),
                avg_stt_ms=round(avg(v["stts"]),   1),
                avg_llm_ms=round(avg(v["llms_lat"]),1),
                avg_bleu=round(avg(v["bleus"]), 3) if v.get("bleus") else None,
                avg_meteor=round(avg(v["meteors"]), 4) if v.get("meteors") else None,
                avg_wer=round(avg(v["wers"]), 4) if v.get("wers") else None,
            )
            for v in sorted(model_map.values(), key=lambda x: x["count"], reverse=True)
        ]

        return schemas.LangfuseMetricsResponse(
            connected=True,
            total_traces=len(by_name.get("latency_total_ms", [])),
            avg_total_ms=round(avg(by_name.get("latency_total_ms", [])), 1),
            avg_stt_ms=round(avg(by_name.get("latency_stt_ms",   [])), 1),
            avg_llm_ms=round(avg(by_name.get("latency_llm_ms",   [])), 1),
            avg_language_prob=round(avg(by_name.get("language_prob", [])), 3),
            avg_bleu=round(avg(by_name.get("bleu", [])), 3),
            avg_meteor=round(avg(by_name.get("meteor", [])), 4),
            avg_wer=round(avg(by_name.get("wer", [])), 4),
            bleu_scores=by_name.get("bleu", []),
            meteor_scores=by_name.get("meteor", []),
            wer_scores=by_name.get("wer", []),
            language_probs=by_name.get("language_prob", []),
            latencies_total=by_name.get("latency_total_ms", []),
            model_stats=model_stats,
        )

    except Exception as exc:
        return schemas.LangfuseMetricsResponse(connected=False, error=str(exc))


@app.get("/admin/traffic/events")
async def admin_traffic_events(_: models.User = Depends(get_admin_user)):
    """Snapshot des événements trafic actuels (max 4 par zone)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{WATCHER_URL}/events")
            return r.json()
    except Exception as exc:
        return {"error": str(exc), "nord": [], "sud": [], "ouest": []}


@app.get("/admin/traffic/stream")
async def admin_traffic_stream(
    token: str | None = None,
    db: Session = Depends(get_db),
):
    """SSE proxy vers le watcher.
    Auth via query param ?token=<access_token> (EventSource ne supporte pas les headers).
    """
    if not token:
        raise HTTPException(status_code=401, detail="Token requis")
    payload = auth_utils.decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token invalide")
    user = db.query(models.User).filter(models.User.id == int(payload["sub"])).first()
    if not user or not user.is_active or not user.is_admin:
        raise HTTPException(status_code=403, detail="Accès administrateur requis")

    async def generator():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", f"{WATCHER_URL}/stream") as r:
                async for chunk in r.aiter_text():
                    yield chunk
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/admin/experiments", response_model=schemas.ExperimentsResponse)
def admin_experiments(_: models.User = Depends(get_admin_user)):
    """Retourne tous les runs d'évaluation depuis results.csv."""
    if not EXPERIMENTS_CSV.exists():
        return schemas.ExperimentsResponse(runs=[], total=0, csv_exists=False)

    runs: list[schemas.ExperimentRun] = []
    with EXPERIMENTS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            def _f(k: str) -> float | None:
                v = row.get(k, "").strip()
                try:
                    return float(v) if v not in ("", "-1", "n/a") else None
                except ValueError:
                    return None
            runs.append(schemas.ExperimentRun(
                run_id           = row.get("run_id", ""),
                audio            = row.get("audio", ""),
                zone             = row.get("zone", ""),
                whisper_model    = row.get("whisper_model", ""),
                llm_model        = row.get("llm_model", ""),
                prompt_version   = row.get("prompt_version", ""),
                target_lang      = row.get("target_lang", ""),
                language_prob    = _f("language_prob"),
                latency_stt_ms   = _f("latency_stt_ms"),
                latency_llm_ms   = _f("latency_llm_ms"),
                latency_total_ms = _f("latency_total_ms"),
                bleu             = _f("bleu"),
                meteor           = _f("meteor"),
                wer              = _f("wer"),
            ))
    return schemas.ExperimentsResponse(runs=runs, total=len(runs), csv_exists=True)


@app.get("/admin/services/health")
async def admin_services_health(_: models.User = Depends(get_admin_user)):
    """Ping tous les microservices et retourne leur statut + latence."""
    services = [
        {"name": "Gateway",  "url": "http://localhost:8004/health", "port": "8004", "color": "#c9a96e"},
        {"name": "Pipeline", "url": f"{PIPELINE_URL}/health",       "port": "8000", "color": "#7ec9a0"},
        {"name": "STT",      "url": f"{STT_URL}/health",            "port": "8001", "color": "#7eb8c9"},
        {"name": "LLM",      "url": f"{LLM_URL}/health",            "port": "8002", "color": "#c9a96e"},
        {"name": "TTS",      "url": f"{TTS_URL}/health",            "port": "8003", "color": "#9b7ec9"},
        {"name": "Watcher",  "url": f"{WATCHER_URL}/health",        "port": "8005", "color": "#e87070"},
    ]
    results = []
    async with httpx.AsyncClient(timeout=3.0) as client:
        for svc in services:
            import time as _time
            t0 = _time.perf_counter()
            try:
                r = await client.get(svc["url"])
                latency_ms = round((_time.perf_counter() - t0) * 1000)
                results.append({
                    "name":       svc["name"],
                    "port":       svc["port"],
                    "color":      svc["color"],
                    "status":     "up" if r.is_success else "error",
                    "latency_ms": latency_ms,
                    "detail":     r.json() if r.is_success else {},
                })
            except Exception:
                latency_ms = round((_time.perf_counter() - t0) * 1000)
                results.append({
                    "name":       svc["name"],
                    "port":       svc["port"],
                    "color":      svc["color"],
                    "status":     "down",
                    "latency_ms": latency_ms,
                    "detail":     {},
                })
    return {"services": results}
