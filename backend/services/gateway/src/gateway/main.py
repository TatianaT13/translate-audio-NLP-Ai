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

from gateway import auth as auth_utils
from gateway import models, schemas
from gateway.database import Base, engine, get_db

# ── Init ──────────────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

DEV_MODE      = os.getenv("DEV_MODE", "false").lower() == "true"
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")

LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

EXPERIMENTS_CSV = Path(os.getenv("EXPERIMENTS_CSV", "/app/experiments/results.csv"))

# Cache CSV en mémoire (évite deadlock avec VS Code sur macOS + rapide)
_csv_cache: dict = {"mtime": 0.0, "rows": []}

def _read_experiments_csv() -> list[dict]:
    """Lit le CSV avec cache basé sur mtime. Copie bytes en mémoire pour éviter
    tout conflit de lock avec d'autres lecteurs (IDE, éditeur)."""
    if not EXPERIMENTS_CSV.exists():
        return []
    try:
        mtime = EXPERIMENTS_CSV.stat().st_mtime
    except OSError:
        return _csv_cache["rows"]
    if mtime == _csv_cache["mtime"] and _csv_cache["rows"]:
        return _csv_cache["rows"]
    try:
        raw = EXPERIMENTS_CSV.read_bytes()
    except OSError:
        return _csv_cache["rows"]
    import io as _io
    reader = csv.DictReader(_io.StringIO(raw.decode("utf-8", errors="replace")), delimiter=";")
    rows = list(reader)
    _csv_cache.update(mtime=mtime, rows=rows)
    return rows

PIPELINE_URL = os.getenv("PIPELINE_URL", "http://pipeline:8000")
MLFLOW_URL   = os.getenv("MLFLOW_URL",   "http://mlflow:5000")
AIRFLOW_URL  = os.getenv("AIRFLOW_URL",  "http://airflow-webserver:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PWD  = os.getenv("AIRFLOW_PASSWORD", "admin")
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

# Prometheus /metrics endpoint (req/s, latency p50/p95/p99 par route, etc.)
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator(excluded_handlers=["/health", "/metrics"]).instrument(app).expose(app)

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
    """Métriques agrégées depuis results.csv (source de vérité pour l'évaluation batch).
    Langfuse reste utilisé pour le tracing en temps réel des runs individuels.
    """
    # Lire depuis le CSV si disponible — agrégation propre et complète
    csv_rows = _read_experiments_csv()
    if csv_rows:
        bleus, meteors, wers, tts_wers = [], [], [], []
        stts, llms_lat, totals, langs = [], [], [], []
        model_map: dict[str, dict] = {}

        for row in csv_rows:
            if True:
                def _fl(k: str) -> float | None:
                    v = row.get(k, "").strip()
                    if v in ("", "-1.0", "-1", "n/a"):
                        return None
                    try:
                        return float(v)
                    except ValueError:
                        return None

                bleu   = _fl("bleu")
                meteor = _fl("meteor")
                wer    = _fl("wer")
                tts_wer = _fl("tts_wer")
                stt_ms   = _fl("latency_stt_ms")
                llm_ms   = _fl("latency_llm_ms")
                total_ms = _fl("latency_total_ms")
                lang_prob = _fl("language_prob")

                if bleu    is not None: bleus.append(bleu)
                if meteor  is not None: meteors.append(meteor)
                if wer     is not None: wers.append(wer)
                if tts_wer is not None: tts_wers.append(tts_wer)
                if stt_ms    is not None: stts.append(stt_ms)
                if llm_ms    is not None: llms_lat.append(llm_ms)
                if total_ms  is not None: totals.append(total_ms)
                if lang_prob is not None: langs.append(lang_prob)

                key = f"{row['whisper_model']}|{row['llm_model']}|{row['prompt_version']}"
                m = model_map.setdefault(key, {
                    "whisper": row["whisper_model"], "llm": row["llm_model"],
                    "prompt_version": row["prompt_version"],
                    "count": 0, "totals": [], "stts": [], "llms_lat": [],
                    "bleus": [], "meteors": [], "wers": [],
                })
                m["count"] += 1
                if total_ms is not None: m["totals"].append(total_ms)
                if stt_ms   is not None: m["stts"].append(stt_ms)
                if llm_ms   is not None: m["llms_lat"].append(llm_ms)
                if bleu     is not None: m["bleus"].append(bleu)
                if meteor   is not None: m["meteors"].append(meteor)
                if wer      is not None: m["wers"].append(wer)

        def avg(lst): return sum(lst) / len(lst) if lst else 0.0

        model_stats = [
            schemas.LangfuseModelStat(
                whisper=v["whisper"], llm=v["llm"], prompt_version=v["prompt_version"],
                count=v["count"],
                avg_total_ms=round(avg(v["totals"]), 1),
                avg_stt_ms=round(avg(v["stts"]), 1),
                avg_llm_ms=round(avg(v["llms_lat"]), 1),
                avg_bleu=round(avg(v["bleus"]), 3) if v["bleus"] else None,
                avg_meteor=round(avg(v["meteors"]), 4) if v["meteors"] else None,
                avg_wer=round(avg(v["wers"]), 4) if v["wers"] else None,
            )
            for v in sorted(model_map.values(), key=lambda x: x["count"], reverse=True)
        ]

        # Supplément : coûts/tokens depuis Langfuse (le CSV ne les contient pas
        # car les runs historiques sont d'avant le tracking cost). Les nouveaux
        # runs en live ajoutent leur cost via le pipeline → on les agrège ici.
        lf_costs:  list[float] = []
        lf_tokens: list[float] = []
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            try:
                async with httpx.AsyncClient(timeout=8.0) as lf_client:
                    page = 1
                    while True:
                        r = await lf_client.get(
                            f"{LANGFUSE_HOST}/api/public/scores",
                            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
                            params={"limit": 100, "page": page, "name": "cost_usd"},
                        )
                        if not r.is_success:
                            break
                        batch = r.json().get("data", [])
                        lf_costs.extend([s["value"] for s in batch if s.get("value") is not None])
                        if len(batch) < 100:
                            break
                        page += 1
                    # tokens
                    page = 1
                    while True:
                        r = await lf_client.get(
                            f"{LANGFUSE_HOST}/api/public/scores",
                            auth=(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY),
                            params={"limit": 100, "page": page, "name": "total_tokens"},
                        )
                        if not r.is_success:
                            break
                        batch = r.json().get("data", [])
                        lf_tokens.extend([s["value"] for s in batch if s.get("value") is not None])
                        if len(batch) < 100:
                            break
                        page += 1
            except Exception:
                pass

        return schemas.LangfuseMetricsResponse(
            connected=True,
            total_traces=len(totals),
            avg_total_ms=round(avg(totals), 1),
            avg_stt_ms=round(avg(stts), 1),
            avg_llm_ms=round(avg(llms_lat), 1),
            avg_language_prob=round(avg(langs), 3),
            avg_bleu=round(avg(bleus), 3),
            avg_meteor=round(avg(meteors), 4),
            avg_wer=round(avg(wers), 4),
            avg_cost_usd=round(avg(lf_costs), 6) if lf_costs else 0,
            total_cost_usd=round(sum(lf_costs), 4) if lf_costs else 0,
            avg_tokens=round(avg(lf_tokens), 1) if lf_tokens else 0,
            total_tokens=int(sum(lf_tokens)) if lf_tokens else 0,
            bleu_scores=bleus,
            meteor_scores=meteors,
            wer_scores=wers,
            language_probs=langs,
            latencies_total=totals,
            cost_scores=lf_costs,
            model_stats=model_stats,
        )

    # Fallback vers Langfuse si CSV absent
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
            elif s["name"] == "cost_usd":
                model_map[key].setdefault("costs", []).append(s["value"])
            elif s["name"] == "total_tokens":
                model_map[key].setdefault("tokens", []).append(s["value"])

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
                avg_cost_usd=round(avg(v["costs"]), 6) if v.get("costs") else None,
                avg_tokens=round(avg(v["tokens"]), 1) if v.get("tokens") else None,
            )
            for v in sorted(model_map.values(), key=lambda x: x["count"], reverse=True)
        ]

        cost_list   = by_name.get("cost_usd", [])
        tokens_list = by_name.get("total_tokens", [])

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
            avg_cost_usd=round(avg(cost_list), 6),
            total_cost_usd=round(sum(cost_list), 4),
            avg_tokens=round(avg(tokens_list), 1),
            total_tokens=int(sum(tokens_list)),
            bleu_scores=by_name.get("bleu", []),
            meteor_scores=by_name.get("meteor", []),
            wer_scores=by_name.get("wer", []),
            language_probs=by_name.get("language_prob", []),
            latencies_total=by_name.get("latency_total_ms", []),
            cost_scores=cost_list,
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
    csv_rows = _read_experiments_csv()
    if not csv_rows:
        return schemas.ExperimentsResponse(runs=[], total=0, csv_exists=EXPERIMENTS_CSV.exists())

    runs: list[schemas.ExperimentRun] = []
    for row in csv_rows:
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
            tts_wer          = _f("tts_wer"),
        ))
    return schemas.ExperimentsResponse(runs=runs, total=len(runs), csv_exists=True)


@app.post("/admin/tts")
async def admin_tts(body: dict, _: models.User = Depends(get_admin_user)):
    """Proxy TTS : frontend → gateway → service TTS, renvoie l'audio binaire."""
    text = body.get("text", "").strip()
    lang = body.get("lang", "en")
    if not text:
        raise HTTPException(status_code=400, detail="text requis")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{TTS_URL}/synthesize", json={"text": text, "lang": lang})
            if not r.is_success:
                raise HTTPException(status_code=r.status_code, detail=r.text[:200])
            return StreamingResponse(
                iter([r.content]),
                media_type=r.headers.get("content-type", "audio/wav"),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS erreur : {exc}")


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


@app.get("/admin/mlflow/summary")
async def admin_mlflow_summary(_: models.User = Depends(get_admin_user)):
    """Récupère les expériences + modèles registry MLflow via son API REST."""
    summary = {
        "connected":   False,
        "url":         "http://localhost:5050",
        "experiments": [],
        "models":      [],
        "total_runs":  0,
        "error":       None,
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            # Liste des expériences
            r = await client.post(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
                                  json={"max_results": 100})
            if not r.is_success:
                summary["error"] = f"MLflow HTTP {r.status_code}"
                return summary
            experiments = r.json().get("experiments", [])

            # Pour chaque expérience, compter les runs
            total_runs = 0
            exp_list = []
            for exp in experiments:
                runs_r = await client.post(
                    f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                    json={"experiment_ids": [exp["experiment_id"]], "max_results": 1000},
                )
                runs_count = len(runs_r.json().get("runs", [])) if runs_r.is_success else 0
                total_runs += runs_count
                exp_list.append({
                    "id":    exp["experiment_id"],
                    "name":  exp["name"],
                    "runs":  runs_count,
                })

            # Models registry
            m = await client.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search",
                params={"max_results": 100},
            )
            models_list = []
            if m.is_success:
                for rm in m.json().get("registered_models", []):
                    tags = {t["key"]: t["value"] for t in rm.get("tags", [])}
                    models_list.append({
                        "name":               rm["name"],
                        "description":        rm.get("description", ""),
                        "production_version": tags.get("production_version", "—"),
                        "provider":           tags.get("provider", ""),
                        "type":               tags.get("type", ""),
                    })

            summary.update({
                "connected":   True,
                "experiments": exp_list,
                "models":      models_list,
                "total_runs":  total_runs,
            })

    except Exception as exc:
        summary["error"] = str(exc)
    return summary


@app.get("/admin/airflow/summary")
async def admin_airflow_summary(_: models.User = Depends(get_admin_user)):
    """Récupère la liste des DAGs Airflow + leur dernier statut."""
    summary = {
        "connected": False,
        "url":       "http://localhost:8080",
        "dags":      [],
        "error":     None,
    }
    try:
        async with httpx.AsyncClient(timeout=6.0, auth=(AIRFLOW_USER, AIRFLOW_PWD)) as client:
            # Liste tous les DAGs
            r = await client.get(f"{AIRFLOW_URL}/api/v1/dags")
            if not r.is_success:
                summary["error"] = f"Airflow HTTP {r.status_code}"
                return summary
            dags = r.json().get("dags", [])

            # Pour chaque DAG, prend le dernier run
            dag_list = []
            for d in dags:
                dag_id = d["dag_id"]
                runs_r = await client.get(
                    f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns",
                    params={"limit": 1, "order_by": "-start_date"},
                )
                last_run = None
                if runs_r.is_success:
                    runs = runs_r.json().get("dag_runs", [])
                    if runs:
                        last_run = {
                            "state":      runs[0].get("state"),
                            "start_date": runs[0].get("start_date"),
                            "end_date":   runs[0].get("end_date"),
                        }
                dag_list.append({
                    "dag_id":        dag_id,
                    "description":   d.get("description") or "",
                    "schedule":      d.get("schedule_interval", {}).get("value")
                                     if isinstance(d.get("schedule_interval"), dict)
                                     else d.get("schedule_interval"),
                    "is_paused":     d.get("is_paused", False),
                    "tags":          [t["name"] for t in d.get("tags", [])],
                    "last_run":      last_run,
                })
            summary.update({"connected": True, "dags": dag_list})
    except Exception as exc:
        summary["error"] = str(exc)
    return summary
