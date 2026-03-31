"""
Gateway Service — Port 8004
Auth complète : register / login / logout / refresh / me /
                change-password / forgot-password / reset-password / delete-account
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
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

    access_token          = auth_utils.create_access_token(user.id, user.email)
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
    access_token          = auth_utils.create_access_token(user.id, user.email)
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
