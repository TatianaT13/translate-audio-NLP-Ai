"""
Historique des traitements d'un utilisateur (traductions, comptes-rendus).

Le navigateur enregistre le resultat d'un traitement termine ; l'utilisateur
le retrouve dans « Mon historique », peut le relire, le telecharger, le supprimer.

Confidentialite :
- le contenu (transcription, traduction, compte-rendu) est chiffre en base ;
  seuls le titre, le type et la date sont en clair, pour afficher la liste ;
- l'audio n'est jamais conserve ;
- chaque element est supprime au bout de HISTORY_RETENTION_DAYS, et avec le compte.

Cle de chiffrement : HISTORY_ENCRYPTION_KEY si definie, sinon une cle generee
au premier demarrage dans le dossier de donnees (history.key). Elle n'est pas
dans la base : une sauvegarde de auth.db seule ne permet pas de lire le contenu.
A conserver a part pour pouvoir relire l'historique apres une restauration.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from gateway import models
from gateway.database import DATA_DIR, get_db

RETENTION_DAYS     = int(os.getenv("HISTORY_RETENTION_DAYS", "90"))
MAX_ITEMS_PER_USER = int(os.getenv("HISTORY_MAX_ITEMS", "200"))
MAX_CONTENT_CHARS  = 600_000   # ~1 h 30 de parole transcrite + traduite, largement

KINDS = ("translation", "meeting")

router = APIRouter(prefix="/history", tags=["history"])


# ── Chiffrement ──────────────────────────────────────────────────────────────
def _load_key() -> bytes:
    key = os.getenv("HISTORY_ENCRYPTION_KEY", "").strip()
    if key:
        return key.encode()
    path = DATA_DIR / "history.key"
    if not path.exists():
        path.write_bytes(Fernet.generate_key())
        os.chmod(path, 0o600)
        print(f"[history] cle de chiffrement generee : {path}", flush=True)
    return path.read_bytes().strip()


_fernet = Fernet(_load_key())


def _encrypt(content: dict) -> str:
    return _fernet.encrypt(json.dumps(content, ensure_ascii=False).encode()).decode()


def _decrypt(payload: str) -> dict:
    try:
        return json.loads(_fernet.decrypt(payload.encode()))
    except (InvalidToken, ValueError):
        # Cle changee ou perdue (restauration sans history.key)
        raise HTTPException(status_code=410, detail="Cet élément n'est plus lisible.")


# ── Schemas ──────────────────────────────────────────────────────────────────
class HistoryIn(BaseModel):
    kind:       str
    title:      str = Field(min_length=1, max_length=200)
    lang:       str | None = Field(default=None, max_length=10)
    duration_s: int | None = Field(default=None, ge=0, le=24 * 3600)
    content:    dict


def _meta(item: models.HistoryItem) -> dict:
    created = item.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)  # SQLite relit les dates sans fuseau
    return {
        "id":         item.id,
        "kind":       item.kind,
        "title":      item.title,
        "lang":       item.lang,
        "duration_s": item.duration_s,
        "created_at": created.isoformat() if created else None,
        "expires_at": (created + timedelta(days=RETENTION_DAYS)).isoformat() if created else None,
    }


def _check(body: HistoryIn) -> str:
    """Valide l'entree et retourne le contenu chiffre."""
    if body.kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"kind doit etre parmi {KINDS}")
    if len(json.dumps(body.content, ensure_ascii=False)) > MAX_CONTENT_CHARS:
        raise HTTPException(status_code=413, detail="Contenu trop volumineux pour l'historique.")
    return _encrypt(body.content)


# ── Acces aux donnees ────────────────────────────────────────────────────────
def purge_expired(db: Session) -> int:
    """Supprime les elements plus anciens que la duree de conservation (tous comptes)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    deleted = db.query(models.HistoryItem).filter(models.HistoryItem.created_at < cutoff).delete()
    db.commit()
    return deleted


def delete_for_user(db: Session, user_id: int) -> int:
    """Supprime tout l'historique d'un compte (sans commit : fait partie de la suppression du compte)."""
    return db.query(models.HistoryItem).filter(models.HistoryItem.user_id == user_id).delete()


def _owned(db: Session, item_id: str, user_id: int) -> models.HistoryItem:
    item = db.query(models.HistoryItem).filter(
        models.HistoryItem.id == item_id,
        models.HistoryItem.user_id == user_id,   # l'element d'un autre compte est introuvable, pas interdit
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Élément introuvable")
    return item


def register_routes(app, get_current_user):
    """Enregistre les routes dans l'app FastAPI parente."""

    @router.get("")
    def list_history(
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Liste des traitements enregistres, du plus recent au plus ancien (sans le contenu)."""
        purge_expired(db)
        items = (
            db.query(models.HistoryItem)
            .filter(models.HistoryItem.user_id == current_user.id)
            .order_by(models.HistoryItem.created_at.desc())
            .limit(MAX_ITEMS_PER_USER)
            .all()
        )
        return {"retention_days": RETENTION_DAYS, "items": [_meta(i) for i in items]}

    @router.post("", status_code=201)
    def create_history(
        body: HistoryIn,
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Enregistre le resultat d'un traitement termine."""
        payload = _check(body)
        purge_expired(db)
        item = models.HistoryItem(
            id=str(uuid.uuid4()), user_id=current_user.id, kind=body.kind, title=body.title.strip(),
            lang=body.lang, duration_s=body.duration_s, payload=payload,
            created_at=datetime.now(timezone.utc),
        )
        db.add(item)
        db.flush()
        # Au-dela du plafond, les plus anciens laissent la place
        overflow = (
            db.query(models.HistoryItem.id)
            .filter(models.HistoryItem.user_id == current_user.id)
            .order_by(models.HistoryItem.created_at.desc())
            .offset(MAX_ITEMS_PER_USER)
            .all()
        )
        if overflow:
            db.query(models.HistoryItem).filter(
                models.HistoryItem.id.in_([row.id for row in overflow])
            ).delete(synchronize_session=False)
        db.commit()
        return _meta(item)

    @router.get("/{item_id}")
    def get_history_item(
        item_id: str,
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Un traitement enregistre, avec son contenu."""
        item = _owned(db, item_id, current_user.id)
        return {**_meta(item), "content": _decrypt(item.payload)}

    @router.put("/{item_id}")
    def update_history_item(
        item_id: str,
        body: HistoryIn,
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Remplace le contenu d'un element (ex : compte-rendu regenere dans un autre style)."""
        payload = _check(body)
        item = _owned(db, item_id, current_user.id)
        item.title, item.lang, item.duration_s, item.payload = body.title.strip(), body.lang, body.duration_s, payload
        db.commit()
        return _meta(item)

    @router.delete("/{item_id}")
    def delete_history_item(
        item_id: str,
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        db.delete(_owned(db, item_id, current_user.id))
        db.commit()
        return {"deleted": 1}

    @router.delete("")
    def clear_history(
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Supprime tout l'historique du compte."""
        deleted = delete_for_user(db, current_user.id)
        db.commit()
        return {"deleted": deleted}

    app.include_router(router)
