"""
Moteur de credits — allocation, consommation, verification de quota.

Deux compteurs separes :
- processing_credits : pour /process (Pipeline STT + LLM + TTS)
- live_credits       : pour Realtime API (couteux, isole)

Pondération de consommation (voir plan tarifaire) :
- Transcription seule        : 1 credit/min
- Traduction texte           : 1 credit/min
- Traduction + voix (defaut) : 2 credits/min
- Live                       : tarification separee

Tiers :
- free      : 10 credits one-time a l'inscription (trial), aucun refill
- pro       : 200 credits/mois processing, report max 400
- business  : 600 credits/mois processing, report max 1200
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Literal

from sqlalchemy.orm import Session

from gateway import models


# ── Config des tiers (en synch avec le plan tarifaire valide) ────────────────
TIER_CONFIG = {
    "free": {
        "processing_monthly":       0,
        "live_monthly":             0,
        "processing_carryover_max": 0,
        "live_carryover_max":       0,
    },
    "pro": {
        "processing_monthly":       200,
        "live_monthly":             0,     # Live hors forfait au lancement
        "processing_carryover_max": 400,
        "live_carryover_max":       0,
    },
    "business": {
        "processing_monthly":       600,
        "live_monthly":             0,
        "processing_carryover_max": 1200,
        "live_carryover_max":       0,
    },
}

TRIAL_CREDITS = 10  # one-time a l'inscription, pour tous les users

Kind = Literal["processing", "live"]


# ── Helpers DB ───────────────────────────────────────────────────────────────
def _record_tx(
    db: Session,
    user_id: int,
    kind: Kind,
    op: str,
    amount: int,
    balance_after: int,
    reason: str | None = None,
    meta: dict | None = None,
) -> None:
    """Ajoute une ligne dans l'historique."""
    tx = models.CreditTransaction(
        user_id=user_id,
        kind=kind,
        op=op,
        amount=amount,
        balance_after=balance_after,
        reason=reason,
        meta=json.dumps(meta) if meta else None,
    )
    db.add(tx)


def _get_or_create(db: Session, user_id: int) -> models.UserCredits:
    """Lazy create du row UserCredits a la premiere consultation."""
    row = db.query(models.UserCredits).filter_by(user_id=user_id).first()
    if row:
        return row
    row = models.UserCredits(user_id=user_id)
    db.add(row)
    db.flush()
    return row


# ── Allocation initiale (trial) ──────────────────────────────────────────────
def grant_trial(db: Session, user_id: int) -> models.UserCredits:
    """Alloue les 10 credits one-time a l'inscription. Idempotent."""
    row = _get_or_create(db, user_id)
    if row.trial_credits_granted:
        return row  # deja fait
    row.processing_balance += TRIAL_CREDITS
    row.trial_credits_granted = True
    _record_tx(
        db, user_id, "processing", "trial_grant",
        amount=TRIAL_CREDITS,
        balance_after=row.processing_balance,
        reason="signup_trial",
    )
    db.commit()
    return row


# ── Allocation / refill mensuel ──────────────────────────────────────────────
def apply_tier(db: Session, user_id: int, tier: str, period_resets_at: datetime | None = None) -> models.UserCredits:
    """Configure les quotas mensuels + alloue tout de suite les credits de la periode.

    Appele depuis le webhook Stripe sur customer.subscription.created ou updated.
    Si tier passe de pro -> business, on remet aussi a jour le monthly immediatement.
    """
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["free"])
    row = _get_or_create(db, user_id)

    row.processing_monthly       = cfg["processing_monthly"]
    row.live_monthly             = cfg["live_monthly"]
    row.processing_carryover_max = cfg["processing_carryover_max"]
    row.live_carryover_max       = cfg["live_carryover_max"]

    if tier in ("pro", "business"):
        # Alloue immediatement les credits de la periode en cours
        allocate_monthly(db, row, tier, period_resets_at=period_resets_at)
    else:
        # free : pas de monthly, mais garde le reliquat et le trial
        row.period_resets_at = None

    db.commit()
    return row


def allocate_monthly(
    db: Session,
    row: models.UserCredits,
    tier: str,
    period_resets_at: datetime | None = None,
) -> None:
    """Ajoute l'allocation mensuelle en appliquant le plafond de report.

    Formule : new_balance = min(current + monthly, carryover_max + monthly)
    (le user peut accumuler jusqu'au carryover_max avant la nouvelle allocation).
    """
    cfg = TIER_CONFIG[tier]

    for kind_name, monthly, cap, balance_attr in [
        ("processing", cfg["processing_monthly"], cfg["processing_carryover_max"], "processing_balance"),
        ("live",       cfg["live_monthly"],       cfg["live_carryover_max"],       "live_balance"),
    ]:
        if monthly <= 0:
            continue
        current = getattr(row, balance_attr)
        new_balance = min(current + monthly, cap + monthly)
        granted = new_balance - current
        if granted > 0:
            setattr(row, balance_attr, new_balance)
            _record_tx(
                db, row.user_id, kind_name, "monthly_refill",  # type: ignore[arg-type]
                amount=granted,
                balance_after=new_balance,
                reason=f"tier={tier}",
            )

    # Prochain refill dans ~30 jours (sera resynchronise par le webhook Stripe)
    if period_resets_at:
        row.period_resets_at = period_resets_at
    else:
        row.period_resets_at = datetime.now(timezone.utc) + timedelta(days=30)


def refill_if_due(db: Session, user_id: int, tier: str) -> models.UserCredits:
    """Refill lazy appele au prochain acces si la periode est terminee."""
    row = _get_or_create(db, user_id)
    if tier in ("pro", "business") and row.period_resets_at and row.period_resets_at <= datetime.now(timezone.utc):
        allocate_monthly(db, row, tier)
        db.commit()
    return row


# ── Verification + consommation ──────────────────────────────────────────────
def has_enough(db: Session, user_id: int, kind: Kind, amount: int) -> bool:
    """True si l'user a assez de credits pour cette operation."""
    row = _get_or_create(db, user_id)
    balance = row.processing_balance if kind == "processing" else row.live_balance
    return balance >= amount


def consume(
    db: Session,
    user_id: int,
    kind: Kind,
    amount: int,
    reason: str,
    meta: dict | None = None,
) -> models.UserCredits:
    """Decremente les credits. Leve ValueError si insuffisant.

    Idempotence : a l'appelant de garantir qu'il ne consomme pas deux fois pour
    le meme job (par ex en passant un job_id dans meta et en verifiant avant).
    """
    row = _get_or_create(db, user_id)
    balance_attr = "processing_balance" if kind == "processing" else "live_balance"
    current = getattr(row, balance_attr)
    if current < amount:
        raise ValueError(f"Credits insuffisants ({kind}) : {current} dispo, {amount} requis")

    new_balance = current - amount
    setattr(row, balance_attr, new_balance)
    _record_tx(
        db, user_id, kind, "consume",
        amount=-amount,
        balance_after=new_balance,
        reason=reason,
        meta=meta,
    )
    db.commit()
    return row


def refund(
    db: Session,
    user_id: int,
    kind: Kind,
    amount: int,
    reason: str,
) -> models.UserCredits:
    """Rembourse des credits (ex : job pipeline echoue cote serveur apres consume).

    Note : le refund peut depasser le carryover_max — c'est volontaire, un
    user ne doit pas perdre des credits a cause d'une panne cote notre service.
    """
    row = _get_or_create(db, user_id)
    balance_attr = "processing_balance" if kind == "processing" else "live_balance"
    current = getattr(row, balance_attr)
    new_balance = current + amount
    setattr(row, balance_attr, new_balance)
    _record_tx(
        db, user_id, kind, "refund",
        amount=amount,
        balance_after=new_balance,
        reason=reason,
    )
    db.commit()
    return row


# ── Pondération business ─────────────────────────────────────────────────────
def credits_for_pipeline(duration_seconds: float, with_tts: bool = True) -> int:
    """Credits consommes pour un job pipeline standard.

    - avec TTS (traduction + voix) : 2 credits par minute d'audio
    - sans TTS (transcription + traduction texte seul) : 1 credit par minute
    Arrondi a l'entier superieur (minimum 1 credit par job).
    """
    minutes = max(duration_seconds / 60, 0)
    rate = 2 if with_tts else 1
    return max(1, int(minutes * rate + 0.999))  # ceil
