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
import uuid
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Literal

from sqlalchemy import func
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

# Seuil (processing credits) sous lequel on envoie un email d'alerte.
# L'alerte n'est envoyee qu'une fois par periode (reset au refill mensuel).
LOW_CREDITS_THRESHOLD = 20

# Duree au-dela de laquelle une reservation non reglee cesse de bloquer le solde
# (pipeline redemarre en plein job). Large : un audio d'1 h prend ~15 min.
HOLD_TTL = timedelta(hours=2)

# Serialise reserve / settle / release : la gateway est mono-worker, mais ses
# endpoints sync tournent dans un pool de threads. Sans ce verrou, deux
# reservations simultanees liraient le meme solde disponible.
_hold_lock = Lock()

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
def _aware(dt: datetime | None) -> datetime | None:
    """SQLite relit les dates sans fuseau : on les considere en UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


def allocation_due(db: Session, user_id: int, newly_subscribed: bool) -> bool:
    """Faut-il crediter l'allocation mensuelle maintenant ?

    Stripe envoie un evenement « abonnement mis a jour » pour bien d'autres
    raisons qu'un renouvellement (resiliation demandee puis annulee, changement
    de carte...). Crediter a chacun offrirait des credits a volonte : on ne
    credite qu'a la souscription, puis une fois la periode de credits ecoulee.
    """
    if newly_subscribed:
        return True
    resets_at = _aware(_get_or_create(db, user_id).period_resets_at)
    # 2 jours de marge : l'evenement de renouvellement peut preceder l'echeance
    return resets_at is None or datetime.now(timezone.utc) >= resets_at - timedelta(days=2)


def apply_tier(
    db: Session, user_id: int, tier: str,
    period_resets_at: datetime | None = None, allocate: bool = True,
) -> models.UserCredits:
    """Configure les quotas mensuels et, si `allocate`, credite la periode en cours.

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
        if allocate:
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

    # Les credits sont mensuels quelle que soit la formule. Mensuel : prochain
    # refill a l'echeance Stripe. Annuel : l'echeance est dans un an, le refill
    # suit donc un pas de 30 jours (refill_if_due).
    now = datetime.now(timezone.utc)
    period_end = _aware(period_resets_at)
    if period_end and period_end <= now + timedelta(days=32):
        row.period_resets_at = period_end
    else:
        row.period_resets_at = now + timedelta(days=30)

    # Reset du flag d'alerte quota : nouvelle periode = user peut retre alerte.
    row.low_credits_alerted = False


def refill_if_due(db: Session, user_id: int, tier: str) -> models.UserCredits:
    """Refill lazy appele au prochain acces si la periode est terminee."""
    row = _get_or_create(db, user_id)
    resets_at = row.period_resets_at
    if resets_at is not None and resets_at.tzinfo is None:
        resets_at = resets_at.replace(tzinfo=timezone.utc)  # SQLite relit les dates sans fuseau
    if tier in ("pro", "business") and resets_at and resets_at <= datetime.now(timezone.utc):
        allocate_monthly(db, row, tier)
        db.commit()
    return row


# ── Verification + consommation ──────────────────────────────────────────────
def _balance(row: models.UserCredits, kind: Kind) -> int:
    return row.processing_balance if kind == "processing" else row.live_balance


def held(db: Session, user_id: int, kind: Kind) -> int:
    """Total des reservations en cours (hors reservations expirees)."""
    cutoff = datetime.now(timezone.utc) - HOLD_TTL
    return db.query(func.coalesce(func.sum(models.CreditHold.amount), 0)).filter(
        models.CreditHold.user_id == user_id,
        models.CreditHold.kind == kind,
        models.CreditHold.created_at >= cutoff,
    ).scalar()


def available(db: Session, user_id: int, kind: Kind) -> int:
    """Solde moins les reservations en cours : ce qu'un nouveau job peut engager."""
    return _balance(_get_or_create(db, user_id), kind) - held(db, user_id, kind)


def has_enough(db: Session, user_id: int, kind: Kind, amount: int) -> bool:
    """True si l'user a assez de credits disponibles pour cette operation."""
    return available(db, user_id, kind) >= amount


# ── Reservation (jobs pipeline) ──────────────────────────────────────────────
def reserve(db: Session, user_id: int, kind: Kind, amount: int) -> str | None:
    """Reserve `amount` credits pour un job. Retourne l'id de reservation, ou
    None si le solde disponible ne suffit pas.

    Le solde n'est debite qu'au settle() : deux jobs lances en parallele ne
    peuvent pas engager deux fois les memes credits.
    """
    with _hold_lock:
        cutoff = datetime.now(timezone.utc) - HOLD_TTL
        db.query(models.CreditHold).filter(models.CreditHold.created_at < cutoff).delete()
        if available(db, user_id, kind) < amount:
            db.commit()
            return None
        hold = models.CreditHold(
            id=str(uuid.uuid4()), user_id=user_id, kind=kind, amount=amount,
            created_at=datetime.now(timezone.utc),
        )
        db.add(hold)
        db.commit()
        return hold.id


def settle(
    db: Session,
    hold_id: str,
    amount: int,
    reason: str,
    meta: dict | None = None,
) -> tuple[int, int] | None:
    """Remplace la reservation par le debit reel. Retourne (debite, solde).

    Le debit reel peut differer de l'estimation reservee. S'il depasse le solde,
    on debite ce qui reste (pas de dette, comme consume_seconds).
    Reservation inconnue (deja reglee ou annulee) : None, sans rien debiter —
    un second appel pour le meme job est donc sans effet.
    """
    with _hold_lock:
        hold = db.get(models.CreditHold, hold_id)
        if hold is None:
            return None
        user_id, kind = hold.user_id, hold.kind
        row = _get_or_create(db, user_id)
        charged = min(max(amount, 0), _balance(row, kind))
        db.delete(hold)
        if charged > 0:
            row = consume(db, user_id, kind, charged, reason, meta)  # meme commit que la suppression
        else:
            db.commit()
        return charged, _balance(row, kind)


def release(db: Session, hold_id: str) -> bool:
    """Annule une reservation sans rien debiter (job echoue). Idempotent."""
    with _hold_lock:
        deleted = db.query(models.CreditHold).filter(models.CreditHold.id == hold_id).delete()
        db.commit()
        return bool(deleted)


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

    # Alerte quota faible (uniquement sur le compteur processing, qui est le
    # principal). Envoye une seule fois par periode : le flag est remis a 0
    # lors du prochain refill mensuel.
    if kind == "processing" and new_balance < LOW_CREDITS_THRESHOLD and not row.low_credits_alerted:
        row.low_credits_alerted = True
        try:
            from gateway import emails
            user = db.query(models.User).filter(models.User.id == user_id).first()
            if user:
                emails.send_low_credits_alert(user.email, new_balance)
        except Exception as e:
            print(f"[credits] low-credits email error for user {user_id}: {e}", flush=True)

    db.commit()
    return row


def consume_seconds(
    db: Session,
    user_id: int,
    seconds: float,
    rate_per_min: float,
    reason: str,
    usage: dict | None = None,
) -> tuple[int, int]:
    """Debite une duree au prorata, en milliemes de credit.

    Une reunion est transcrite par tranches de 30 s : un debit entier par tranche
    doublerait le prix. La fraction restante est reportee sur l'appel suivant.
    Solde insuffisant : on debite ce qui reste et on abandonne la fraction (pas
    de dette). Retourne (credits debites, solde).
    """
    row = _get_or_create(db, user_id)
    milli = row.processing_remainder_milli + round(max(seconds, 0.0) * rate_per_min * 1000 / 60)
    due, remainder = divmod(milli, 1000)
    charged = min(due, row.processing_balance)
    row.processing_remainder_milli = remainder if charged == due else 0
    # seconds : volume affiche au client ; usage : backend utilise, pour le cout reel
    meta = {**(usage or {}), "seconds": round(max(seconds, 0.0), 1)}
    if charged > 0:
        row = consume(db, user_id, "processing", charged, reason, meta)  # commit + alerte solde faible
    else:
        _record_tx(db, user_id, "processing", "consume", amount=0,
                   balance_after=row.processing_balance, reason=reason, meta=meta)
        db.commit()
    return charged, row.processing_balance


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


def revoke_subscription_credits(db: Session, user_id: int, reason: str) -> int:
    """Retire les credits apportes par l'abonnement (remboursement apres
    retractation). On ne retire pas plus que le solde : pas de dette."""
    row = _get_or_create(db, user_id)
    revoked = min(row.processing_balance, row.processing_monthly)
    if revoked > 0:
        row.processing_balance -= revoked
        _record_tx(db, user_id, "processing", "withdrawal", amount=-revoked,
                   balance_after=row.processing_balance, reason=reason)
    db.commit()
    return revoked


# ── Pondération business ─────────────────────────────────────────────────────
def credits_for_pipeline(
    duration_seconds: float,
    with_tts: bool = True,
    with_diarize: bool = False,
) -> int:
    """Credits consommes pour un job pipeline standard.

    - transcription seule           : 1 credit/min
    - avec TTS (traduction + voix)  : 2 credits/min
    - avec diarisation              : +0.5 credit/min (sur le total)
    Arrondi a l'entier superieur (minimum 1 credit par job).
    """
    minutes = max(duration_seconds / 60, 0)
    rate = (2 if with_tts else 1) + (0.5 if with_diarize else 0)
    return max(1, int(minutes * rate + 0.999))  # ceil
