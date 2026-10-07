"""
Stripe billing endpoints — Checkout + Customer Portal + Webhooks.

Approche : abonnement mensuel ou annuel a une seule offre "Pro". Le tier
"free" est le defaut (pas de facturation). Le tier "pro" ouvre les quotas
etendus et les features premium.

Flow user :
1. User connecte va sur /tarifs, clique "S'abonner" -> POST /billing/checkout
2. Gateway cree une Checkout Session Stripe, retourne l'url_hosted
3. Frontend redirige vers l'url Stripe -> user paye
4. Stripe redirige vers success_url (/tarifs/succes) ET envoie un webhook
5. Le webhook update le status User dans notre DB
6. User peut gerer son abonnement via /billing/portal (portail Stripe hosted)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from gateway import models, credits as _credits
from gateway.database import get_db


# ── Config ───────────────────────────────────────────────────────────────────
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Token partage entre services internes (Pipeline <-> Gateway) pour les appels
# machine-to-machine qui ne portent pas de JWT user (ex: /billing/consume
# apres un pipeline/process reussi). Doit matcher la var cote Pipeline.
SERVICE_TOKEN = os.getenv("GATEWAY_SERVICE_TOKEN", "")

# Prices IDs a creer dans le dashboard Stripe puis coller dans .env
STRIPE_PRICE_PRO_MONTHLY = os.getenv("STRIPE_PRICE_PRO_MONTHLY", "")
STRIPE_PRICE_PRO_YEARLY  = os.getenv("STRIPE_PRICE_PRO_YEARLY", "")

# URL du frontend (utilise pour construire les success_url / cancel_url)
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://traduction-audio.fr")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


router = APIRouter(prefix="/billing", tags=["billing"])


# ── Helpers ──────────────────────────────────────────────────────────────────
def _require_stripe() -> None:
    if not stripe.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing indisponible : STRIPE_SECRET_KEY non configure.",
        )


def _ensure_customer(user: models.User, db: Session) -> str:
    """Retourne le stripe_customer_id pour cet user (le cree si absent)."""
    if user.stripe_customer_id:
        return user.stripe_customer_id
    customer = stripe.Customer.create(
        email=user.email,
        metadata={"user_id": str(user.id)},
    )
    user.stripe_customer_id = customer.id
    db.commit()
    return customer.id


def _sync_subscription_state(user: models.User, sub: dict, db: Session) -> None:
    """Met a jour le tier / status / periode fin depuis un objet Subscription Stripe."""
    status_ = sub.get("status")
    period_end_ts = sub.get("current_period_end")
    period_end = datetime.fromtimestamp(period_end_ts, tz=timezone.utc) if period_end_ts else None
    is_active = status_ in ("active", "trialing")
    user.subscription_tier   = "pro" if is_active else "free"
    user.subscription_status = status_
    user.current_period_end  = period_end
    db.commit()


# ── Schemas ──────────────────────────────────────────────────────────────────
class CheckoutRequest(BaseModel):
    plan: str  # "monthly" | "yearly"


class CheckoutResponse(BaseModel):
    url: str


class PortalResponse(BaseModel):
    url: str


class SubscriptionResponse(BaseModel):
    tier:               str
    status:             str | None
    current_period_end: str | None
    stripe_customer_id: str | None


# ── Endpoints ────────────────────────────────────────────────────────────────

def register_routes(app, get_current_user):
    """Enregistre les routes dans l'app FastAPI parente.

    `get_current_user` est injecte depuis main.py pour eviter les imports
    circulaires (main importe billing pour les routes, billing a besoin de
    l'auth de main).
    """

    @router.post("/checkout", response_model=CheckoutResponse)
    def create_checkout_session(
        req: CheckoutRequest,
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Cree une Checkout Session Stripe, retourne l'URL hosted."""
        _require_stripe()

        price_id = STRIPE_PRICE_PRO_MONTHLY if req.plan == "monthly" else STRIPE_PRICE_PRO_YEARLY
        if not price_id:
            raise HTTPException(status_code=503, detail=f"Prix '{req.plan}' non configure cote serveur.")

        customer_id = _ensure_customer(current_user, db)

        session = stripe.checkout.Session.create(
            # ── sample_only (preserves : valeurs reelles de l'app) ─────────
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{APP_BASE_URL}/tarifs/succes?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP_BASE_URL}/tarifs/annule",
            metadata={"user_id": str(current_user.id)},
            # ── fixed_by_ui (Stripe Checkout Studio) ───────────────────────
            ui_mode="hosted_page",
            billing_address_collection="auto",
            phone_number_collection={"enabled": False},
            automatic_tax={"enabled": False},
            allow_promotion_codes=False,
            payment_method_collection="always",
            submit_type="auto",
            integration_identifier="hosted_web_0001",
            origin_context="web",
            # Desactive Managed Payments pour ce request. L'editeur est en
            # franchise de TVA (art. 293 B CGI) donc Stripe ne doit PAS
            # collecter automatiquement de TVA. Managed Payments etant active
            # par defaut sur le compte, on le desactive explicitement ici pour
            # que automatic_tax={enabled: False} soit accepte.
            managed_payments={"enabled": False},
        )
        return CheckoutResponse(url=session.url)

    @router.post("/portal", response_model=PortalResponse)
    def create_portal_session(
        current_user: models.User = Depends(get_current_user),
    ):
        """Genere une URL vers le portail client Stripe (gestion abonnement)."""
        _require_stripe()
        if not current_user.stripe_customer_id:
            raise HTTPException(status_code=400, detail="Aucun abonnement Stripe pour cet utilisateur.")
        portal = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=f"{APP_BASE_URL}/tarifs",
        )
        return PortalResponse(url=portal.url)

    @router.get("/subscription", response_model=SubscriptionResponse)
    def get_subscription(
        current_user: models.User = Depends(get_current_user),
    ):
        """Retourne l'etat courant de l'abonnement (miroir cache en DB)."""
        return SubscriptionResponse(
            tier=current_user.subscription_tier or "free",
            status=current_user.subscription_status,
            current_period_end=(
                current_user.current_period_end.isoformat()
                if current_user.current_period_end else None
            ),
            stripe_customer_id=current_user.stripe_customer_id,
        )

    @router.post("/webhook")
    async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
        """Endpoint webhook Stripe (doit etre expose publiquement sans auth JWT).

        Configure dans Stripe Dashboard -> Developpeurs -> Webhooks avec l'URL
        https://traduction-audio.fr/api/billing/webhook et les events :
          - customer.subscription.created
          - customer.subscription.updated
          - customer.subscription.deleted
          - invoice.paid
          - invoice.payment_failed
          - checkout.session.completed
        """
        _require_stripe()
        payload = await request.body()
        sig = request.headers.get("stripe-signature", "")

        try:
            event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
        except (ValueError, stripe.SignatureVerificationError) as exc:
            raise HTTPException(status_code=400, detail=f"Webhook invalide : {exc}")

        event_type = event["type"]
        data = event["data"]["object"]

        # Retrouver l'user via metadata ou customer_id
        user = None
        customer_id = data.get("customer") or data.get("customer_id")
        if customer_id:
            user = db.query(models.User).filter(models.User.stripe_customer_id == customer_id).first()
        if not user:
            user_id = (data.get("metadata") or {}).get("user_id")
            if user_id:
                user = db.query(models.User).filter(models.User.id == int(user_id)).first()

        if not user:
            # Event non lie a un user connu — on log et ignore pour eviter les erreurs
            return JSONResponse({"received": True, "warning": "user_not_found", "type": event_type})

        # Sync selon le type d'event
        if event_type in ("customer.subscription.created", "customer.subscription.updated"):
            _sync_subscription_state(user, data, db)
            # Allouer / mettre a jour les credits selon le nouveau tier
            tier = "pro" if user.subscription_tier == "pro" else "free"
            period_end = user.current_period_end
            try:
                _credits.apply_tier(db, user.id, tier, period_resets_at=period_end)
            except Exception as e:
                print(f"[billing] credits.apply_tier error for user {user.id}: {e}", flush=True)
        elif event_type == "customer.subscription.deleted":
            user.subscription_tier   = "free"
            user.subscription_status = "canceled"
            db.commit()
            # Downgrade free : on garde le solde existant (user peut consommer le reliquat)
            try:
                _credits.apply_tier(db, user.id, "free")
            except Exception as e:
                print(f"[billing] credits.apply_tier(free) error for user {user.id}: {e}", flush=True)
        elif event_type == "checkout.session.completed":
            # Confirme le customer_id (utile si create() n'a pas ete appele avant)
            if data.get("customer") and not user.stripe_customer_id:
                user.stripe_customer_id = data["customer"]
                db.commit()

        return JSONResponse({"received": True, "type": event_type})

    # ── Endpoints credits (user) ─────────────────────────────────────────

    @router.get("/credits")
    def get_credits(
        current_user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        """Retourne le solde de credits + l'allocation mensuelle."""
        # Refill lazy si la periode est terminee
        _credits.refill_if_due(db, current_user.id, current_user.subscription_tier or "free")
        row = _credits._get_or_create(db, current_user.id)
        return {
            "processing": {
                "balance":       row.processing_balance,
                "monthly":       row.processing_monthly,
                "carryover_max": row.processing_carryover_max,
            },
            "live": {
                "balance":       row.live_balance,
                "monthly":       row.live_monthly,
                "carryover_max": row.live_carryover_max,
            },
            "tier":                  current_user.subscription_tier or "free",
            "trial_credits_granted": row.trial_credits_granted,
            "period_resets_at":      row.period_resets_at.isoformat() if row.period_resets_at else None,
        }

    # ── Endpoints service-to-service (Pipeline → Gateway) ────────────────

    def _check_service_token(x_service_token: str | None) -> None:
        if not SERVICE_TOKEN or x_service_token != SERVICE_TOKEN:
            raise HTTPException(status_code=401, detail="Service token invalide")

    class PreAuthorizeRequest(BaseModel):
        user_id:   int
        kind:      str      # "processing" | "live"
        amount:    int

    @router.post("/pre-authorize")
    def pre_authorize(
        req: PreAuthorizeRequest,
        request: Request,
        db: Session = Depends(get_db),
    ):
        """Verifie que l'user a assez de credits avant lancement du job.

        Appele par le Pipeline AVANT de taper OpenAI pour eviter de bruler du
        quota API alors que l'user n'a plus de credits.

        Pas de consommation reelle ici — c'est juste un check.
        """
        _check_service_token(request.headers.get("x-service-token"))
        if req.kind not in ("processing", "live"):
            raise HTTPException(status_code=400, detail="kind doit etre 'processing' ou 'live'")
        user = db.query(models.User).filter(models.User.id == req.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User introuvable")
        _credits.refill_if_due(db, user.id, user.subscription_tier or "free")
        ok = _credits.has_enough(db, user.id, req.kind, req.amount)  # type: ignore[arg-type]
        row = _credits._get_or_create(db, user.id)
        balance = row.processing_balance if req.kind == "processing" else row.live_balance
        return {
            "ok":       ok,
            "balance":  balance,
            "needed":   req.amount,
            "tier":     user.subscription_tier or "free",
        }

    class ConsumeRequest(BaseModel):
        user_id:   int
        kind:      str
        amount:    int
        reason:    str
        meta:      dict | None = None

    @router.post("/consume")
    def consume_credits(
        req: ConsumeRequest,
        request: Request,
        db: Session = Depends(get_db),
    ):
        """Decremente le solde apres un job reussi."""
        _check_service_token(request.headers.get("x-service-token"))
        if req.kind not in ("processing", "live"):
            raise HTTPException(status_code=400, detail="kind doit etre 'processing' ou 'live'")
        try:
            row = _credits.consume(db, req.user_id, req.kind, req.amount, req.reason, req.meta)  # type: ignore[arg-type]
        except ValueError as e:
            raise HTTPException(status_code=402, detail=str(e))
        balance = row.processing_balance if req.kind == "processing" else row.live_balance
        return {"ok": True, "balance": balance}

    class RefundRequest(BaseModel):
        user_id:   int
        kind:      str
        amount:    int
        reason:    str

    @router.post("/refund")
    def refund_credits(
        req: RefundRequest,
        request: Request,
        db: Session = Depends(get_db),
    ):
        """Rembourse des credits (ex : job pipeline echoue cote notre serveur)."""
        _check_service_token(request.headers.get("x-service-token"))
        if req.kind not in ("processing", "live"):
            raise HTTPException(status_code=400, detail="kind doit etre 'processing' ou 'live'")
        row = _credits.refund(db, req.user_id, req.kind, req.amount, req.reason)  # type: ignore[arg-type]
        balance = row.processing_balance if req.kind == "processing" else row.live_balance
        return {"ok": True, "balance": balance}

    app.include_router(router)
