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

from gateway import models
from gateway.database import get_db


# ── Config ───────────────────────────────────────────────────────────────────
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

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
            ui_mode="hosted",
            billing_address_collection="auto",
            phone_number_collection={"enabled": False},
            automatic_tax={"enabled": False},
            allow_promotion_codes=False,
            payment_method_collection="always",
            submit_type="auto",
            integration_identifier="hosted_web_0001",
            origin_context="web",
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
        elif event_type == "customer.subscription.deleted":
            user.subscription_tier   = "free"
            user.subscription_status = "canceled"
            db.commit()
        elif event_type == "checkout.session.completed":
            # Confirme le customer_id (utile si create() n'a pas ete appele avant)
            if data.get("customer") and not user.stripe_customer_id:
                user.stripe_customer_id = data["customer"]
                db.commit()

        return JSONResponse({"received": True, "type": event_type})

    app.include_router(router)
