from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.sql import func

from gateway.database import Base


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active       = Column(Boolean, default=True)
    is_admin        = Column(Boolean, default=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())

    # ── Stripe billing ────────────────────────────────────────────────────
    # stripe_customer_id : id Customer Stripe (cus_...) — cree a la premiere
    # souscription. Reste attache a l'user meme apres resiliation pour permettre
    # la re-souscription sans creer de doublon dans Stripe.
    stripe_customer_id  = Column(String, unique=True, index=True, nullable=True)

    # subscription_tier : "free" | "pro" — determine les quotas et features
    subscription_tier   = Column(String, default="free", nullable=False)

    # subscription_status : miroir du statut Stripe pour eviter un round-trip
    # a chaque check. "active", "trialing", "past_due", "canceled", "incomplete",
    # "incomplete_expired", "unpaid". None = pas d'abonnement historique.
    subscription_status = Column(String, nullable=True)

    # current_period_end : date de fin de la periode facturee courante.
    # Utilise pour l'affichage cote frontend et pour laisser l'acces jusqu'a
    # cette date apres une resiliation.
    current_period_end  = Column(DateTime(timezone=True), nullable=True)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked    = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used       = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserCredits(Base):
    """Solde de credits par user, separe processing (Pipeline) et live (Realtime).

    Deux compteurs independants pour eviter qu'un user Pro consomme ses credits
    processing en Live (qui coute 10-20x plus) → pricing du plan tarifaire.

    Refill lazy : au prochain acces, si current_period_end est passe, on alloue
    les credits de la nouvelle periode en appliquant le report plafonne.
    """
    __tablename__ = "user_credits"

    id       = Column(Integer, primary_key=True, index=True)
    user_id  = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)

    # Solde courant
    processing_balance = Column(Integer, default=0, nullable=False)
    live_balance       = Column(Integer, default=0, nullable=False)

    # Allocation mensuelle (0 pour tier free au-dela du trial)
    processing_monthly = Column(Integer, default=0, nullable=False)
    live_monthly       = Column(Integer, default=0, nullable=False)

    # Report plafonne (ex : 400 pour Pro = 2x le monthly)
    processing_carryover_max = Column(Integer, default=0, nullable=False)
    live_carryover_max       = Column(Integer, default=0, nullable=False)

    # Prochain refill — ignore pour les credits trial one-time
    period_resets_at   = Column(DateTime(timezone=True), nullable=True)

    # Trial one-time (10 credits a l'inscription, non renouvelable)
    trial_credits_granted = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class CreditTransaction(Base):
    """Historique immuable des mouvements de credits (audit + debug).

    kind    : "processing" | "live"
    op      : "allocate" | "consume" | "refund" | "trial_grant" | "monthly_refill"
    amount  : signe (positif = credit, negatif = debit)
    reason  : courte string pour trace (ex : "pipeline_process:5min:audio_translated")
    metadata: JSON string pour contexte supplementaire
    """
    __tablename__ = "credit_transactions"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kind       = Column(String, nullable=False)   # processing | live
    op         = Column(String, nullable=False)   # allocate | consume | refund | trial_grant | monthly_refill
    amount     = Column(Integer, nullable=False)  # signe
    balance_after = Column(Integer, nullable=False)  # solde apres cette tx
    reason     = Column(String, nullable=True)
    meta       = Column(String, nullable=True)    # JSON serialise, optionnel
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
