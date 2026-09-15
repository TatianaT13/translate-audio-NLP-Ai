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
