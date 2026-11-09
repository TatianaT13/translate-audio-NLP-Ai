"""Tests unitaires du moteur de credits (gateway.credits).

Couvre :
  - essai gratuit, debit, remboursement, historique
  - reservation / reglement / annulation (jobs pipeline)
  - debit au prorata des reunions (consume_seconds)
  - refill mensuel et plafond de report
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from gateway import credits, models
from gateway.database import Base


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    yield session
    session.close()


@pytest.fixture
def user_id(db) -> int:
    """Un user avec ses 10 credits d'essai."""
    user = models.User(email="client@example.com", hashed_password="x")
    db.add(user)
    db.commit()
    credits.grant_trial(db, user.id)
    return user.id


def _balance(db, user_id: int) -> int:
    return credits._get_or_create(db, user_id).processing_balance


def _txs(db, user_id: int, op: str) -> list[models.CreditTransaction]:
    return db.query(models.CreditTransaction).filter_by(user_id=user_id, op=op).all()


# ── Essai gratuit ────────────────────────────────────────────────────────────

class TestTrial:
    def test_grant_trial_gives_trial_credits(self, db, user_id):
        assert _balance(db, user_id) == credits.TRIAL_CREDITS

    def test_grant_trial_is_idempotent(self, db, user_id):
        credits.grant_trial(db, user_id)
        credits.grant_trial(db, user_id)
        assert _balance(db, user_id) == credits.TRIAL_CREDITS
        assert len(_txs(db, user_id, "trial_grant")) == 1


# ── Debit / remboursement ────────────────────────────────────────────────────

class TestConsumeRefund:
    def test_consume_debits_and_records_transaction(self, db, user_id):
        credits.consume(db, user_id, "processing", 4, "pipeline_process:120s:audio_translated")
        assert _balance(db, user_id) == 6
        (tx,) = _txs(db, user_id, "consume")
        assert (tx.amount, tx.balance_after) == (-4, 6)

    def test_consume_refuses_when_balance_too_low(self, db, user_id):
        with pytest.raises(ValueError):
            credits.consume(db, user_id, "processing", 11, "trop")
        assert _balance(db, user_id) == 10

    def test_refund_restores_credits(self, db, user_id):
        credits.consume(db, user_id, "processing", 4, "job")
        credits.refund(db, user_id, "processing", 4, "job_failed")
        assert _balance(db, user_id) == 10

    def test_processing_and_live_balances_are_separate(self, db, user_id):
        assert credits.has_enough(db, user_id, "processing", 10)
        assert not credits.has_enough(db, user_id, "live", 1)


# ── Reservation (jobs pipeline) ──────────────────────────────────────────────

class TestReservation:
    def test_reserve_blocks_credits_without_debiting(self, db, user_id):
        hold_id = credits.reserve(db, user_id, "processing", 6)
        assert hold_id is not None
        assert _balance(db, user_id) == 10
        assert credits.available(db, user_id, "processing") == 4

    def test_parallel_jobs_cannot_commit_the_same_credits(self, db, user_id):
        """Le scenario a l'origine de la reservation : N envois simultanes avec
        un solde pour un seul."""
        assert credits.reserve(db, user_id, "processing", 8) is not None
        assert credits.reserve(db, user_id, "processing", 8) is None
        assert credits.available(db, user_id, "processing") == 2

    def test_reserve_refuses_more_than_balance(self, db, user_id):
        assert credits.reserve(db, user_id, "processing", 11) is None

    def test_has_enough_counts_holds(self, db, user_id):
        credits.reserve(db, user_id, "processing", 10)
        assert not credits.has_enough(db, user_id, "processing", 1)

    def test_settle_charges_real_cost_and_frees_the_rest(self, db, user_id):
        hold_id = credits.reserve(db, user_id, "processing", 8)
        assert credits.settle(db, hold_id, 3, "pipeline_process:90s:audio_translated") == (3, 7)
        assert _balance(db, user_id) == 7
        assert credits.available(db, user_id, "processing") == 7
        assert len(_txs(db, user_id, "consume")) == 1

    def test_settle_twice_charges_once(self, db, user_id):
        hold_id = credits.reserve(db, user_id, "processing", 8)
        credits.settle(db, hold_id, 3, "job")
        assert credits.settle(db, hold_id, 3, "job") is None
        assert _balance(db, user_id) == 7

    def test_settle_above_estimate_is_capped_at_balance(self, db, user_id):
        """Estimation trop basse : on debite ce qui reste, sans dette."""
        hold_id = credits.reserve(db, user_id, "processing", 2)
        assert credits.settle(db, hold_id, 25, "job") == (10, 0)

    def test_settle_unknown_hold_charges_nothing(self, db, user_id):
        assert credits.settle(db, "inconnu", 5, "job") is None
        assert _balance(db, user_id) == 10

    def test_release_frees_credits_without_debit(self, db, user_id):
        hold_id = credits.reserve(db, user_id, "processing", 8)
        assert credits.release(db, hold_id) is True
        assert credits.available(db, user_id, "processing") == 10
        assert _txs(db, user_id, "consume") == []

    def test_release_is_idempotent(self, db, user_id):
        hold_id = credits.reserve(db, user_id, "processing", 8)
        credits.release(db, hold_id)
        assert credits.release(db, hold_id) is False

    def test_expired_hold_stops_blocking_and_is_purged(self, db, user_id):
        """Pipeline redemarre en plein job : la reservation orpheline expire."""
        hold_id = credits.reserve(db, user_id, "processing", 8)
        hold = db.get(models.CreditHold, hold_id)
        hold.created_at = datetime.now(timezone.utc) - credits.HOLD_TTL - timedelta(minutes=1)
        db.commit()

        assert credits.available(db, user_id, "processing") == 10
        assert credits.reserve(db, user_id, "processing", 8) is not None
        assert db.get(models.CreditHold, hold_id) is None

    def test_holds_are_per_user(self, db, user_id):
        other = models.User(email="autre@example.com", hashed_password="x")
        db.add(other)
        db.commit()
        credits.grant_trial(db, other.id)
        credits.reserve(db, user_id, "processing", 10)
        assert credits.reserve(db, other.id, "processing", 10) is not None


# ── Reunions : debit au prorata ──────────────────────────────────────────────

class TestConsumeSeconds:
    def test_fractions_carry_over_between_chunks(self, db, user_id):
        """4 tranches de 30 s a 0.5 credit/min = 1 credit, pas 4."""
        charged = [credits.consume_seconds(db, user_id, 30, 0.5, "meeting_transcription")[0]
                   for _ in range(4)]
        assert charged == [0, 0, 0, 1]
        assert _balance(db, user_id) == 9

    def test_never_goes_into_debt(self, db, user_id):
        charged, balance = credits.consume_seconds(db, user_id, 3600, 2, "meeting_transcription")
        assert (charged, balance) == (10, 0)


# ── Abonnement : allocation mensuelle ────────────────────────────────────────

class TestMonthlyAllocation:
    def test_apply_tier_pro_adds_monthly_credits(self, db, user_id):
        credits.apply_tier(db, user_id, "pro")
        assert _balance(db, user_id) == 10 + 200

    def test_carryover_is_capped(self, db, user_id):
        row = credits._get_or_create(db, user_id)
        row.processing_balance = 500
        credits.apply_tier(db, user_id, "pro")
        assert _balance(db, user_id) == 600  # carryover_max 400 + monthly 200

    def test_refill_only_when_period_is_over(self, db, user_id):
        credits.apply_tier(db, user_id, "pro",
                           period_resets_at=datetime.now(timezone.utc) + timedelta(days=10))
        credits.refill_if_due(db, user_id, "pro")
        assert _balance(db, user_id) == 210

        row = credits._get_or_create(db, user_id)
        row.period_resets_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()
        credits.refill_if_due(db, user_id, "pro")
        assert _balance(db, user_id) == 410

    def test_downgrade_to_free_keeps_remaining_credits(self, db, user_id):
        credits.apply_tier(db, user_id, "pro")
        credits.apply_tier(db, user_id, "free")
        assert _balance(db, user_id) == 210


class TestPipelinePricing:
    @pytest.mark.parametrize("seconds, with_tts, with_diarize, expected", [
        (60,  True,  False, 2),   # 2 credits/min avec voix
        (60,  False, False, 1),   # transcription seule
        (61,  True,  False, 3),   # arrondi a l'entier superieur
        (60,  True,  True,  3),   # +0.5/min de diarisation
        (1,   True,  False, 1),   # minimum 1 credit
    ])
    def test_credits_for_pipeline(self, seconds, with_tts, with_diarize, expected):
        assert credits.credits_for_pipeline(seconds, with_tts, with_diarize) == expected
