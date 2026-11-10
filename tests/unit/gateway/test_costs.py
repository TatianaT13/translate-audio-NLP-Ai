"""Tests du suivi des coûts réels : calcul du coût d'un débit et marge par service."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from gateway import auth, costs, credits, models
from gateway.database import SessionLocal
from gateway.main import app

HEADERS = {"x-service-token": "test-service-token"}
client = TestClient(app)


class TestUsageCost:
    def test_translation_job_adds_stt_tts_and_llm(self):
        meta = {"seconds": 600, "stt_backend": "openai", "tts_chars": 9000,
                "tts_backend": "mistral", "llm_cost_usd": 0.002}
        # 10 min x 0,006 + 9 000 car. x 0,016/1000 + LLM
        assert costs.usage_cost_usd(meta) == pytest.approx(0.06 + 0.144 + 0.002)

    def test_local_backends_cost_nothing_per_use(self):
        meta = {"seconds": 600, "stt_backend": "local", "tts_chars": 9000, "tts_backend": "local"}
        assert costs.usage_cost_usd(meta) == 0

    def test_meeting_chunk_has_only_stt(self):
        assert costs.usage_cost_usd({"seconds": 30, "stt_backend": "openai"}) == pytest.approx(0.003)

    def test_groq_transcription_is_nine_times_cheaper_than_openai(self):
        groq = costs.usage_cost_usd({"seconds": 600, "stt_backend": "groq"})
        openai = costs.usage_cost_usd({"seconds": 600, "stt_backend": "openai"})
        assert groq == pytest.approx(0.0067) and openai / groq == pytest.approx(9, abs=0.1)

    def test_voice_on_demand_costs_only_the_synthesis(self):
        meta = {"seconds": 0, "stt_backend": "none", "tts_chars": 4500, "tts_backend": "mistral"}
        assert costs.usage_cost_usd(meta) == pytest.approx(0.072)

    def test_unknown_backend_uses_the_highest_rate(self):
        """Ne jamais sous-estimer un coût."""
        unknown = costs.usage_cost_usd({"seconds": 60, "stt_backend": "unknown"})
        assert unknown == pytest.approx(max(costs.STT_USD_PER_MIN.values()))

    @pytest.mark.parametrize("meta", [None, {}, {"seconds": 120}])
    def test_untracked_debit_has_no_cost(self, meta):
        """Débits antérieurs au suivi : coût inconnu, pas zéro."""
        assert costs.usage_cost_usd(meta) is None


def _new_user(is_admin: bool = False) -> int:
    db = SessionLocal()
    user = models.User(email=f"{uuid.uuid4().hex}@example.com", hashed_password="x", is_admin=is_admin)
    db.add(user)
    db.commit()
    credits.apply_tier(db, user.id, "pro")
    uid = user.id
    db.close()
    return uid


def _admin_headers() -> dict:
    uid = _new_user(is_admin=True)
    return {"Authorization": f"Bearer {auth.create_access_token(uid, 'admin@example.com', True)}"}


def _service(report: dict, name: str) -> dict:
    return next(s for s in report["services"] if s["service"] == name)


class TestAdminMargins:
    def test_requires_admin(self):
        uid = _new_user()
        token = auth.create_access_token(uid, "user@example.com", False)
        assert client.get("/admin/margins").status_code == 401
        assert client.get("/admin/margins", headers={"Authorization": f"Bearer {token}"}).status_code == 403

    def test_margin_of_a_translation_job(self):
        admin = _admin_headers()
        before = _service_or_blank(client.get("/admin/margins", headers=admin).json(), "translation")

        uid = _new_user()
        hold = client.post("/billing/reserve", headers=HEADERS,
                           json={"user_id": uid, "kind": "processing", "amount": 20}).json()["hold_id"]
        client.post("/billing/settle", headers=HEADERS, json={
            "hold_id": hold, "amount": 20, "reason": "pipeline_process:600s:audio_translated",
            "meta": {"seconds": 600, "stt_backend": "openai", "tts_chars": 9000,
                     "tts_backend": "mistral", "llm_cost_usd": 0.002},
        })

        after = _service(client.get("/admin/margins", headers=admin).json(), "translation")
        assert after["credits"] - before["credits"] == 20
        assert after["seconds"] - before["seconds"] == pytest.approx(600)
        assert after["revenue_eur"] - before["revenue_eur"] == pytest.approx(20 * costs.CREDIT_VALUE_EUR, abs=1e-3)
        assert after["cost_eur"] - before["cost_eur"] == pytest.approx(0.206 * costs.USD_TO_EUR, abs=1e-3)

    def test_meeting_chunks_carry_their_stt_backend(self):
        admin = _admin_headers()
        before = _service_or_blank(client.get("/admin/margins", headers=admin).json(), "meeting_transcription")

        uid = _new_user()
        for _ in range(4):  # 4 tranches de 30 s : 1 crédit, 2 min d'audio
            client.post("/billing/consume-seconds", headers=HEADERS, json={
                "user_id": uid, "seconds": 30, "rate_per_min": 0.5, "reason": "meeting_transcription",
                "meta": {"stt_backend": "openai", "stt_model": "whisper-1"},
            })

        after = _service(client.get("/admin/margins", headers=admin).json(), "meeting_transcription")
        assert after["credits"] - before["credits"] == 1
        assert after["seconds"] - before["seconds"] == pytest.approx(120)
        assert after["cost_eur"] - before["cost_eur"] == pytest.approx(0.012 * costs.USD_TO_EUR, abs=1e-3)

    def test_debits_without_usage_are_counted_apart(self):
        admin = _admin_headers()
        before = _service_or_blank(client.get("/admin/margins", headers=admin).json(), "translation")

        uid = _new_user()
        client.post("/billing/consume", headers=HEADERS, json={
            "user_id": uid, "kind": "processing", "amount": 6,
            "reason": "pipeline_process:132s:audio_translated",
        })

        after = _service(client.get("/admin/margins", headers=admin).json(), "translation")
        assert after["untracked_credits"] - before["untracked_credits"] == 6
        assert after["credits"] == before["credits"]          # hors calcul de marge
        assert after["cost_eur"] == before["cost_eur"]

    def test_margin_percentage(self):
        report = client.get("/admin/margins", headers=_admin_headers()).json()
        total = report["total"]
        assert total["margin_pct"] == pytest.approx(
            (total["revenue_eur"] - total["cost_eur"]) / total["revenue_eur"] * 100, abs=0.2)


def _service_or_blank(report: dict, name: str) -> dict:
    blank = {"credits": 0, "seconds": 0.0, "revenue_eur": 0.0, "cost_eur": 0.0, "untracked_credits": 0}
    return next((s for s in report["services"] if s["service"] == name), blank)
