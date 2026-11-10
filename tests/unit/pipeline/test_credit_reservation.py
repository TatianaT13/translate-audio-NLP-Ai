"""Tests unitaires — facturation cote pipeline (reserve / settle / release).

Verifie surtout les cas de panne : Gateway injoignable ou en erreur. Avant la
reservation, ces cas laissaient passer le traitement sans le facturer.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import HTTPException

from pipeline import main


class _FakeGateway:
    """Remplace httpx.AsyncClient : rejoue une liste de reponses / exceptions."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.calls.append((url.rsplit("/", 1)[-1], json))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))


@pytest.fixture
def gateway(monkeypatch):
    def install(*outcomes) -> _FakeGateway:
        fake = _FakeGateway(*outcomes)
        monkeypatch.setattr(main.httpx, "AsyncClient", fake)
        return fake
    monkeypatch.setattr(main, "_SETTLE_RETRY_DELAYS", (0, 0))
    return install


class TestReserve:
    def test_returns_hold_id(self, gateway):
        gateway((200, {"ok": True, "hold_id": "h-1", "balance": 10, "needed": 4}))
        assert asyncio.run(main._reserve_credits(1, 4)) == "h-1"

    def test_insufficient_credits_is_402(self, gateway):
        gateway((200, {"ok": False, "hold_id": None, "balance": 2, "needed": 4, "tier": "free"}))
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main._reserve_credits(1, 4))
        assert exc.value.status_code == 402
        assert exc.value.detail == {"reason": "credits_insufficient", "balance": 2, "needed": 4, "tier": "free"}

    @pytest.mark.parametrize("outcome", [
        httpx.ConnectError("gateway down"),
        (500, {"detail": "boom"}),
        (401, {"detail": "Service token invalide"}),
    ])
    def test_gateway_failure_blocks_the_job(self, gateway, outcome):
        """Pas de reservation, pas de traitement : une panne n'offre rien."""
        gateway(outcome)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main._reserve_credits(1, 4))
        assert exc.value.status_code == 503


class TestSettle:
    def test_settles_once_on_success(self, gateway):
        fake = gateway((200, {"ok": True, "charged": 3, "balance": 7}))
        asyncio.run(main._settle_credits("h-1", 3, "pipeline_process:90s:audio_translated"))
        assert fake.calls == [("settle", {"hold_id": "h-1", "amount": 3, "meta": None,
                                          "reason": "pipeline_process:90s:audio_translated"})]

    def test_retries_after_a_failure(self, gateway):
        fake = gateway(httpx.ConnectError("down"), (500, {}), (200, {"ok": True}))
        asyncio.run(main._settle_credits("h-1", 3, "job"))
        assert len(fake.calls) == 3

    def test_gives_up_without_raising(self, gateway):
        """Le resultat est deja calcule : on ne le refuse pas au client."""
        fake = gateway(*[httpx.ConnectError("down")] * 3)
        asyncio.run(main._settle_credits("h-1", 3, "job"))
        assert len(fake.calls) == 3


class TestRelease:
    def test_releases_the_hold(self, gateway):
        fake = gateway((200, {"ok": True, "released": True}))
        asyncio.run(main._release_credits("h-1"))
        assert fake.calls == [("release", {"hold_id": "h-1"})]

    def test_no_hold_no_call(self, gateway):
        fake = gateway()
        asyncio.run(main._release_credits(None))
        assert fake.calls == []

    def test_failure_is_swallowed(self, gateway):
        gateway(httpx.ConnectError("down"))
        asyncio.run(main._release_credits("h-1"))


class TestBillableSeconds:
    """La duree facturee est celle de l'audio, pas une estimation sur le poids du fichier."""

    def test_uses_duration_measured_by_stt(self):
        # 2 min d'audio dans un fichier lourd (WAV) : 120 s, pas 20 min
        result = {"stt_duration": 120.0, "stt_segments": [{"end": 118.2}]}
        assert main._billable_seconds(result, audio_bytes=10 * 1024 * 1024) == 120.0

    def test_falls_back_to_last_segment_end(self):
        result = {"stt_duration": 0, "stt_segments": [{"end": 12.0}, {"end": 95.5}]}
        assert main._billable_seconds(result, audio_bytes=10 * 1024 * 1024) == 95.5

    def test_falls_back_to_file_size_when_stt_gives_nothing(self):
        assert main._billable_seconds({}, audio_bytes=8 * 1024 * 60) == 60.0

    def test_stt_step_keeps_the_duration(self, monkeypatch):
        class FakeStt:
            def __call__(self, *a, **k): return self
            async def __aenter__(self): return self
            async def __aexit__(self, *exc): return False
            async def post(self, url, **kwargs):
                return httpx.Response(200, request=httpx.Request("POST", url), json={
                    "text": "Bonjour à tous, la réunion commence.", "language": "fr",
                    "language_probability": 0.99, "duration": 652.4,
                    "segments": [{"start": 0.0, "end": 650.0, "text": "Bonjour à tous, la réunion commence."}],
                })

        monkeypatch.setattr(main.httpx, "AsyncClient", FakeStt())
        out = asyncio.run(main._stt_step({"filename": "a.mp3", "audio_bytes": b"x", "whisper_model": "small"}))
        assert main._billable_seconds(out, audio_bytes=1) == 652.4


class TestUsageMeta:
    """Volumes remontes a la Gateway avec le debit, pour le suivi des couts reels."""

    def test_reports_what_the_job_consumed(self):
        result = {"stt_backend": "openai", "stt_model": "whisper-1", "translation": "x" * 3900,
                  "audio_b64": "QUJD", "audio_content_type": "audio/mpeg", "cost_usd": 0.00075}
        assert main._usage_meta(result, 312.44) == {
            "seconds": 312.4, "stt_backend": "openai", "stt_model": "whisper-1",
            "tts_chars": 3900, "tts_backend": "mistral", "llm_cost_usd": 0.00075,
        }

    def test_no_voice_generated_means_no_voice_cost(self):
        """Voix à la demande : sans audio produit, aucun caractère synthétisé à compter."""
        meta = main._usage_meta({"translation": "x" * 3900, "audio_b64": "", "audio_content_type": ""}, 300)
        assert meta["tts_chars"] == 0

    def test_local_tts_is_detected_from_wav_output(self):
        meta = main._usage_meta({"translation": "Hello.", "audio_b64": "QUJD", "audio_content_type": "audio/wav"}, 5)
        assert meta["tts_backend"] == "local"
        assert meta["stt_backend"] == "unknown"   # la Gateway appliquera le tarif le plus haut

    def test_settle_sends_the_usage(self, gateway):
        fake = gateway((200, {"ok": True}))
        asyncio.run(main._settle_credits("h-1", 3, "job", {"seconds": 90.0, "stt_backend": "local"}))
        assert fake.calls[0][1]["meta"] == {"seconds": 90.0, "stt_backend": "local"}
