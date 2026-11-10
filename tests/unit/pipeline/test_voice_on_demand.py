"""Tests unitaires — voix à la demande, tarif sans voix, affichage progressif.

La voix de synthèse est le poste le plus coûteux d'une traduction : elle n'est
générée (et facturée) que si l'utilisateur l'écoute.
"""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from pipeline import main

SECRET = "s" * 32
HEADERS = {"Authorization": "Bearer " + jwt.encode({"sub": "7", "type": "access"}, SECRET, algorithm="HS256")}


@pytest.fixture
def billing(monkeypatch):
    """Gateway simulée : réservations, débits et annulations enregistrés."""
    log = {"reserved": [], "settled": [], "released": []}

    async def reserve(user_id, amount, kind="processing"):
        log["reserved"].append(amount)
        return "h-1"

    async def settle(hold_id, amount, reason, meta=None):
        log["settled"].append((amount, reason, meta))

    async def release(hold_id):
        if hold_id:
            log["released"].append(hold_id)

    monkeypatch.setattr(main, "JWT_SECRET", SECRET)
    monkeypatch.setattr(main, "_reserve_credits", reserve)
    monkeypatch.setattr(main, "_settle_credits", settle)
    monkeypatch.setattr(main, "_release_credits", release)
    monkeypatch.setattr(main, "_user_jobs", {})
    monkeypatch.setattr(main, "_jobs_total", 0)
    return log


class TestPricing:
    @pytest.mark.parametrize("with_tts, diarize, expected, kind", [
        (False, False, 5,  "text_translated"),            # 1 crédit/min : traduction seule
        (True,  False, 10, "audio_translated"),           # +1 : voix
        (False, True,  8,  "text_translated_diarized"),   # +0,5 : intervenants (7,5 arrondi)
        (True,  True,  13, "audio_translated_diarized"),
    ])
    def test_five_minutes_of_audio(self, with_tts, diarize, expected, kind):
        result = {"stt_duration": 300.0, "stt_backend": "groq", "translation": "Hello.",
                  "audio_b64": "QUJD" if with_tts else "", "audio_content_type": "audio/mpeg" if with_tts else ""}
        credits, reason, meta = main._job_charge(result, 1, with_tts, diarize, "pipeline_process")
        assert credits == expected
        assert reason == f"pipeline_process:300s:{kind}"
        assert meta["seconds"] == 300.0

    def test_estimate_before_processing_follows_the_same_rate(self):
        size = 8 * 1024 * 60 * 5      # 5 min à 64 kbps
        assert main._estimate_credits_from_filesize(size, with_tts=False) == 5
        assert main._estimate_credits_from_filesize(size, with_tts=True) == 10

    @pytest.mark.parametrize("chars, credits", [(1, 1), (900, 1), (901, 2), (4500, 5)])
    def test_voice_is_one_credit_per_900_characters(self, chars, credits):
        assert main._voice_credits(chars) == credits


class TestTtsStep:
    def test_skipped_when_voice_is_not_requested(self):
        out = asyncio.run(main._tts_step({"translation": "Hello.", "target_lang": "en", "with_tts": False}))
        assert (out["audio_b64"], out["latency_tts_ms"], out["n_tts_chunks"]) == ("", 0, 0)


class TestVoiceEndpoint:
    @pytest.fixture
    def tts(self, monkeypatch):
        async def fake(state):
            return {**state, "audio_b64": base64.b64encode(b"MP3" + state["translation"].encode()).decode(),
                    "audio_content_type": "audio/mpeg"}
        monkeypatch.setattr(main, "_tts_step", fake)

    def test_returns_the_audio_and_charges_by_length(self, billing, tts):
        text = "Hello everyone. " * 100                      # 1 600 caractères → 2 crédits
        r = TestClient(main.app).post("/tts", headers=HEADERS, json={"text": text, "target_lang": "en"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mpeg"
        assert r.content.startswith(b"MP3Hello everyone.")
        assert billing["reserved"] == [2]
        (amount, reason, meta), = billing["settled"]
        assert (amount, reason) == (2, f"voice:{len(text.strip())}c")
        assert meta["tts_chars"] == len(text.strip()) and meta["tts_backend"] == "mistral"

    def test_failed_synthesis_charges_nothing(self, billing, monkeypatch):
        async def boom(state):
            raise RuntimeError("fournisseur en panne")
        monkeypatch.setattr(main, "_tts_step", boom)
        r = TestClient(main.app).post("/tts", headers=HEADERS, json={"text": "Hello."})
        assert r.status_code == 502
        assert billing["settled"] == [] and billing["released"] == ["h-1"]

    def test_provider_refusal_is_passed_on_and_not_charged(self, billing, monkeypatch):
        async def refused(state):
            raise HTTPException(status_code=422, detail="Synthèse vocale refusée")
        monkeypatch.setattr(main, "_tts_step", refused)
        r = TestClient(main.app).post("/tts", headers=HEADERS, json={"text": "Hello."})
        assert r.status_code == 422 and billing["settled"] == []

    def test_login_is_required(self):
        assert TestClient(main.app).post("/tts", json={"text": "Hello."}).status_code == 401

    def test_empty_text_is_refused(self, billing, tts):
        r = TestClient(main.app).post("/tts", headers=HEADERS, json={"text": "   "})
        assert r.status_code == 400 and billing["reserved"] == []

    def test_oversized_text_is_refused(self, billing, tts, monkeypatch):
        monkeypatch.setattr(main, "MAX_VOICE_CHARS", 10)
        r = TestClient(main.app).post("/tts", headers=HEADERS, json={"text": "x" * 11})
        assert r.status_code == 413


class TestProgressiveJob:
    """Le job publie la transcription, puis la traduction, avant la fin."""

    def _run(self, monkeypatch, with_tts: bool):
        seen = []

        async def stt(state):
            return {**state, "source_text": "Bonjour à tous.", "language": "fr", "language_prob": 0.99,
                    "latency_stt_ms": 1, "stt_duration": 60.0, "stt_backend": "groq"}

        async def llm(state):
            job = await main._jobs.store.get(state["job_id"])
            seen.append(("après transcription", dict(job.partial)))
            return {**state, "translation": "Hello everyone.", "translation_segments": None, "latency_llm_ms": 1}

        async def tts(state):
            job = await main._jobs.store.get(state["job_id"])
            seen.append(("après traduction", dict(job.partial)))
            return {**state, "audio_b64": "QUJD" if state["with_tts"] else "",
                    "audio_content_type": "audio/mpeg" if state["with_tts"] else "", "latency_tts_ms": 0}

        monkeypatch.setattr(main, "_stt_step", stt)
        monkeypatch.setattr(main, "_llm_step", llm)
        monkeypatch.setattr(main, "_tts_step", tts)
        monkeypatch.setattr(main._jobs, "store", main._jobs.JobStore())

        async def run():
            monkeypatch.setattr(main, "_job_slots", asyncio.Semaphore(8))
            job = await main._jobs.store.create(user_id=7, hold_id="h-1")
            # le job_id voyage dans l'état pour que les étapes simulées lisent le job
            original = main._stt_step
            async def stt_with_id(state):
                return await original({**state, "job_id": job.id})
            monkeypatch.setattr(main, "_stt_step", stt_with_id)
            await main._execute_job(
                job_id=job.id, audio_bytes=b"x", filename="a.mp3", target_lang="en",
                llm_model="openai/gpt-4o-mini", prompt_version="v2.0", whisper_model="small",
                diarize=False, hold_id="h-1", user_id=None, with_tts=with_tts,
            )
            return await main._jobs.store.get(job.id)

        return asyncio.run(run()), seen

    def test_transcript_then_translation_are_published_before_the_end(self, billing, monkeypatch):
        job, seen = self._run(monkeypatch, with_tts=False)
        assert seen[0] == ("après transcription",
                           {"source_text": "Bonjour à tous.", "language": "fr", "language_prob": 0.99})
        assert seen[1][1]["translation"] == "Hello everyone."
        assert job.status == "done" and job.partial is None
        assert job.result["translation"] == "Hello everyone." and job.result["audio_b64"] == ""

    def test_job_without_voice_is_charged_at_the_text_rate(self, billing, monkeypatch):
        self._run(monkeypatch, with_tts=False)
        (amount, reason, _), = billing["settled"]
        assert (amount, reason) == (1, "pipeline_process_async:60s:text_translated")

    def test_job_with_voice_is_charged_at_the_voice_rate(self, billing, monkeypatch):
        self._run(monkeypatch, with_tts=True)
        (amount, reason, _), = billing["settled"]
        assert (amount, reason) == (2, "pipeline_process_async:60s:audio_translated")
