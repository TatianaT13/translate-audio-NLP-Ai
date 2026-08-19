"""Test d'intégration bout-en-bout du pipeline complet.

Envoie un vrai audio FR → attend transcription + traduction + audio synthétisé.
Prérequis : pipeline + stt + llm + tts services up (docker compose up).

Note : ces tests sont LENTS (30-60s chacun) car ils font tourner Whisper + Groq + TTS.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def audio_upload(sample_audio_path: Path) -> tuple:
    """Prépare le tuple (filename, bytes, content_type) pour httpx.files."""
    return (sample_audio_path.name, sample_audio_path.read_bytes(), "audio/mpeg")


@pytest.fixture(scope="module")
def process_response(pipeline_url: str, audio_upload: tuple) -> dict:
    """Fait UN appel /process au pipeline (module-scoped → réutilisé par tous les tests).
    Skip élégant si une dépendance externe (LLM provider, TTS) est indisponible."""
    with httpx.Client(timeout=180.0) as client:
        r = client.post(
            f"{pipeline_url}/process",
            files={"file": audio_upload},
            data={"target_lang": "en", "whisper_model": "small"},
        )

    # Dépendances externes indisponibles → skip plutôt que fail
    if r.status_code == 502:
        pytest.skip(f"Service aval indisponible : {r.text[:200]}")
    if r.status_code == 422 and "guardrail" in r.text.lower():
        pytest.skip(f"Content guardrail déclenché : {r.text[:200]}")

    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
    return r.json()


class TestPipelineE2E:
    """Le pipeline complet transforme un audio FR en traduction + audio cible."""

    def test_process_returns_all_expected_fields(self, process_response):
        for key in (
            "source_text", "translation", "audio_b64", "audio_content_type",
            "language", "language_prob",
            "latency_stt_ms", "latency_llm_ms", "latency_tts_ms", "latency_total_ms",
        ):
            assert key in process_response, f"champ manquant : {key}"

    def test_transcription_is_french(self, process_response):
        assert process_response["language"] == "fr"
        assert process_response["language_prob"] > 0.5
        assert len(process_response["source_text"]) > 10

    def test_translation_is_english_and_non_empty(self, process_response):
        translation = process_response["translation"]
        assert isinstance(translation, str)
        assert len(translation) > 10
        assert translation.strip() != process_response["source_text"].strip()

    def test_audio_b64_is_valid_base64(self, process_response):
        audio_bytes = base64.b64decode(process_response["audio_b64"])
        assert len(audio_bytes) > 1000  # au moins 1 Ko d'audio
        assert process_response["audio_content_type"] in ("audio/mpeg", "audio/wav")

    def test_latencies_are_positive(self, process_response):
        assert process_response["latency_stt_ms"] > 0
        assert process_response["latency_llm_ms"] > 0
        assert process_response["latency_tts_ms"] > 0
        parts = (
            process_response["latency_stt_ms"]
            + process_response["latency_llm_ms"]
            + process_response["latency_tts_ms"]
        )
        assert process_response["latency_total_ms"] >= parts * 0.9

    def test_health_endpoint_ok(self, client, pipeline_url):
        r = client.get(f"{pipeline_url}/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        for svc in ("stt", "llm", "tts"):
            assert svc in body["services"]
