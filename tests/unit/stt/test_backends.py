"""Tests unitaires — choix du fournisseur de transcription (Groq, OpenAI, AssemblyAI, local)."""

from __future__ import annotations

import httpx
import pytest

from stt import assemblyai_backend, backends, groq_backend, openai_backend

RESULT = {"text": "Bonjour.", "language": "fr", "language_probability": 1.0, "duration": 1.0, "segments": []}


@pytest.fixture
def providers(monkeypatch):
    """Active ou non chaque fournisseur, et enregistre ceux qui sont appelés."""
    calls = []

    def setup(groq=False, openai=False, assemblyai=False, groq_fails=False):
        monkeypatch.setattr(groq_backend, "is_available", lambda: groq)
        monkeypatch.setattr(openai_backend, "is_available", lambda: openai)
        monkeypatch.setattr(assemblyai_backend, "is_available", lambda: assemblyai)

        def run_groq(audio_path, language):
            calls.append("groq")
            if groq_fails:
                raise RuntimeError("Groq STT HTTP 503")
            return RESULT

        monkeypatch.setattr(groq_backend, "transcribe_with_groq", run_groq)
        monkeypatch.setattr(openai_backend, "transcribe_with_openai",
                            lambda audio_path, language: calls.append("openai") or RESULT)
        monkeypatch.setattr(assemblyai_backend, "transcribe_with_assemblyai",
                            lambda audio_path, language, diarize: calls.append("assemblyai") or RESULT)
        return calls

    return setup


def test_groq_is_used_when_selected(providers):
    calls = providers(groq=True, openai=True)
    _, model, backend = backends.transcribe_remote("a.wav", "fr", False, "groq")
    assert (backend, calls) == ("groq", ["groq"])
    assert model == f"groq/{groq_backend.GROQ_STT_MODEL}"


def test_groq_failure_falls_back_to_openai(providers):
    calls = providers(groq=True, openai=True, groq_fails=True)
    _, model, backend = backends.transcribe_remote("a.wav", "fr", False, "groq")
    assert (backend, calls) == ("openai", ["groq", "openai"])      # facturé au tarif OpenAI
    assert model == openai_backend.OPENAI_STT_MODEL


def test_groq_failure_without_fallback_is_an_error(providers):
    providers(groq=True, openai=False, groq_fails=True)
    with pytest.raises(RuntimeError):
        backends.transcribe_remote("a.wav", "fr", False, "groq")


def test_groq_selected_without_key_uses_openai(providers):
    calls = providers(groq=False, openai=True)
    assert backends.transcribe_remote("a.wav", "fr", False, "groq")[2] == "openai"
    assert backends.active("groq") == "openai"
    assert calls == ["openai"]


def test_speaker_identification_always_goes_to_assemblyai(providers):
    calls = providers(groq=True, openai=True, assemblyai=True)
    assert backends.transcribe_remote("a.wav", "fr", True, "groq")[2] == "assemblyai"
    assert calls == ["assemblyai"]


def test_no_provider_configured_means_local_whisper(providers):
    providers()
    assert backends.transcribe_remote("a.wav", "fr", False, "openai") is None
    assert backends.transcribe_remote("a.wav", "fr", False, "local") is None
    assert backends.active("local") == "local"


def test_groq_request_uses_the_openai_compatible_format(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)
    sent = {}

    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def post(self, url, headers, files, data):
            sent.update(url=url, auth=headers["Authorization"], data=data, mime=files["file"][2])
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "text": "Bonjour à tous.", "language": "french", "duration": 12.5,
                "segments": [{"start": 0.0, "end": 12.5, "text": "Bonjour à tous."}],
            })

    monkeypatch.setattr(openai_backend.httpx, "Client", FakeClient)
    monkeypatch.setattr(groq_backend, "GROQ_API_KEY", "gsk_test")

    out = groq_backend.transcribe_with_groq(str(wav), "fr")

    assert sent["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert sent["auth"] == "Bearer gsk_test"
    assert sent["data"] == {"model": groq_backend.GROQ_STT_MODEL, "response_format": "verbose_json", "language": "fr"}
    assert out["duration"] == 12.5 and out["segments"][0]["end"] == 12.5     # base de la facturation


def test_provider_error_names_the_provider(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def post(self, url, **kwargs):
            return httpx.Response(429, request=httpx.Request("POST", url), text="rate limit")

    monkeypatch.setattr(openai_backend.httpx, "Client", FakeClient)
    monkeypatch.setattr(groq_backend, "GROQ_API_KEY", "gsk_test")
    with pytest.raises(RuntimeError, match="Groq STT HTTP 429"):
        groq_backend.transcribe_with_groq(str(wav), "fr")
