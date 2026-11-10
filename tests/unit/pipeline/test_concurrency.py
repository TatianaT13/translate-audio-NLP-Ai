"""Tests unitaires — appels LLM / TTS en parallèle dans le pipeline.

Les chunks d'un même job partent en parallèle (borné) ; le résultat doit rester
strictement dans l'ordre du texte source.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import HTTPException

from pipeline import main


class TestMapConcurrent:
    def test_results_keep_item_order(self):
        async def fn(n):
            await asyncio.sleep(0.01 * (5 - n))  # les derniers finissent en premier
            return n * 10

        assert asyncio.run(main._map_concurrent(fn, [1, 2, 3, 4], limit=4)) == [10, 20, 30, 40]

    def test_never_exceeds_the_limit(self):
        running = peak = 0

        async def fn(_):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

        asyncio.run(main._map_concurrent(fn, list(range(12)), limit=3))
        assert peak == 3

    def test_runs_in_parallel(self):
        async def fn(_):
            await asyncio.sleep(0.05)

        async def timed():
            t0 = asyncio.get_running_loop().time()
            await main._map_concurrent(fn, list(range(6)), limit=6)
            return asyncio.get_running_loop().time() - t0

        assert asyncio.run(timed()) < 0.2  # 0.3 s en séquentiel

    def test_first_error_is_raised_and_cancels_the_rest(self):
        started = []

        async def fn(n):
            started.append(n)
            if n == 0:
                raise HTTPException(status_code=422, detail="boom")
            await asyncio.sleep(1)

        async def run():
            t0 = asyncio.get_running_loop().time()
            with pytest.raises(HTTPException) as exc:
                await main._map_concurrent(fn, list(range(10)), limit=2)
            return exc.value, asyncio.get_running_loop().time() - t0

        err, elapsed = asyncio.run(run())
        assert err.status_code == 422       # l'erreur d'origine, pas un ExceptionGroup
        assert elapsed < 0.5                # n'attend pas les autres appels
        assert len(started) <= 3            # la file ne part pas (au plus un appel prend la place libérée)

    def test_empty_list(self):
        async def fn(_):
            raise AssertionError("ne doit pas être appelé")

        assert asyncio.run(main._map_concurrent(fn, [], limit=3)) == []


def _fake_translate(delays: dict[str, float] | None = None, seen: dict[str, str] | None = None):
    async def translate(client, text, target_lang, llm_model, prompt_version, context=""):
        if seen is not None:
            seen[text] = context
        await asyncio.sleep((delays or {}).get(text, 0))
        return {"translation": f"EN({text})", "prompt_tokens": 10, "completion_tokens": 5,
                "total_tokens": 15, "cost_usd": 0.001}
    return translate


def _state(**extra) -> dict:
    return {"target_lang": "en", "llm_model": "openai/gpt-4o-mini", "prompt_version": "v1.1", **extra}


class TestLlmStep:
    def test_chunks_are_joined_in_source_order(self, monkeypatch):
        monkeypatch.setattr(main, "LLM_CHUNK_MAX_CHARS", 30)
        source = "Phrase un premier. Phrase deux ok. Phrase trois voilà."
        chunks = main._split_into_chunks(source, 30)
        assert len(chunks) == 3
        # le premier chunk répond en dernier
        monkeypatch.setattr(main, "_translate_chunk", _fake_translate({chunks[0]: 0.05}))

        out = asyncio.run(main._llm_step(_state(source_text=source)))

        assert out["translation"] == " ".join(f"EN({c})" for c in chunks)
        assert out["n_chunks"] == 3
        assert (out["prompt_tokens"], out["completion_tokens"], out["total_tokens"]) == (30, 15, 45)
        assert out["cost_usd"] == pytest.approx(0.003)
        assert out["translation_segments"] is None

    def test_diarized_segments_keep_their_speaker(self, monkeypatch):
        monkeypatch.setattr(main, "_translate_chunk", _fake_translate({"Bonjour à tous.": 0.05}))
        segments = [
            {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Bonjour à tous."},
            {"speaker": "B", "start": 1.0, "end": 2.0, "text": "   "},          # segment vide : ignoré
            {"speaker": "B", "start": 2.0, "end": 3.0, "text": " Merci, on commence. "},
        ]
        source = "Bonjour à tous. Merci, on commence."

        out = asyncio.run(main._llm_step(_state(source_text=source, diarize=True, stt_segments=segments)))

        assert out["translation_segments"] == [
            {"speaker": "A", "start": 0.0, "end": 1.0, "original": "Bonjour à tous.",
             "translation": "EN(Bonjour à tous.)"},
            {"speaker": "B", "start": 2.0, "end": 3.0, "original": "Merci, on commence.",
             "translation": "EN(Merci, on commence.)"},
        ]
        assert out["translation"] == "EN(Bonjour à tous.) EN(Merci, on commence.)"
        assert out["total_tokens"] == 30


class TestTranslationContext:
    """Chaque passage est traduit avec ce qui le précède pour contexte."""

    def test_meeting_segments_get_the_two_previous_lines(self, monkeypatch):
        seen: dict[str, str] = {}
        monkeypatch.setattr(main, "_translate_chunk", _fake_translate(seen=seen))
        segments = [
            {"speaker": "A", "text": "Paul reprend le dossier Durand."},
            {"speaker": "B", "text": "Et Marie ?"},
            {"speaker": "A", "text": "Oui, elle aussi."},
            {"speaker": "B", "text": "Parfait."},
        ]
        source = " ".join(s["text"] for s in segments)
        asyncio.run(main._llm_step(_state(source_text=source, diarize=True, stt_segments=segments)))

        assert seen["Paul reprend le dossier Durand."] == ""                       # rien avant
        assert seen["Oui, elle aussi."] == "Paul reprend le dossier Durand. Et Marie ?"
        assert seen["Parfait."] == "Et Marie ? Oui, elle aussi."                    # 2 répliques, pas plus

    def test_long_text_chunks_get_the_end_of_the_previous_chunk(self, monkeypatch):
        seen: dict[str, str] = {}
        monkeypatch.setattr(main, "LLM_CHUNK_MAX_CHARS", 30)
        monkeypatch.setattr(main, "_translate_chunk", _fake_translate(seen=seen))
        source = "Phrase un premier. Phrase deux ok. Phrase trois voilà."
        chunks = main._split_into_chunks(source, 30)
        asyncio.run(main._llm_step(_state(source_text=source)))
        assert [seen[c] for c in chunks] == ["", chunks[0], chunks[1]]

    def test_context_is_capped(self):
        long = ["x" * 5000, "suite"]
        assert len(main._contexts(long, previous=1)[1]) == main.CONTEXT_MAX_CHARS


class _FakeTts:
    """Remplace httpx.AsyncClient pour le service TTS."""

    def __init__(self, respond):
        self.respond = respond

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        return await self.respond(json["text"], httpx.Request("POST", url))


class TestTtsStep:
    def test_audio_parts_are_concatenated_in_text_order(self, monkeypatch):
        monkeypatch.setattr(main, "TTS_CHUNK_MAX_CHARS", 30)
        translation = "Sentence one first. Sentence two ok. Sentence three here."
        chunks = main._split_into_chunks(translation, 30)
        assert len(chunks) == 3

        async def respond(text, request):
            await asyncio.sleep(0.05 if text == chunks[0] else 0)  # le premier répond en dernier
            return httpx.Response(200, content=f"<{text}>".encode(),
                                  headers={"content-type": "audio/mpeg"}, request=request)

        monkeypatch.setattr(main.httpx, "AsyncClient", _FakeTts(respond))
        out = asyncio.run(main._tts_step(_state(translation=translation)))

        import base64
        assert base64.b64decode(out["audio_b64"]) == "".join(f"<{c}>" for c in chunks).encode()
        assert out["audio_content_type"] == "audio/mpeg"
        assert out["n_tts_chunks"] == 3

    def test_provider_guardrail_becomes_422(self, monkeypatch):
        async def respond(text, request):
            return httpx.Response(500, text="guardrail_violation", request=request)

        monkeypatch.setattr(main.httpx, "AsyncClient", _FakeTts(respond))
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main._tts_step(_state(translation="Hello.")))
        assert exc.value.status_code == 422
