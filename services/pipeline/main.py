"""
Pipeline Service — Port 8000
Orchestrateur Langchain LCEL : Audio → STT → LLM → TTS

POST /process  : audio file → { source_text, translation, audio_b64, latencies }
GET  /health   : statut + URLs des services aval
"""

import base64
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnableLambda

load_dotenv()

STT_URL = os.getenv("STT_URL", "http://localhost:8001")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8002")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8003")

app = FastAPI(title="Pipeline Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------------------
# Langchain LCEL — 3 étapes chaînées
# ---------------------------------------------------------------------------

async def _stt_step(state: dict) -> dict:
    """Étape 1 : transcription audio → texte via STT Service."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{STT_URL}/transcribe",
            files={"file": (state["filename"], state["audio_bytes"], "audio/mpeg")},
            data={"model": state["whisper_model"], "language": "fr"},
        )
    resp.raise_for_status()
    data = resp.json()

    text = data.get("text", "").strip()
    lang_prob = data.get("language_probability", 0)

    if not text:
        raise ValueError("No speech detected in audio.")

    if lang_prob < 0.4:
        raise ValueError(
            f"Audio unclear or not French (confidence {lang_prob:.0%}). "
            "Please use a clear French speech recording."
        )

    return {
        **state,
        "source_text": text,
        "language": data.get("language", "fr"),
        "language_prob": lang_prob,
        "latency_stt_ms": round((time.perf_counter() - t0) * 1000),
    }


async def _llm_step(state: dict) -> dict:
    """Étape 2 : traduction texte → texte traduit via LLM Service."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{LLM_URL}/translate",
            json={
                "text": state["source_text"],
                "target_lang": state["target_lang"],
                "model": state["llm_model"],
                "prompt_version": state["prompt_version"],
            },
        )
    resp.raise_for_status()
    data = resp.json()
    return {
        **state,
        "translation": data["translation"],
        "latency_llm_ms": round((time.perf_counter() - t0) * 1000),
    }


async def _tts_step(state: dict) -> dict:
    """Étape 3 : synthèse vocale texte traduit → audio via TTS Service."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{TTS_URL}/synthesize",
            json={"text": state["translation"], "lang": state["target_lang"]},
        )
    resp.raise_for_status()
    audio_bytes = resp.content
    audio_b64 = base64.b64encode(audio_bytes).decode()
    content_type = resp.headers.get("content-type", "audio/mpeg")
    return {
        **state,
        "audio_b64": audio_b64,
        "audio_content_type": content_type,
        "latency_tts_ms": round((time.perf_counter() - t0) * 1000),
    }


# Chaîne LCEL : STT | LLM | TTS
pipeline_chain = (
    RunnableLambda(_stt_step)
    | RunnableLambda(_llm_step)
    | RunnableLambda(_tts_step)
)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "services": {
            "stt": STT_URL,
            "llm": LLM_URL,
            "tts": TTS_URL,
        },
    }


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    target_lang: str = Form("en"),
    llm_model: str = Form("groq/llama-3.1-8b-instant"),
    prompt_version: str = Form("v1.1"),
    whisper_model: str = Form("small"),
):
    """
    Pipeline complet : audio → transcription → traduction → synthèse vocale.

    - **file**          : fichier audio (MP3, WAV, M4A...)
    - **target_lang**   : langue cible (en, uk, es, de)
    - **llm_model**     : modèle LiteLLM (groq/llama-3.1-8b-instant...)
    - **prompt_version**: version du prompt (v1.0, v1.1, v1.2)
    - **whisper_model** : modèle Whisper (small, large-v3...)
    """
    audio_bytes = await file.read()
    filename = file.filename or "audio.mp3"

    initial_state = {
        "audio_bytes": audio_bytes,
        "filename": filename,
        "target_lang": target_lang,
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "whisper_model": whisper_model,
    }

    t_total = time.perf_counter()
    try:
        result = await pipeline_chain.ainvoke(initial_state)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Service error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_total_ms = round((time.perf_counter() - t_total) * 1000)

    return {
        "source_text":        result["source_text"],
        "language":           result["language"],
        "language_prob":      result["language_prob"],
        "translation":        result["translation"],
        "audio_b64":          result["audio_b64"],
        "audio_content_type": result["audio_content_type"],
        "latency_stt_ms":     result["latency_stt_ms"],
        "latency_llm_ms":     result["latency_llm_ms"],
        "latency_tts_ms":     result["latency_tts_ms"],
        "latency_total_ms":   latency_total_ms,
    }
