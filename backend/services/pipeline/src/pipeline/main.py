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

from pipeline.prompt_guard import check_input, check_output, sandbox_user_text

load_dotenv()

STT_URL = os.getenv("STT_URL", "http://localhost:8001")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8002")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8003")

# ── Langfuse (optionnel — désactivé si clés absentes) ────────────────────────
_lf = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse import Langfuse
        _lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
        _lf = None

app = FastAPI(title="Pipeline Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------------------
# Langchain LCEL — 3 étapes chaînées
# ---------------------------------------------------------------------------

async def _stt_step(state: dict) -> dict:
    """Étape 1 : transcription audio → texte via STT Service.
    Timeout 600s pour couvrir le téléchargement initial du modèle (large-v3 = 3 Go)."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=600) as client:
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

    # ── Garde-fou : pre-check anti prompt injection sur le texte transcrit ──
    guard = check_input(text)
    if not guard.safe:
        # Loggué en clair pour audit + remonté en 422 lisible côté front
        print(
            f"[pipeline] BLOCKED input — reason={guard.reason} "
            f"pattern={guard.matched_pattern!r} text={text[:120]!r}",
            flush=True,
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "Contenu audio suspect détecté. Cette plateforme traduit uniquement "
                "des messages d'information routière — merci de réessayer avec un audio approprié."
            ),
        )

    return {
        **state,
        "source_text": text,
        "language": data.get("language", "fr"),
        "language_prob": lang_prob,
        "latency_stt_ms": round((time.perf_counter() - t0) * 1000),
    }


async def _llm_step(state: dict) -> dict:
    """Étape 2 : traduction texte → texte traduit via LLM Service.
    Le texte est sandboxé (échappement balises) avant envoi au LLM."""
    t0 = time.perf_counter()
    safe_text = sandbox_user_text(state["source_text"])
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{LLM_URL}/translate",
            json={
                "text": safe_text,
                "target_lang": state["target_lang"],
                "model": state["llm_model"],
                "prompt_version": state["prompt_version"],
            },
        )
    resp.raise_for_status()
    data = resp.json()
    translation = data["translation"]

    # ── Garde-fou : post-check sur la sortie LLM ─────────────────────────────
    guard = check_output(translation, state["source_text"])
    if not guard.safe:
        print(
            f"[pipeline] BLOCKED output — reason={guard.reason} "
            f"marker={guard.matched_pattern!r} translation={translation[:120]!r}",
            flush=True,
        )
        raise HTTPException(
            status_code=422,
            detail="Réponse du modèle incohérente avec la tâche de traduction. Veuillez réessayer.",
        )

    return {
        **state,
        "translation": translation,
        "latency_llm_ms": round((time.perf_counter() - t0) * 1000),
        "prompt_tokens":     data.get("prompt_tokens",     0),
        "completion_tokens": data.get("completion_tokens", 0),
        "total_tokens":      data.get("total_tokens",      0),
        "cost_usd":          data.get("cost_usd",          0.0),
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
        import traceback
        print(f"[pipeline] HTTPStatusError: {e.response.status_code} {e.response.text}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Service error: {e.response.text}")
    except Exception as e:
        import traceback
        print(f"[pipeline] Exception {type(e).__name__}: {e!r}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    latency_total_ms = round((time.perf_counter() - t_total) * 1000)

    # ── Trace Langfuse ────────────────────────────────────────────────────────
    if _lf:
        try:
            import uuid as _uuid
            tid     = str(_uuid.uuid4())
            comment = f"{filename} | {whisper_model} | {llm_model} | {prompt_version}"
            _lf.trace(
                id=tid,
                name="translation",
                metadata={
                    "whisper_model":  whisper_model,
                    "llm_model":      llm_model,
                    "prompt_version": prompt_version,
                    "target_lang":    target_lang,
                    "prompt_tokens":     result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "total_tokens":      result.get("total_tokens", 0),
                    "cost_usd":          result.get("cost_usd", 0.0),
                },
            )
            for name, value in [
                ("latency_total_ms",  latency_total_ms),
                ("latency_stt_ms",    result["latency_stt_ms"]),
                ("latency_llm_ms",    result["latency_llm_ms"]),
                ("latency_tts_ms",    result["latency_tts_ms"]),
                ("language_prob",     result["language_prob"]),
                ("cost_usd",          result.get("cost_usd", 0.0)),
                ("total_tokens",      result.get("total_tokens", 0)),
            ]:
                _lf.score(trace_id=tid, name=name, value=value, comment=comment)
            _lf.flush()
        except Exception:
            pass  # Ne jamais bloquer le pipeline pour Langfuse

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
        "prompt_tokens":      result.get("prompt_tokens", 0),
        "completion_tokens":  result.get("completion_tokens", 0),
        "total_tokens":       result.get("total_tokens", 0),
        "cost_usd":           result.get("cost_usd", 0.0),
    }
