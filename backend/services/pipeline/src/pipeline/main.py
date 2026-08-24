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

# ── MLflow Tracing (opt-in via ENABLE_MLFLOW_TRACING=true) ───────────────────
# Note : le tracing MLflow depuis un service distant nécessite un artifact store
# accessible (S3/MinIO/HTTP proxy). Sans ça, mlflow tente d'écrire sur le disque
# local et plante avec « Permission denied: '/mlflow' ». On l'active donc
# explicitement par opt-in — désactivé par défaut en prod.
_mlflow = None
if os.getenv("ENABLE_MLFLOW_TRACING", "false").lower() == "true":
    try:
        import mlflow as _mlflow
        _mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        _mlflow.set_experiment("pipeline-traces-live")
        print("[pipeline] MLflow tracing activé", flush=True)
    except Exception as e:
        print(f"[pipeline] MLflow tracing désactivé (erreur init) : {e}", flush=True)
        _mlflow = None


def _trace(name: str):
    """Décorateur conditionnel : applique mlflow.trace si dispo, sinon no-op."""
    def decorator(fn):
        if _mlflow is None:
            return fn
        return _mlflow.trace(name=name)(fn)
    return decorator

app = FastAPI(title="Pipeline Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator(excluded_handlers=["/health", "/metrics"]).instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Langchain LCEL — 3 étapes chaînées
# ---------------------------------------------------------------------------

@_trace(name="stt_transcribe")
async def _stt_step(state: dict) -> dict:
    """Étape 1 : transcription audio → texte via STT Service.
    Timeout 600s pour couvrir le téléchargement initial du modèle (large-v3 = 3 Go)."""
    from datetime import datetime, timezone
    t0 = time.perf_counter()
    stt_start_iso = datetime.now(timezone.utc).isoformat()
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
        "stt_start_iso":  stt_start_iso,
        "stt_end_iso":    datetime.now(timezone.utc).isoformat(),
    }


@_trace(name="llm_translate")
async def _llm_step(state: dict) -> dict:
    """Étape 2 : traduction texte → texte traduit via LLM Service.
    Le texte est sandboxé (échappement balises) avant envoi au LLM."""
    from datetime import datetime, timezone
    t0 = time.perf_counter()
    llm_start_iso = datetime.now(timezone.utc).isoformat()
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
        "llm_start_iso":     llm_start_iso,
        "llm_end_iso":       datetime.now(timezone.utc).isoformat(),
        "safe_input_text":   safe_text,
    }


@_trace(name="tts_synthesize")
async def _tts_step(state: dict) -> dict:
    """Étape 3 : synthèse vocale texte traduit → audio via TTS Service."""
    from datetime import datetime, timezone
    t0 = time.perf_counter()
    tts_start_iso = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{TTS_URL}/synthesize",
            json={"text": state["translation"], "lang": state["target_lang"]},
        )

    # Mistral Voxtral peut renvoyer "guardrail_violation" sur du contenu sensible.
    # Le TTS service le ré-emballe en 500 avec le body Mistral en clair → on détecte
    # le mot-clé dans le body, quel que soit le code de retour.
    if not resp.is_success and ("guardrail" in resp.text.lower() or "guardrail_violation" in resp.text.lower()):
        print(f"[pipeline] TTS guardrail (Mistral) — translation préservée", flush=True)
        raise HTTPException(
            status_code=422,
            detail=(
                "Synthèse vocale refusée par le fournisseur TTS (politique de contenu Mistral). "
                "La traduction texte reste disponible — l'audio n'a pas pu être généré."
            ),
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
        "audio_out_size_kb": round(len(audio_bytes) / 1024, 1),
        "tts_start_iso":   tts_start_iso,
        "tts_end_iso":     datetime.now(timezone.utc).isoformat(),
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


# ── Alias serveur : redirection des modèles obsolètes / instables ─────────────
# Protège les visiteurs dont le localStorage garde d'anciens modèles :
#   - groq/llama-3.1-8b-instant      : déprécié par Groq (août 2026)
#   - groq/openai/gpt-oss-*           : 60% de faux positifs prompt_guard
#   - groq/qwen/qwen3.6-27b           : idem instable
# → tout est redirigé vers openai/gpt-4o-mini (stable, pay-as-you-go).
# Le user qui a explicitement choisi Groq via l'UI avancée verra un warning en logs
# mais sa traduction passera sur OpenAI (garantie de succès pour la démo).
_LEGACY_MODEL_ALIASES = {
    "groq/llama-3.1-8b-instant":   "openai/gpt-4o-mini",
    "groq/openai/gpt-oss-20b":     "openai/gpt-4o-mini",
    "groq/openai/gpt-oss-120b":    "openai/gpt-4o-mini",
    "groq/qwen/qwen3.6-27b":       "openai/gpt-4o-mini",
    "groq/llama-3.3-70b-versatile":"openai/gpt-4o-mini",
    "groq/llama-3.1-70b-versatile":"openai/gpt-4o-mini",
    "groq/mixtral-8x7b-32768":     "openai/gpt-4o-mini",
}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    target_lang: str = Form("en"),
    llm_model: str = Form("openai/gpt-4o-mini"),
    prompt_version: str = Form("v1.1"),
    whisper_model: str = Form("small"),
):
    """
    Pipeline complet : audio → transcription → traduction → synthèse vocale.

    - **file**          : fichier audio (MP3, WAV, M4A...)
    - **target_lang**   : langue cible (en, uk, es, de)
    - **llm_model**     : modèle LiteLLM (openai/gpt-4o-mini par défaut)
    - **prompt_version**: version du prompt (v1.0, v1.1, v1.2)
    - **whisper_model** : modèle Whisper (small, large-v3...)
    """
    # Alias auto : redirige les modèles obsolètes / instables vers un modèle stable
    if llm_model in _LEGACY_MODEL_ALIASES:
        aliased = _LEGACY_MODEL_ALIASES[llm_model]
        print(f"[pipeline] Model alias: {llm_model} → {aliased} (client outdated)", flush=True)
        llm_model = aliased

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
    except HTTPException:
        # Erreurs déjà formatées par les étapes (garde-fou, TTS guardrail, etc.)
        raise
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

    # ── Tracing Langfuse v4 end-to-end (SDK, plus de raw /api/public/ingestion) ─
    # Migration v3→v4 : le SDK client gère OTEL sous le capot via
    # start_as_current_observation(). Chaque span est exporté automatiquement
    # à la sortie du context manager (flush explicite à la fin pour garantir
    # l'export avant que la requête FastAPI se termine).
    if _lf:
        try:
            from datetime import datetime as _dt

            prompt_tokens     = result.get("prompt_tokens",     0)
            completion_tokens = result.get("completion_tokens", 0)
            total_tokens      = result.get("total_tokens",      0)
            cost_usd          = result.get("cost_usd",          0.0)
            audio_in_size_kb  = round(len(initial_state["audio_bytes"]) / 1024, 1)
            comment           = f"{filename} | {whisper_model} | {llm_model} | {prompt_version}"

            def _iso(k: str):
                v = result.get(k)
                return _dt.fromisoformat(v) if v else None

            # Trace racine — pipeline complet
            with _lf.start_as_current_observation(
                name="translation",
                input={"audio_kb": audio_in_size_kb, "target_lang": target_lang},
                metadata={
                    "whisper_model":    whisper_model,
                    "llm_model":        llm_model,
                    "prompt_version":   prompt_version,
                    "target_lang":      target_lang,
                    "filename":         filename,
                    "total_latency_ms": latency_total_ms,
                    "cost_usd":         cost_usd,
                },
            ) as root:
                root.update(output={"translation": result["translation"][:500]})

                # STT — span
                with root.start_observation(
                    name="stt",
                    input={
                        "filename":      filename,
                        "audio_size_kb": audio_in_size_kb,
                        "whisper_model": whisper_model,
                    },
                    # Note : SDK v4 utilise le temps réel du context manager
                    # (les timestamps historiques passent en metadata plus bas)
                ) as stt_span:
                    stt_span.update(
                        output={
                            "text":          result["source_text"][:1000],
                            "language":      result["language"],
                            "language_prob": result["language_prob"],
                        },
                        metadata={"latency_ms": result["latency_stt_ms"]},
                    )

                # LLM — generation (avec usage tokens + coût)
                with root.start_observation(
                    name="llm_translate",
                    as_type="generation",
                    model=llm_model,
                    model_parameters={"prompt_version": prompt_version, "target_lang": target_lang},
                    input=result.get("safe_input_text", result["source_text"])[:2000],
                ) as gen:
                    gen.update(
                        output=result["translation"][:2000],
                        usage_details={
                            "input":  prompt_tokens,
                            "output": completion_tokens,
                            "total":  total_tokens,
                        },
                        cost_details={"total": cost_usd},
                        metadata={
                            "start_time": result.get("llm_start_iso"),
                            "end_time":   result.get("llm_end_iso"),
                            "latency_ms": result["latency_llm_ms"],
                        },
                    )

                # TTS — span
                with root.start_observation(
                    name="tts",
                    input={"text": result["translation"][:1000], "lang": target_lang},
                ) as tts_span:
                    tts_span.update(
                        output={
                            "audio_size_kb": result.get("audio_out_size_kb"),
                            "content_type":  result["audio_content_type"],
                        },
                        metadata={"latency_ms": result["latency_tts_ms"]},
                    )

                # Scores associés à la trace
                for name, value in [
                    ("latency_total_ms", latency_total_ms),
                    ("latency_stt_ms",   result["latency_stt_ms"]),
                    ("latency_llm_ms",   result["latency_llm_ms"]),
                    ("latency_tts_ms",   result["latency_tts_ms"]),
                    ("language_prob",    result["language_prob"]),
                    ("cost_usd",         cost_usd),
                    ("total_tokens",     total_tokens),
                ]:
                    root.score_trace(name=name, value=float(value), comment=comment)

            # Force l'export avant retour de la requête (BatchSpanProcessor asynchrone)
            _lf.flush()
        except Exception as e:
            print(f"[pipeline] Langfuse v4 tracing warning: {e}", flush=True)

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
