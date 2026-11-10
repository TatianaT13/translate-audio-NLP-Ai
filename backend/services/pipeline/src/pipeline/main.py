"""
Pipeline Service — Port 8000
Orchestrateur Langchain LCEL : Audio → STT → LLM → TTS

POST /process  : audio file → { source_text, translation, audio_b64, latencies }
GET  /health   : statut + URLs des services aval
"""

import asyncio
import base64
import hmac
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from langchain_core.runnables import RunnableLambda

from pipeline import jobs as _jobs
from pipeline import uploads
from pipeline.prompt_guard import check_input, check_output, sandbox_user_text

load_dotenv()

STT_URL = os.getenv("STT_URL", "http://localhost:8001")
LLM_URL = os.getenv("LLM_URL", "http://localhost:8002")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8003")

# Pour les crédits : appel service-to-service vers Gateway
GATEWAY_URL           = os.getenv("GATEWAY_URL", "http://gateway:8004")
GATEWAY_SERVICE_TOKEN = os.getenv("GATEWAY_SERVICE_TOKEN", "")
JWT_SECRET            = os.getenv("JWT_SECRET", "")
JWT_ALGO              = "HS256"
# STT et LLM exigent ce jeton pour les appels internes (sans JWT utilisateur)
_SERVICE_HEADERS      = {"x-service-token": GATEWAY_SERVICE_TOKEN}

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

# Sentry : inactif sans SENTRY_DSN. Ni corps de requete ni variables locales :
# send_default_pii=False ne suffit pas, les variables des frames contiennent
# le corps (mots de passe, transcriptions) et les en-tetes (JWT, IP).
if os.getenv("SENTRY_DSN"):
    import sentry_sdk
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        send_default_pii=False,
        max_request_body_size="never",
        include_local_variables=False,
        traces_sample_rate=0.0,
    )

app = FastAPI(title="Pipeline Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.middleware("http")(uploads.reject_oversized_body)

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
            headers=_SERVICE_HEADERS,
            files={"file": (state["filename"], state["audio_bytes"], "audio/mpeg")},
            data={
                "model":    state["whisper_model"],
                "language": "fr",
                "diarize":  str(state.get("diarize", False)).lower(),
            },
        )
    resp.raise_for_status()
    data = resp.json()

    text = data.get("text", "").strip()
    lang_prob = data.get("language_probability", 0)
    # Diarisation : conserve les segments avec speakers pour les restituer en sortie
    state["stt_segments"] = data.get("segments") or []
    state["stt_duration"] = data.get("duration") or 0  # secondes, base de la facturation
    state["stt_backend"]  = data.get("backend") or "unknown"
    state["stt_model"]    = data.get("model")
    state["stt_speakers"] = data.get("speakers")

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
        # Message user-friendly différencié selon la raison du blocage
        if guard.reason == "text_too_long":
            from pipeline.prompt_guard import MAX_INPUT_CHARS
            detail = (
                f"Audio trop long : la transcription fait {len(text):,} caractères "
                f"(limite {MAX_INPUT_CHARS:,}). Découpe ton audio en segments plus courts "
                f"ou augmente MAX_INPUT_CHARS côté serveur."
            )
        else:
            detail = (
                "Contenu audio suspect détecté. Cette plateforme traduit uniquement "
                "des messages d'information routière — merci de réessayer avec un audio approprié."
            )
        raise HTTPException(status_code=422, detail=detail)

    return {
        **state,
        "source_text": text,
        "language": data.get("language", "fr"),
        "language_prob": lang_prob,
        "latency_stt_ms": round((time.perf_counter() - t0) * 1000),
        "stt_start_iso":  stt_start_iso,
        "stt_end_iso":    datetime.now(timezone.utc).isoformat(),
    }


# Taille max d'un chunk envoyé au LLM. Au-delà, on chunke automatiquement.
# 3000 chars ≈ 750 tokens = petit contexte, traduit vite, coût minimal.
LLM_CHUNK_MAX_CHARS = int(os.getenv("LLM_CHUNK_MAX_CHARS", "3000"))

# Appels simultanes vers les services LLM / TTS pour un meme job. Borne pour
# rester sous les limites de debit des fournisseurs (OpenAI, Mistral).
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "6"))
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "3"))


async def _map_concurrent(fn, items: list, limit: int) -> list:
    """Applique `fn` a chaque item avec au plus `limit` appels simultanes.

    Les resultats sont dans l'ordre des items. A la premiere erreur, les appels
    restants sont annules (inutile de payer la suite d'un job qui echoue) et
    l'erreur d'origine est relancee telle quelle.
    """
    sem = asyncio.Semaphore(max(1, limit))

    async def run(item):
        async with sem:
            return await fn(item)

    tasks = [asyncio.create_task(run(item)) for item in items]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise


def _split_into_chunks(text: str, max_chars: int = LLM_CHUNK_MAX_CHARS) -> list[str]:
    """Découpe le texte en chunks de max_chars caractères, aux frontières de phrases.

    Préserve la sémantique en cassant au niveau des ponctuations fortes (. ! ?)
    plutôt qu'au milieu d'un mot. Overlap zéro (les traductions sont assez
    autonomes pour ne pas nécessiter de contexte croisé).
    """
    if len(text) <= max_chars:
        return [text]

    import re
    # Split par phrases (garde le délimiteur avec un lookbehind)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # Si une phrase seule dépasse max_chars (rare mais possible), on la coupe brutalement
        while len(sentence) > max_chars:
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks


async def _translate_chunk(
    client: httpx.AsyncClient,
    text: str,
    target_lang: str,
    llm_model: str,
    prompt_version: str,
) -> dict:
    """Traduit un chunk unique via le service LLM. Renvoie le dict de réponse."""
    resp = await client.post(
        f"{LLM_URL}/translate",
        headers=_SERVICE_HEADERS,
        json={
            "text": sandbox_user_text(text),
            "target_lang": target_lang,
            "model": llm_model,
            "prompt_version": prompt_version,
        },
    )
    resp.raise_for_status()
    return resp.json()


@_trace(name="llm_translate")
async def _llm_step(state: dict) -> dict:
    """Étape 2 : traduction texte → texte traduit via LLM Service.

    Chunking automatique si le texte source dépasse LLM_CHUNK_MAX_CHARS
    (default 3000). Les chunks sont traduits en parallèle (LLM_CONCURRENCY)
    puis joints dans l'ordre.
    """
    from datetime import datetime, timezone
    t0 = time.perf_counter()
    llm_start_iso = datetime.now(timezone.utc).isoformat()

    source_text = state["source_text"]

    # Diarisation : traduire segment par segment pour preserver les speakers
    translation_segments = None
    stt_segments = state.get("stt_segments") or []
    has_diarize = (
        state.get("diarize")
        and stt_segments
        and any(s.get("speaker") for s in stt_segments)
    )

    if has_diarize:
        # Un appel LLM par segment → preserve le mapping speaker↔traduction
        segs = [seg for seg in stt_segments if (seg.get("text") or "").strip()]
        texts = [seg["text"].strip() for seg in segs]
        chunks_count = len(stt_segments)
    else:
        segs = []
        texts = _split_into_chunks(source_text, LLM_CHUNK_MAX_CHARS)
        chunks_count = len(texts)
    if len(texts) > 1:
        print(f"[pipeline] LLM: {len(source_text)} chars → {len(texts)} appels "
              f"({LLM_CONCURRENCY} en parallèle)", flush=True)

    async with httpx.AsyncClient(timeout=300) as client:
        results = await _map_concurrent(
            lambda text: _translate_chunk(
                client, text, state["target_lang"], state["llm_model"], state["prompt_version"],
            ),
            texts,
            LLM_CONCURRENCY,
        )

    translation = " ".join(data["translation"] for data in results)
    prompt_tokens     = sum(data.get("prompt_tokens",     0)   for data in results)
    completion_tokens = sum(data.get("completion_tokens", 0)   for data in results)
    total_tokens      = sum(data.get("total_tokens",      0)   for data in results)
    cost_usd          = sum(data.get("cost_usd",          0.0) for data in results)

    if has_diarize:
        translation_segments = [
            {
                "speaker":     seg.get("speaker"),
                "start":       seg.get("start"),
                "end":         seg.get("end"),
                "original":    text,
                "translation": data["translation"],
            }
            for seg, text, data in zip(segs, texts, results)
        ]

    # ── Garde-fou : post-check sur la sortie LLM ─────────────────────────────
    guard = check_output(translation, source_text)
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
        "translation_segments": translation_segments,
        "latency_llm_ms": round((time.perf_counter() - t0) * 1000),
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
        "cost_usd":          round(cost_usd, 6),
        "llm_start_iso":     llm_start_iso,
        "llm_end_iso":       datetime.now(timezone.utc).isoformat(),
        "safe_input_text":   source_text[:2000],
        "n_chunks":          chunks_count,
    }


# Taille max d'un chunk envoyé au TTS. Voxtral limite ≈ 5000 chars par appel.
# On chunke + concatène les audios pour supporter les longues traductions.
TTS_CHUNK_MAX_CHARS = int(os.getenv("TTS_CHUNK_MAX_CHARS", "3000"))


@_trace(name="tts_synthesize")
async def _tts_step(state: dict) -> dict:
    """Étape 3 : synthèse vocale texte traduit → audio via TTS Service.

    Chunking automatique si la traduction dépasse TTS_CHUNK_MAX_CHARS.
    Les audios chunks sont concaténés en un seul buffer côté serveur.
    """
    from datetime import datetime, timezone
    t0 = time.perf_counter()
    tts_start_iso = datetime.now(timezone.utc).isoformat()
    translation = state["translation"]
    tts_chunks = _split_into_chunks(translation, TTS_CHUNK_MAX_CHARS)
    if len(tts_chunks) > 1:
        print(f"[pipeline] TTS chunking: {len(translation)} chars → {len(tts_chunks)} chunks", flush=True)

    async with httpx.AsyncClient(timeout=300) as client:

        async def synthesize(chunk: str) -> httpx.Response:
            resp = await client.post(
                f"{TTS_URL}/synthesize",
                json={"text": chunk, "lang": state["target_lang"]},
            )
            # Mistral Voxtral peut renvoyer "guardrail_violation" sur du contenu sensible.
            # Le TTS service ré-emballe les guardrails Mistral en 500 avec le body en clair
            if not resp.is_success and "guardrail" in resp.text.lower():
                print("[pipeline] TTS guardrail (Mistral) — synthèse refusée", flush=True)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Synthèse vocale refusée par le fournisseur TTS (politique de contenu Mistral). "
                        "La traduction texte reste disponible — l'audio n'a pas pu être généré."
                    ),
                )
            resp.raise_for_status()
            return resp

        responses = await _map_concurrent(synthesize, tts_chunks, TTS_CONCURRENCY)

    audio_parts = [resp.content for resp in responses]
    # Voxtral renvoie audio/mpeg, MMS local renvoie audio/wav — on garde le premier
    content_type_seen = responses[0].headers.get("content-type", "audio/mpeg")

    # Concaténation naïve des audio bytes (fonctionne pour MP3 et WAV même si
    # techniquement pas parfait pour WAV — un vrai fix demanderait pydub/ffmpeg).
    audio_bytes = b"".join(audio_parts)
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return {
        **state,
        "audio_b64": audio_b64,
        "audio_content_type": content_type_seen,
        "latency_tts_ms": round((time.perf_counter() - t0) * 1000),
        "audio_out_size_kb": round(len(audio_bytes) / 1024, 1),
        "tts_start_iso":   tts_start_iso,
        "tts_end_iso":     datetime.now(timezone.utc).isoformat(),
        "n_tts_chunks":    len(tts_chunks),
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


def _authenticate(authorization: str | None, x_service_token: str | None) -> int | None:
    """user_id pour un JWT valide, None pour un appel interne (jeton de service).

    Tout autre appel est refuse : sans ce controle, /pipeline etant public,
    n'importe qui pourrait traduire sans compte et sans consommer de credits.
    """
    if _is_service_call(x_service_token):
        return None
    user_id = _decode_user_id(authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentification requise")
    return user_id


def _is_service_call(x_service_token: str | None) -> bool:
    # La valeur par defaut du docker-compose est publique (dans le repo) : jamais acceptee.
    if not GATEWAY_SERVICE_TOKEN or GATEWAY_SERVICE_TOKEN.startswith("change-me"):
        return False
    return bool(x_service_token) and hmac.compare_digest(x_service_token, GATEWAY_SERVICE_TOKEN)


def _decode_user_id(auth_header: str | None) -> int | None:
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.removeprefix("Bearer ").strip()
    if not JWT_SECRET or not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None


def _estimate_credits_from_filesize(audio_bytes: int, with_tts: bool = True, with_diarize: bool = False) -> int:
    """Estime les credits requis AVANT d'avoir la duree reelle.

    Approximation : 1 Mo de MP3 ≈ 1 min a 128kbps mono, ≈ 2 min a 64kbps.
    On prend l'estimation large (64kbps mono = 8 KB/s) pour ne pas sous-estimer.
    Pondération : TTS = +2/min, Diarisation = +0.5/min, Transcription = +1/min.
    """
    estimated_minutes = max(audio_bytes / (8 * 1024 * 60), 1)
    rate = (2 if with_tts else 1) + (0.5 if with_diarize else 0)
    return max(1, int(estimated_minutes * rate + 0.999))


def _billable_seconds(result: dict, audio_bytes: int) -> float:
    """Duree d'audio facturee, en secondes.

    Celle mesuree par le STT ; a defaut la fin du dernier segment ; en dernier
    recours une estimation sur la taille du fichier (juste pour du MP3 64 kbps
    seulement : un WAV serait facture 20 fois trop).
    """
    duration = result.get("stt_duration") or 0
    if not duration:
        duration = max((seg.get("end") or 0 for seg in result.get("stt_segments") or []), default=0)
    if not duration:
        duration = audio_bytes / (8 * 1024)
    return float(duration)


def _usage_meta(result: dict, duration_sec: float) -> dict:
    """Volumes consommes par le job, enregistres avec le debit. La Gateway en
    deduit le cout fournisseur reel (les tarifs sont chez elle : gateway.costs)."""
    return {
        "seconds":      round(duration_sec, 1),
        "stt_backend":  result.get("stt_backend") or "unknown",
        "stt_model":    result.get("stt_model"),
        "tts_chars":    len(result.get("translation") or ""),
        # Voxtral renvoie du MP3, le MMS local du WAV
        "tts_backend":  "local" if "wav" in (result.get("audio_content_type") or "") else "mistral",
        "llm_cost_usd": result.get("cost_usd") or 0.0,
    }


async def _reserve_credits(user_id: int, amount: int, kind: str = "processing") -> str:
    """Reserve les credits estimes aupres de la Gateway ; retourne l'id de reservation.

    402 si le solde disponible ne suffit pas. 503 si la Gateway ne repond pas :
    sans reservation on ne lance rien, sinon une panne de la Gateway offrirait
    des traitements gratuits.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                f"{GATEWAY_URL}/billing/reserve",
                json={"user_id": user_id, "kind": kind, "amount": amount},
                headers=_SERVICE_HEADERS,
            )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[pipeline] reserve impossible pour user {user_id}: {e!r}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Service de facturation indisponible. Réessayez dans quelques instants.",
        )
    if not data.get("ok"):
        raise HTTPException(
            status_code=402,
            detail={
                "reason":  "credits_insufficient",
                "balance": data.get("balance"),
                "needed":  data.get("needed"),
                "tier":    data.get("tier"),
            },
        )
    return data["hold_id"]


_SETTLE_RETRY_DELAYS = (1, 3)  # secondes entre les tentatives


async def _settle_credits(hold_id: str, amount: int, reason: str, meta: dict | None = None) -> None:
    """Debite le cout reel et leve la reservation. Ne leve jamais : le resultat
    est deja calcule, on ne le refuse pas au client pour une erreur de facturation.

    Reessaie en cas d'echec (cote Gateway, un settle rejoue est sans effet).
    Echec definitif : la reservation expire seule, le job n'est pas facture —
    d'ou la remontee Sentry.
    """
    last_error = ""
    for delay in (0, *_SETTLE_RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(
                    f"{GATEWAY_URL}/billing/settle",
                    json={"hold_id": hold_id, "amount": amount, "reason": reason, "meta": meta},
                    headers=_SERVICE_HEADERS,
                )
            if r.status_code == 200:
                return
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_error = repr(e)
    msg = f"[pipeline] settle abandonne, job NON FACTURE ({amount} credits, {reason}): {last_error}"
    print(msg, flush=True)
    if os.getenv("SENTRY_DSN"):
        import sentry_sdk
        sentry_sdk.capture_message(msg, level="error")


async def _release_credits(hold_id: str | None) -> None:
    """Annule la reservation d'un job echoue. Au pire elle expire seule."""
    if not hold_id:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{GATEWAY_URL}/billing/release",
                json={"hold_id": hold_id},
                headers=_SERVICE_HEADERS,
            )
    except Exception as e:
        print(f"[pipeline] release echoue pour {hold_id} (expirera seule): {e!r}", flush=True)


# ── Limites de charge ────────────────────────────────────────────────────────
# Chaque job garde son audio en memoire et lance des appels en parallele vers
# les fournisseurs. Sans plafond, un afflux de demandes sature la RAM du serveur
# et les quotas des API : on borne donc le nombre de jobs en cours, les suivants
# attendent leur tour.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "8"))
MAX_JOBS_PER_USER   = int(os.getenv("MAX_JOBS_PER_USER", "2"))

_job_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_user_jobs: dict[int, int] = {}   # user_id -> jobs en cours ou en attente


def _enter_user(user_id: int | None) -> None:
    """Compte un job de plus pour ce compte ; 429 s'il en a deja trop."""
    if user_id is None:
        return
    if _user_jobs.get(user_id, 0) >= MAX_JOBS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Tu as déjà {MAX_JOBS_PER_USER} traitements en cours. Attends la fin de l'un d'eux.",
        )
    _user_jobs[user_id] = _user_jobs.get(user_id, 0) + 1


def _leave_user(user_id: int | None) -> None:
    if user_id is None:
        return
    remaining = _user_jobs.get(user_id, 0) - 1
    if remaining > 0:
        _user_jobs[user_id] = remaining
    else:
        _user_jobs.pop(user_id, None)


async def _job_owner(
    authorization: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
):
    """Dependance : authentifie l'appelant et lui compte un job le temps de la requete."""
    user_id = _authenticate(authorization, x_service_token)
    _enter_user(user_id)
    try:
        yield user_id
    finally:
        _leave_user(user_id)


async def _run_pipeline(initial_state: dict) -> dict:
    """Execute la chaine STT → LLM → TTS quand un creneau se libere."""
    async with _job_slots:
        return await pipeline_chain.ainvoke(initial_state)


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    target_lang: str = Form("en"),
    llm_model: str = Form("openai/gpt-4o-mini"),
    prompt_version: str = Form("v1.1"),
    whisper_model: str = Form("small"),
    diarize: bool = Form(False),
    user_id: int | None = Depends(_job_owner),
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

    audio_bytes = await uploads.read_capped(file)
    filename = file.filename or "audio.mp3"

    # ── Reservation des credits (si JWT fourni) ───────────────────────────
    # On reserve une estimation basee sur la taille du fichier : 402 si le solde
    # disponible ne suffit pas, 503 si la Gateway ne repond pas. A la fin, la
    # reservation est remplacee par le debit reel (vraie duree), ou annulee si
    # le job echoue.
    hold_id = None
    if user_id is not None:
        estimated_credits = _estimate_credits_from_filesize(
            len(audio_bytes), with_tts=True, with_diarize=diarize,
        )
        hold_id = await _reserve_credits(user_id, estimated_credits, kind="processing")

    initial_state = {
        "audio_bytes": audio_bytes,
        "filename": filename,
        "target_lang": target_lang,
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "whisper_model": whisper_model,
        "diarize": diarize,
    }

    t_total = time.perf_counter()
    try:
        result = await _run_pipeline(initial_state)
    except HTTPException:
        # Erreurs déjà formatées par les étapes (garde-fou, TTS guardrail, etc.)
        await _release_credits(hold_id)
        raise
    except httpx.HTTPStatusError as e:
        await _release_credits(hold_id)
        import traceback
        print(f"[pipeline] HTTPStatusError: {e.response.status_code} {e.response.text}", flush=True)
        traceback.print_exc()
        # Détection des erreurs ffmpeg courantes → message user-friendly
        err_text = e.response.text.lower()
        if "ffmpeg" in err_text and ("invalid data" in err_text or "misdetection" in err_text or "failed to find" in err_text):
            raise HTTPException(
                status_code=415,
                detail=(
                    "Format audio non reconnu. Le fichier n'est pas un MP3, WAV ou M4A valide "
                    "(souvent : extension .mp3 mais contenu différent, ou fichier corrompu). "
                    "Réencode-le avec un outil comme https://cloudconvert.com/mp3-converter, "
                    "ou utilise le bouton « Lancer la démo » pour tester."
                ),
            )
        raise HTTPException(status_code=502, detail=f"Service error: {e.response.text}")
    except Exception as e:
        await _release_credits(hold_id)
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

                # STT — child span (LangfuseSpan n'est pas context manager en v4,
                # on doit appeler .end() explicitement)
                stt_span = root.start_observation(
                    name="stt",
                    input={
                        "filename":      filename,
                        "audio_size_kb": audio_in_size_kb,
                        "whisper_model": whisper_model,
                    },
                )
                stt_span.update(
                    output={
                        "text":          result["source_text"][:1000],
                        "language":      result["language"],
                        "language_prob": result["language_prob"],
                    },
                    metadata={"latency_ms": result["latency_stt_ms"]},
                )
                stt_span.end()

                # LLM — generation
                gen = root.start_observation(
                    name="llm_translate",
                    as_type="generation",
                    model=llm_model,
                    model_parameters={"prompt_version": prompt_version, "target_lang": target_lang},
                    input=result.get("safe_input_text", result["source_text"])[:2000],
                )
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
                gen.end()

                # TTS — child span
                tts_span = root.start_observation(
                    name="tts",
                    input={"text": result["translation"][:1000], "lang": target_lang},
                )
                tts_span.update(
                    output={
                        "audio_size_kb": result.get("audio_out_size_kb"),
                        "content_type":  result["audio_content_type"],
                    },
                    metadata={"latency_ms": result["latency_tts_ms"]},
                )
                tts_span.end()

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

    # ── Consomme les credits reels (duree audio mesuree par le STT) ──
    if hold_id is not None:
        duration_sec = _billable_seconds(result, len(audio_bytes))
        rate = 2 + (0.5 if diarize else 0)  # 2 crédits/min TTS, +0.5 si diarisation
        credits_real = max(1, int((duration_sec / 60) * rate + 0.999))
        reason = (
            f"pipeline_process:{int(duration_sec)}s:"
            + ("audio_translated_diarized" if diarize else "audio_translated")
        )
        await _settle_credits(hold_id, credits_real, reason, _usage_meta(result, duration_sec))

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
        # Diarisation : None si pas demandee
        "translation_segments": result.get("translation_segments"),
        "speakers":             result.get("stt_speakers"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Jobs async — pour les gros audios (>10 Mo ou >5 min) le client peut submit
# le travail puis poller /jobs/{id} au lieu de laisser l'HTTP ouverte.
# Avantage : pas de timeout cote proxy / nginx, UX avec barre de progression.
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_job(
    job_id:         str,
    audio_bytes:    bytes,
    filename:       str,
    target_lang:    str,
    llm_model:      str,
    prompt_version: str,
    whisper_model:  str,
    diarize:        bool,
    hold_id:        str | None,
    user_id:        int | None,
) -> None:
    """Executeur de job en background. Met a jour le store a chaque etape."""
    try:
        await _jobs.store.update(job_id, status="running", progress=5)

        initial_state = {
            "audio_bytes":    audio_bytes,
            "filename":       filename,
            "target_lang":    target_lang,
            "llm_model":      llm_model,
            "prompt_version": prompt_version,
            "whisper_model":  whisper_model,
            "diarize":        diarize,
        }
        t_total = time.perf_counter()

        # On ne peut pas trivialement observer la progression intra-chain
        # (STT→LLM→TTS), donc on affiche des paliers coherents : 5 → 85 au
        # lancement du pipeline, 90 apres completion, 100 apres consume credits.
        await _jobs.store.update(job_id, progress=15)
        result = await _run_pipeline(initial_state)
        await _jobs.store.update(job_id, progress=85)

        latency_total_ms = round((time.perf_counter() - t_total) * 1000)

        # Consomme les credits reels (meme logique que /process)
        if hold_id is not None:
            duration_sec = _billable_seconds(result, len(audio_bytes))
            rate = 2 + (0.5 if diarize else 0)
            credits_real = max(1, int((duration_sec / 60) * rate + 0.999))
            reason = (
                f"pipeline_process_async:{int(duration_sec)}s:"
                + ("audio_translated_diarized" if diarize else "audio_translated")
            )
            await _settle_credits(hold_id, credits_real, reason, _usage_meta(result, duration_sec))
            hold_id = None  # regle : ne plus l'annuler si la suite echoue

        await _jobs.store.update(job_id, progress=95)

        response = {
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
            "translation_segments": result.get("translation_segments"),
            "speakers":             result.get("stt_speakers"),
        }
        await _jobs.store.update(job_id, status="done", progress=100, result=response)
    except HTTPException as e:
        await _release_credits(hold_id)
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)
        await _jobs.store.update(job_id, status="error", error=detail)
    except Exception as e:
        await _release_credits(hold_id)
        import traceback; traceback.print_exc()
        if os.getenv("SENTRY_DSN"):
            import sentry_sdk
            sentry_sdk.capture_exception(e)
        await _jobs.store.update(job_id, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        _leave_user(user_id)


@app.post("/process-async", status_code=202)
async def process_async(
    file: UploadFile = File(...),
    target_lang: str = Form("en"),
    llm_model: str = Form("openai/gpt-4o-mini"),
    prompt_version: str = Form("v1.1"),
    whisper_model: str = Form("small"),
    diarize: bool = Form(False),
    authorization: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
):
    """Submit un job async. Retourne un job_id a poller via GET /jobs/{job_id}.

    Les gros fichiers (>10 Mo ou >5 min d'audio) devraient utiliser cet endpoint
    pour eviter les timeouts reverse-proxy. Le client affiche une barre de
    progression base sur Job.progress (0-100).
    """
    if llm_model in _LEGACY_MODEL_ALIASES:
        llm_model = _LEGACY_MODEL_ALIASES[llm_model]

    audio_bytes = await uploads.read_capped(file)
    filename    = file.filename or "audio.mp3"
    user_id     = _authenticate(authorization, x_service_token)

    # Le job survit a la requete : son compteur est libere par _execute_job,
    # ou ici si le job n'a pas pu etre lance.
    _enter_user(user_id)
    try:
        # Reservation des credits (meme logique que /process)
        hold_id = None
        if user_id is not None:
            estimated = _estimate_credits_from_filesize(len(audio_bytes), with_tts=True, with_diarize=diarize)
            hold_id = await _reserve_credits(user_id, estimated, kind="processing")

        # Lance le cleanup au premier job (idempotent)
        await _jobs.store.start_cleanup()

        job = await _jobs.store.create(user_id=user_id)
    except BaseException:
        _leave_user(user_id)
        raise
    asyncio.create_task(_execute_job(
        job_id=job.id,
        audio_bytes=audio_bytes,
        filename=filename,
        target_lang=target_lang,
        llm_model=llm_model,
        prompt_version=prompt_version,
        whisper_model=whisper_model,
        diarize=diarize,
        hold_id=hold_id,
        user_id=user_id,
    ))
    return {"job_id": job.id, "status": "pending"}


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    authorization: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
):
    """Etat d'un job, lisible uniquement par son proprietaire (ou un appel interne)."""
    caller_id = _authenticate(authorization, x_service_token)
    job = await _jobs.store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable (expire ou id inconnu)")
    if caller_id is not None and job.user_id is not None and caller_id != job.user_id:
        raise HTTPException(status_code=403, detail="Pas le proprietaire de ce job")

    response = {
        "job_id":   job.id,
        "status":   job.status,
        "progress": job.progress,
    }
    if job.status == "done" and job.result:
        response["result"] = job.result
    if job.status == "error" and job.error:
        response["error"] = job.error
    return response
