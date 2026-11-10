"""
STT Service — Port 8001
POST /transcribe  : audio file -> texte transcrit
GET  /health      : statut du service
"""

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from stt.whisper_service import WhisperService
from stt import backends, groq_backend, openai_backend
from stt import billing
from stt import uploads
from stt.auth import require_caller

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

app = FastAPI(title="STT Service", version="1.0.0")
# CORS utile en developpement local seulement (en production : meme origine via le proxy du site)
app.add_middleware(
    CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.middleware("http")(uploads.reject_oversized_body)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator(excluded_handlers=["/health", "/metrics"]).instrument(app).expose(app)

_whisper_cache: dict[str, WhisperService] = {}
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")

# STT_BACKEND : "local" (Faster-Whisper self-hosted), "openai" (Whisper API,
# ~0.006 USD/min) ou "groq" (meme modele Whisper, bien plus rapide et ~9x moins
# cher, avec repli sur OpenAI en cas d'echec).
# local reste utile pour la demo LLMOps ou pour minimiser les couts variables
# a fort volume constant.
STT_BACKEND = os.getenv("STT_BACKEND", "local").lower()


def convert_audio(src: str, dst: str) -> None:
    """Convertit n'importe quel audio en WAV 16kHz mono via ffmpeg.

    Robuste aux mauvaises extensions et aux fichiers corrompus :

    ▸ ESSAI 1 — options standard permissives :
      - probesize 100M / analyzeduration 100M : sniffing du format sur le contenu
        (défaut = 5 MB, insuffisant pour du M4A avec moov box en fin de fichier)
      - loudnorm : normalise le volume (audio faible → mieux transcrit par Whisper)

    ▸ ESSAI 2 (fallback si essai 1 échoue) — options de récupération d'erreur :
      - err_detect ignore_err : ignore les erreurs de header/frame corrompues
      - fflags +igndts+genpts : régénère les timestamps si corrompus
      - ignore_unknown : passe outre les codecs inconnus

    ▸ ÉCHEC final : lève RuntimeError avec le message ffmpeg pour que le
      pipeline le convertisse en 415 avec message user-friendly.
    """
    common_args = ["-y", "-analyzeduration", "100M", "-probesize", "100M"]
    output_args = [
        "-vn", "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",  # normalise volume audio
        "-c:a", "pcm_s16le", dst,
    ]

    # Essai 1 : options standard
    result = subprocess.run(
        ["ffmpeg"] + common_args + ["-i", src] + output_args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        return

    # Essai 2 : options de récupération d'erreur (fichiers corrompus, headers manquants)
    fallback_args = [
        "-err_detect", "ignore_err",
        "-fflags", "+igndts+genpts",
        "-ignore_unknown",
    ]
    result2 = subprocess.run(
        ["ffmpeg"] + common_args + fallback_args + ["-i", src] + output_args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result2.returncode == 0:
        print(f"[stt] convert_audio: recovered via fallback for {src}", flush=True)
        return

    # Les deux essais ont échoué → fichier vraiment inutilisable
    raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")


def get_whisper(model_name: str) -> WhisperService:
    if model_name not in _whisper_cache:
        svc = WhisperService()
        svc.load(model_name, device="cpu")
        _whisper_cache[model_name] = svc
    return _whisper_cache[model_name]


def _transcribe_blocking(
    tmp_in_path: str, tmp_wav_path: str, model: str, language: str, beam_size: int, diarize: bool,
) -> tuple[dict, str, str]:
    """Convertit puis transcrit : (resultat, modele utilise, fournisseur)."""
    convert_audio(tmp_in_path, tmp_wav_path)
    remote = backends.transcribe_remote(tmp_wav_path, language, diarize, STT_BACKEND)
    if remote is not None:
        return remote
    svc = get_whisper(model)
    result = svc.transcribe_wav_with_segments(
        wav_path=tmp_wav_path,
        language=None if language == "auto" else language,
        beam_size=beam_size,
    )
    return result, model, "local"


@app.get("/health")
async def health():
    """Async pour répondre immédiatement même si le thread pool est saturé
    par les transcriptions Whisper en cours."""
    return {
        "status":        "ok",
        "backend":       backends.active(STT_BACKEND),
        "loaded_models": list(_whisper_cache.keys()),
        "openai_model":  openai_backend.OPENAI_STT_MODEL if STT_BACKEND in ("openai", "groq") else None,
        "groq_model":    groq_backend.GROQ_STT_MODEL if STT_BACKEND == "groq" else None,
    }


@app.post("/transcribe")
async def transcribe(
    user_id: int | None = Depends(require_caller),
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form("fr"),
    beam_size: int = Form(5),
    diarize: bool = Form(False),
):
    """
    Transcrit un fichier audio (MP3, WAV, M4A...) en texte.

    - **file** : fichier audio
    - **model** : whisper model (small, large-v3...)
    - **language** : langue source (fr, en, auto)
    - **beam_size** : précision beam search
    """
    # Suffix .bin (générique) au lieu de l'extension client : force ffmpeg à
    # détecter le format sur le CONTENU du fichier, pas sur son nom. Résout
    # les cas où un fichier .mp3 contient en fait du M4A/AAC/WebM (enregistreurs
    # iOS/Android) → ffmpeg refuserait de le lire comme MP3 sinon.
    rate = await billing.rate_for(user_id, diarize) if user_id is not None else 0.0

    audio = await uploads.read_capped(file)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        tmp_in.write(audio)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + ".wav"

    try:
        # Conversion et transcription sont bloquantes (ffmpeg, appels HTTP, modele
        # local) : dans un thread, pour que le service traite plusieurs audios a la fois.
        result, used_model, backend = await asyncio.to_thread(
            _transcribe_blocking, tmp_in_path, tmp_wav_path, model, language, beam_size, diarize,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_wav_path).unlink(missing_ok=True)

    if user_id is not None:
        await billing.charge(user_id, result.get("duration") or 0, rate,
                             "meeting_diarization" if diarize else "meeting_transcription",
                             usage={"stt_backend": backend, "stt_model": used_model})

    return JSONResponse({
        "text":                 result["text"],
        "language":             result["language"],
        "language_probability": result["language_probability"],
        "duration":             result["duration"],
        "segments":             result["segments"],
        "speakers":             result.get("speakers"),  # None si pas de diarisation
        "model":                used_model,
        "backend":              backend,
    })
