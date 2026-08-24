"""
STT Service — Port 8001
POST /transcribe  : audio file -> texte transcrit
GET  /health      : statut du service
"""

import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from stt.whisper_service import WhisperService

app = FastAPI(title="STT Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator(excluded_handlers=["/health", "/metrics"]).instrument(app).expose(app)

_whisper_cache: dict[str, WhisperService] = {}
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")


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


@app.get("/health")
async def health():
    """Async pour répondre immédiatement même si le thread pool est saturé
    par les transcriptions Whisper en cours."""
    return {"status": "ok", "loaded_models": list(_whisper_cache.keys())}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form("fr"),
    beam_size: int = Form(5),
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
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        tmp_in.write(await file.read())
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + ".wav"

    try:
        convert_audio(tmp_in_path, tmp_wav_path)
        svc = get_whisper(model)
        lang_arg = None if language == "auto" else language
        result = svc.transcribe_wav_with_segments(
            wav_path=tmp_wav_path,
            language=lang_arg,
            beam_size=beam_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_wav_path).unlink(missing_ok=True)

    return JSONResponse({
        "text":                 result["text"],
        "language":             result["language"],
        "language_probability": result["language_probability"],
        "duration":             result["duration"],
        "segments":             result["segments"],
        "model":                model,
    })
