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

# Ajouter src/ au path AVANT les imports du package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import direct depuis le module (sans passer par __init__.py qui charge sounddevice)
from flash_nlp.transcription.whisper_service import WhisperService

app = FastAPI(title="STT Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_whisper_cache: dict[str, WhisperService] = {}
DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")


def convert_audio(src: str, dst: str) -> None:
    """Convertit n'importe quel audio en WAV 16kHz mono via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", dst],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")


def get_whisper(model_name: str) -> WhisperService:
    if model_name not in _whisper_cache:
        svc = WhisperService()
        svc.load(model_name, device="cpu")
        _whisper_cache[model_name] = svc
    return _whisper_cache[model_name]


@app.get("/health")
def health():
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
    suffix = Path(file.filename or "audio.mp3").suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await file.read())
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path.replace(suffix, ".wav")

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
