"""
STT Service — Port 8001
POST /transcribe  : audio file -> texte transcrit
GET  /health      : statut du service
"""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

# Rendre le package src accessible
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flash_nlp.transcription.audio_utils import convert_to_wav
from flash_nlp.transcription.whisper_service import WhisperService

app = FastAPI(title="STT Service", version="1.0.0")

# Cache des modèles Whisper (chargés à la demande, gardés en mémoire)
_whisper_cache: dict[str, WhisperService] = {}

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "small")


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
    - **model** : modèle Whisper à utiliser (small, large-v3...)
    - **language** : langue source (fr, en, auto...)
    - **beam_size** : précision de la recherche (5 par défaut)
    """
    suffix = Path(file.filename).suffix.lower() if file.filename else ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(await file.read())
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path.replace(suffix, ".wav")

    try:
        convert_to_wav(tmp_in_path, tmp_wav_path)
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
