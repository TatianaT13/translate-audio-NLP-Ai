"""
TTS Service — Port 8003
POST /synthesize : texte -> fichier WAV (MMS-TTS Meta)
GET  /health     : statut + langues disponibles
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"

app = FastAPI(title="TTS Service", version="1.0.0")

# Langues disponibles -> dossier du modèle
LANG_MODELS = {
    "en": MODELS_DIR / "mms-tts-eng",
    "uk": MODELS_DIR / "mms-tts-ukr",
}

# Cache des pipelines TTS
_tts_cache: dict = {}


def get_tts(lang: str):
    if lang in _tts_cache:
        return _tts_cache[lang]

    model_dir = LANG_MODELS.get(lang)
    if model_dir is None or not model_dir.exists():
        raise ValueError(f"Modèle TTS non disponible pour la langue : {lang}")

    from transformers import VitsModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = VitsModel.from_pretrained(str(model_dir))
    model.eval()

    _tts_cache[lang] = (tokenizer, model)
    return _tts_cache[lang]


class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "en"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_languages": list(LANG_MODELS.keys()),
        "loaded_models": list(_tts_cache.keys()),
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    """
    Synthétise un texte en audio WAV.

    - **text** : texte à lire
    - **lang** : langue (en, uk)

    Retourne un fichier audio WAV (16kHz mono).
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le champ text est vide.")
    if req.lang not in LANG_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Langue non supportée. Langues disponibles : {list(LANG_MODELS.keys())}",
        )

    try:
        import torch
        import scipy.io.wavfile as wav_io

        tokenizer, model = get_tts(req.lang)
        inputs = tokenizer(req.text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform

        audio_np = output.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate

        # Normaliser et convertir en int16
        audio_int16 = (audio_np / np.max(np.abs(audio_np)) * 32767).astype(np.int16)

        # Écrire en mémoire
        buf = io.BytesIO()
        wav_io.write(buf, sample_rate, audio_int16)
        buf.seek(0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesis.wav"},
    )
