"""
TTS Service — Port 8003

Deux backends disponibles (variable d'env TTS_BACKEND) :
  - "mistral"  : Voxtral TTS via API Mistral (par défaut si MISTRAL_API_KEY présent)
  - "local"    : MMS-TTS Meta (local, fallback)

Variables d'env :
  MISTRAL_API_KEY  : clé API Mistral
  MISTRAL_VOICE_ID : voice_id créé sur console.mistral.ai (requis pour Mistral)
  TTS_BACKEND      : "mistral" ou "local" (défaut: auto-détecté)
"""

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_VOICE_ID = os.getenv("MISTRAL_VOICE_ID", "")
TTS_BACKEND = os.getenv("TTS_BACKEND", "mistral" if MISTRAL_API_KEY else "local")

app = FastAPI(title="TTS Service", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Cache MMS local
_tts_cache: dict = {}
LANG_MODELS = {
    "en": MODELS_DIR / "mms-tts-eng",
    "uk": MODELS_DIR / "mms-tts-ukr",
}


# ---------------------------------------------------------------------------
# Backend Mistral Voxtral TTS
# ---------------------------------------------------------------------------

def synthesize_mistral(text: str, lang: str = "en") -> bytes:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY manquante dans .env")
    if not MISTRAL_VOICE_ID:
        raise ValueError("MISTRAL_VOICE_ID manquant dans .env — crée une voix sur console.mistral.ai")

    from mistralai.client import Mistral
    client = Mistral(api_key=MISTRAL_API_KEY)

    response = client.audio.speech.complete(
        model="voxtral-mini-tts-2603",
        input=text,
        voice_id=MISTRAL_VOICE_ID,
        response_format="mp3",
    )
    return base64.b64decode(response.audio_data)


# ---------------------------------------------------------------------------
# Backend MMS-TTS local (fallback)
# ---------------------------------------------------------------------------

def get_mms(lang: str):
    if lang in _tts_cache:
        return _tts_cache[lang]
    model_dir = LANG_MODELS.get(lang)
    if model_dir is None or not model_dir.exists():
        raise ValueError(f"Modèle MMS-TTS non disponible pour : {lang}")
    from transformers import VitsModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = VitsModel.from_pretrained(str(model_dir))
    model.eval()
    _tts_cache[lang] = (tokenizer, model)
    return _tts_cache[lang]


def synthesize_local(text: str, lang: str = "en") -> bytes:
    import torch
    import scipy.io.wavfile as wav_io

    tokenizer, model = get_mms(lang)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    audio_np = output.squeeze().cpu().numpy()
    audio_int16 = (audio_np / np.max(np.abs(audio_np)) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_io.write(buf, model.config.sampling_rate, audio_int16)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    lang: str = "en"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": TTS_BACKEND,
        "mistral_configured": bool(MISTRAL_API_KEY and MISTRAL_VOICE_ID),
        "local_languages": list(LANG_MODELS.keys()),
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Le champ text est vide.")

    try:
        if TTS_BACKEND == "mistral":
            audio_bytes = synthesize_mistral(req.text, req.lang)
            media_type = "audio/mpeg"
            filename = "synthesis.mp3"
        else:
            audio_bytes = synthesize_local(req.text, req.lang)
            media_type = "audio/wav"
            filename = "synthesis.wav"

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(audio_bytes)),
        },
    )
