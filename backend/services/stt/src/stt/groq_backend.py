"""
Backend STT Groq — Whisper large-v3-turbo.

Meme modele Whisper que l'API OpenAI, servi beaucoup plus vite (de l'ordre de
200 fois le temps reel) et moins cher (~0,04 USD par heure d'audio). L'API est
compatible avec celle d'OpenAI : meme requete, meme format de reponse.

Bascule via STT_BACKEND=groq dans .env. En cas d'echec, le service retombe sur
OpenAI s'il est configure (voir stt.backends).
"""
from __future__ import annotations

import os

from stt import openai_backend

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
GROQ_STT_URL   = "https://api.groq.com/openai/v1/audio/transcriptions"


def is_available() -> bool:
    return bool(GROQ_API_KEY)


def transcribe_with_groq(audio_path: str, language: str = "fr") -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY manquante — impossible d'utiliser le backend Groq STT.")
    return openai_backend.transcribe_compatible(
        url=GROQ_STT_URL, api_key=GROQ_API_KEY, model=GROQ_STT_MODEL,
        audio_path=audio_path, language=language, provider="Groq",
    )
