"""
Backend STT OpenAI Whisper API.

Alternative au Faster-Whisper self-hosted pour les cas ou la vitesse
prime sur le cout (env 0.006 USD / minute, mais latence 10-20x inferieure).

Bascule via STT_BACKEND=openai dans .env.
Modeles supportes : whisper-1 (defaut), gpt-4o-transcribe, gpt-4o-mini-transcribe.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_STT_URL   = "https://api.openai.com/v1/audio/transcriptions"


def is_available() -> bool:
    """Le backend est utilisable si la cle API est configuree."""
    return bool(OPENAI_API_KEY)


def transcribe_with_openai(
    audio_path: str,
    language: str = "fr",
) -> dict:
    """Transcrit un audio via l'API OpenAI Whisper.

    Retourne le meme format que Faster-Whisper local pour rester compatible
    avec le contrat de sortie du service STT.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquante — impossible d'utiliser le backend OpenAI STT.")

    lang = None if language == "auto" else language

    # verbose_json donne segments + duration comme faster-whisper
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f, "audio/wav")}
        data: dict[str, str] = {
            "model":           OPENAI_STT_MODEL,
            "response_format": "verbose_json",
        }
        if lang:
            data["language"] = lang

        with httpx.Client(timeout=600) as client:
            r = client.post(
                OPENAI_STT_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=files,
                data=data,
            )

    if not r.is_success:
        raise RuntimeError(f"OpenAI STT HTTP {r.status_code}: {r.text[:300]}")

    resp = r.json()

    # Normalisation du format pour matcher la sortie faster-whisper
    segments = [
        {
            "start": seg.get("start", 0),
            "end":   seg.get("end", 0),
            "text":  seg.get("text", ""),
        }
        for seg in resp.get("segments", [])
    ]

    return {
        "text":                 resp.get("text", ""),
        "language":             resp.get("language", lang or "fr"),
        # OpenAI ne renvoie pas de probabilite → on met 1.0 (certitude assumee)
        "language_probability": 1.0,
        "duration":             resp.get("duration", 0),
        "segments":             segments,
    }
