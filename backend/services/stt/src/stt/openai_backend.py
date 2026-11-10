"""
Backend STT OpenAI Whisper API.

Alternative au Faster-Whisper self-hosted pour les cas ou la vitesse
prime sur le cout (env 0.006 USD / minute, mais latence 10-20x inferieure).

Bascule via STT_BACKEND=openai dans .env.
Modeles supportes : whisper-1 (defaut), gpt-4o-transcribe, gpt-4o-mini-transcribe.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_STT_URL   = "https://api.openai.com/v1/audio/transcriptions"


# L'API refuse les fichiers de plus de 25 Mo. Le WAV 16 kHz mono qu'on lui
# envoie pese ~1,9 Mo/min : au-dela de 13 min il faut le compresser.
OPENAI_MAX_BYTES = 24 * 1024 * 1024


def _fit_upload(audio_path: str) -> tuple[str, str]:
    """(chemin, type MIME) du fichier a envoyer, sous la limite de l'API.

    Sous la limite : le WAV tel quel. Au-dela : MP3 mono 32 kbps (~0,24 Mo/min,
    soit ~1 h 40 d'audio), suffisant pour de la parole.
    """
    if os.path.getsize(audio_path) <= OPENAI_MAX_BYTES:
        return audio_path, "audio/wav"
    mp3_path = audio_path + ".mp3"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "32k", mp3_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        Path(mp3_path).unlink(missing_ok=True)
        raise RuntimeError(f"Compression audio impossible : {result.stderr.decode(errors='replace')[-300:]}")
    if os.path.getsize(mp3_path) > OPENAI_MAX_BYTES:
        Path(mp3_path).unlink(missing_ok=True)
        raise RuntimeError("Audio trop long pour la transcription (limite : environ 1 h 40).")
    return mp3_path, "audio/mpeg"


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
    return transcribe_compatible(
        url=OPENAI_STT_URL, api_key=OPENAI_API_KEY, model=OPENAI_STT_MODEL,
        audio_path=audio_path, language=language, provider="OpenAI",
    )


def transcribe_compatible(
    url: str, api_key: str, model: str, audio_path: str, language: str, provider: str,
) -> dict:
    """Transcription via une API compatible OpenAI (OpenAI, Groq)."""
    lang = None if language == "auto" else language

    upload_path, mime = _fit_upload(audio_path)
    try:
        # verbose_json donne segments + duration comme faster-whisper
        with open(upload_path, "rb") as f:
            files = {"file": (Path(upload_path).name, f, mime)}
            data: dict[str, str] = {
                "model":           model,
                "response_format": "verbose_json",
            }
            if lang:
                data["language"] = lang

            with httpx.Client(timeout=600) as client:
                r = client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data=data,
                )
    finally:
        if upload_path != audio_path:
            Path(upload_path).unlink(missing_ok=True)

    if not r.is_success:
        raise RuntimeError(f"{provider} STT HTTP {r.status_code}: {r.text[:300]}")

    resp = r.json()

    # Normalisation du format pour matcher la sortie faster-whisper
    segments = [
        {
            "start": seg.get("start", 0),
            "end":   seg.get("end", 0),
            "text":  seg.get("text", ""),
        }
        for seg in resp.get("segments") or []
    ]

    return {
        "text":                 resp.get("text", ""),
        "language":             resp.get("language", lang or "fr"),
        # Ces API ne renvoient pas de probabilite → on met 1.0 (certitude assumee)
        "language_probability": 1.0,
        "duration":             resp.get("duration", 0),
        "segments":             segments,
    }
