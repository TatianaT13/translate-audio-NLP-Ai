"""
Backend STT AssemblyAI — transcription + diarisation (identification des speakers).

Utilise quand le user demande explicitement diarize=true. AssemblyAI retourne
des segments avec speaker IDs (A, B, C...) et timestamps precis.

API : https://www.assemblyai.com/docs/
Pricing Nano : ~0.37 USD/h (~0.33 EUR/h)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE    = "https://api.assemblyai.com/v2"

# Modele AssemblyAI : "best" (qualite max), "nano" (econo)
# Nano = ~0.37 USD/h, Best = ~0.65 USD/h. Nano suffit pour la plupart des cas.
ASSEMBLYAI_MODEL = os.getenv("ASSEMBLYAI_MODEL", "nano")

# Mapping ISO → langue AssemblyAI
LANG_MAP = {
    "fr": "fr", "en": "en", "en_us": "en_us", "en_uk": "en_uk",
    "es": "es", "de": "de", "it": "it", "pt": "pt", "nl": "nl",
    "uk": "uk", "ru": "ru", "pl": "pl",
}


def is_available() -> bool:
    """Le backend est utilisable si la cle API est configuree."""
    return bool(ASSEMBLYAI_API_KEY)


def _upload_audio(audio_path: str) -> str:
    """Upload l'audio sur AssemblyAI et retourne l'URL temporaire."""
    with open(audio_path, "rb") as f:
        with httpx.Client(timeout=600) as client:
            r = client.post(
                f"{ASSEMBLYAI_BASE}/upload",
                headers={"authorization": ASSEMBLYAI_API_KEY},
                content=f.read(),
            )
    r.raise_for_status()
    return r.json()["upload_url"]


def _submit_transcription(audio_url: str, language: str = "fr", diarize: bool = True) -> str:
    """Lance le job de transcription + diarisation. Retourne le transcript_id."""
    body = {
        "audio_url":           audio_url,
        "speaker_labels":      diarize,
        "speech_model":        ASSEMBLYAI_MODEL,
        "format_text":         True,
        "punctuate":           True,
    }
    lang = LANG_MAP.get(language.lower()) if language and language != "auto" else None
    if lang:
        body["language_code"] = lang

    with httpx.Client(timeout=30) as client:
        r = client.post(
            f"{ASSEMBLYAI_BASE}/transcript",
            headers={"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"},
            json=body,
        )
    r.raise_for_status()
    return r.json()["id"]


def _poll_result(transcript_id: str, timeout_s: int = 1800) -> dict:
    """Poll le job jusqu'a completion. Timeout 30 min par defaut."""
    url = f"{ASSEMBLYAI_BASE}/transcript/{transcript_id}"
    start = time.time()
    delay = 2.0
    with httpx.Client(timeout=30) as client:
        while True:
            r = client.get(url, headers={"authorization": ASSEMBLYAI_API_KEY})
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise RuntimeError(f"AssemblyAI erreur: {data.get('error', 'unknown')}")
            if time.time() - start > timeout_s:
                raise RuntimeError(f"AssemblyAI timeout apres {timeout_s}s (status={status})")
            time.sleep(delay)
            delay = min(delay * 1.3, 10.0)  # backoff doux, max 10s


def transcribe_with_assemblyai(
    audio_path: str,
    language: str = "fr",
    diarize: bool = True,
) -> dict:
    """Transcrit avec diarisation. Retourne le format compatible pipeline.

    Format de sortie (compatible avec Whisper backend) :
    {
        "text":                 "...",                    # texte brut concatene
        "language":             "fr",
        "language_probability": 1.0,
        "duration":             123.4,                    # secondes
        "segments":             [                         # avec speaker IDs
            {"start": 0.0, "end": 3.2, "text": "...", "speaker": "A"},
            ...
        ],
        "speakers":             ["A", "B", "C"],          # liste des speakers detectes
    }
    """
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY manquante — impossible d'utiliser le backend AssemblyAI.")

    upload_url = _upload_audio(audio_path)
    transcript_id = _submit_transcription(upload_url, language=language, diarize=diarize)
    result = _poll_result(transcript_id)

    # Normalisation vers le format commun
    utterances = result.get("utterances") or []
    segments = [
        {
            "start":   u.get("start", 0) / 1000.0,  # ms → s
            "end":     u.get("end", 0) / 1000.0,
            "text":    u.get("text", ""),
            "speaker": u.get("speaker", "?"),
        }
        for u in utterances
    ]

    # Fallback si pas d'utterances (quand speaker_labels=False)
    if not segments:
        for w in (result.get("words") or []):
            segments.append({
                "start":   w.get("start", 0) / 1000.0,
                "end":     w.get("end", 0) / 1000.0,
                "text":    w.get("text", ""),
                "speaker": w.get("speaker", "?"),
            })

    speakers = sorted({s["speaker"] for s in segments if s.get("speaker")})

    return {
        "text":                 result.get("text", ""),
        "language":             result.get("language_code", language),
        "language_probability": 1.0,
        "duration":             (result.get("audio_duration") or 0),
        "segments":             segments,
        "speakers":             speakers,
    }
