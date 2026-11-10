"""Choix du fournisseur de transcription.

Module sans dependance a faster-whisper : la selection reste testable sans
charger de modele.
"""
from __future__ import annotations

from stt import assemblyai_backend, groq_backend, openai_backend


def active(preferred: str) -> str:
    """Fournisseur qui servira une transcription simple avec ce reglage."""
    if preferred == "assemblyai" and assemblyai_backend.is_available():
        return "assemblyai"
    if preferred == "groq" and groq_backend.is_available():
        return "groq"
    if preferred in ("groq", "openai") and openai_backend.is_available():
        return "openai"
    return "local"


def transcribe_remote(wav_path: str, language: str, diarize: bool, preferred: str) -> tuple[dict, str, str] | None:
    """Transcrit via un fournisseur externe : (resultat, modele, fournisseur).

    None si aucun fournisseur n'est configure pour ce reglage : l'appelant
    utilise alors Whisper local.
    """
    # Identification des intervenants : seul AssemblyAI la fournit
    if (diarize or preferred == "assemblyai") and assemblyai_backend.is_available():
        result = assemblyai_backend.transcribe_with_assemblyai(
            audio_path=wav_path, language=language, diarize=diarize,
        )
        return result, f"assemblyai/{assemblyai_backend.ASSEMBLYAI_MODEL}", "assemblyai"

    if preferred == "groq" and groq_backend.is_available():
        try:
            result = groq_backend.transcribe_with_groq(audio_path=wav_path, language=language)
            return result, f"groq/{groq_backend.GROQ_STT_MODEL}", "groq"
        except Exception as e:
            if not openai_backend.is_available():
                raise
            print(f"[stt] Groq en echec ({e}) : repli sur OpenAI", flush=True)

    # Groq choisi mais indisponible (pas de cle, ou appel en echec) : OpenAI prend le relais
    if preferred in ("groq", "openai") and openai_backend.is_available():
        result = openai_backend.transcribe_with_openai(audio_path=wav_path, language=language)
        return result, openai_backend.OPENAI_STT_MODEL, "openai"

    return None
