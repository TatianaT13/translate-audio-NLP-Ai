"""Cout reel d'un traitement, calcule a partir de l'usage remonte par les services.

Les services (pipeline, STT) n'envoient que des volumes : secondes transcrites,
caracteres synthetises, cout LLM mesure par LiteLLM. Les tarifs des fournisseurs
sont ici et nulle part ailleurs : quand un prix change, on modifie une ligne
(ou la variable d'environnement correspondante).
"""
from __future__ import annotations

import os


def _rate(env: str, default: str) -> float:
    return float(os.getenv(env, default))


# USD par minute d'audio transcrite, selon le backend STT utilise.
STT_USD_PER_MIN = {
    "openai":     _rate("COST_STT_OPENAI_USD_PER_MIN", "0.006"),
    "groq":       _rate("COST_STT_GROQ_USD_PER_MIN", "0.00067"),   # 0,04 $/h
    # 0,21 $/h (modele le plus cher) + 0,02 $/h de diarisation : hypothese haute
    "assemblyai": _rate("COST_STT_ASSEMBLYAI_USD_PER_MIN", "0.0038"),
    "local":      0.0,  # Whisper self-hosted : cout fixe du serveur, pas a l'usage
    "none":       0.0,  # debit sans transcription (voix a la demande)
}

# USD pour 1 000 caracteres synthetises, selon le backend TTS.
TTS_USD_PER_1K_CHARS = {
    "mistral": _rate("COST_TTS_MISTRAL_USD_PER_1K_CHARS", "0.016"),
    "local":   0.0,
}

USD_TO_EUR = _rate("USD_TO_EUR", "0.92")

# Ce que rapporte un credit, frais Stripe deduits. Valeur de la formule la moins
# chere (annuel : 99 EUR pour 2 400 credits) pour ne pas surestimer la marge ;
# le mensuel rapporte 0,0475 EUR.
CREDIT_VALUE_EUR = _rate("CREDIT_VALUE_EUR", "0.0405")


def usage_cost_usd(meta: dict | None) -> float | None:
    """Cout fournisseur d'un debit, en USD. None si `meta` ne decrit pas
    l'usage (transactions anterieures au suivi des couts).

    Backend inconnu : on applique le tarif le plus eleve, pour ne jamais
    sous-estimer un cout.
    """
    if not meta or "stt_backend" not in meta:
        return None
    stt_rate = STT_USD_PER_MIN.get(meta["stt_backend"], max(STT_USD_PER_MIN.values()))
    tts_rate = TTS_USD_PER_1K_CHARS.get(meta.get("tts_backend"), max(TTS_USD_PER_1K_CHARS.values()))
    return (
        (meta.get("seconds") or 0) / 60 * stt_rate
        + (meta.get("tts_chars") or 0) / 1000 * tts_rate
        + (meta.get("llm_cost_usd") or 0)
    )
