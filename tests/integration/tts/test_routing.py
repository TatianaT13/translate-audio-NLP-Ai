"""Test d'intégration du routing TTS multi-backend.

Contexte : Mistral Voxtral supporte bien EN/ES/DE/FR mais pas l'ukrainien
(voix anglaise avec accent médiocre). On route donc UK vers MMS-TTS local
(modèle Meta open-source, mms-tts-ukr) qui produit une voix ukrainienne native.

Le routing se fait dans backend/services/tts/src/tts/main.py :
    MISTRAL_UNSUPPORTED_LANGS = {"uk"}
    use_mistral = (TTS_BACKEND == "mistral" and lang not in MISTRAL_UNSUPPORTED_LANGS)

Distinguer les deux backends par le content-type de la réponse :
  - Voxtral   → audio/mpeg  (MP3)
  - MMS local → audio/wav   (WAV)
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


class TestTTSRouting:
    """Le TTS service route par langue vers Voxtral ou MMS-TTS local."""

    def test_health_reports_backend_and_local_langs(self, client, tts_url):
        """Le /health doit exposer le backend actif + les langues MMS locales."""
        r = client.get(f"{tts_url}/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["backend"] in ("mistral", "local")
        assert "uk" in body["local_languages"]

    def test_english_uses_mistral_voxtral(self, client, tts_url):
        """EN doit passer par Voxtral (Mistral) → MP3."""
        r = client.post(
            f"{tts_url}/synthesize",
            json={"text": "Traffic jams on the A6 highway.", "lang": "en"},
            timeout=60,
        )
        if r.status_code != 200 and "guardrail" in r.text.lower():
            pytest.skip(f"Voxtral guardrail: {r.text[:200]}")
        assert r.status_code == 200, r.text[:300]
        assert r.headers["content-type"] == "audio/mpeg"
        assert len(r.content) > 1000

    def test_ukrainian_uses_mms_local_not_voxtral(self, client, tts_url):
        """UK doit tomber sur MMS-TTS local → WAV (pas MP3 Voxtral).

        Le point critique testé ici : UK NE DOIT PAS renvoyer audio/mpeg (=Voxtral).
        Si le modèle MMS local n'est pas téléchargé, on obtient 500 avec un message
        explicite — ce qui prouve quand même que le routing n'a pas basculé sur
        Voxtral (sinon on aurait un 200 audio/mpeg). Skip élégant dans ce cas.
        """
        r = client.post(
            f"{tts_url}/synthesize",
            json={"text": "Пробки на автомагістралі A6.", "lang": "uk"},
            timeout=60,
        )

        # Cas "modèle non téléchargé" → prouve que routing UK != Voxtral (bien)
        if r.status_code == 500 and "MMS-TTS non disponible" in r.text:
            pytest.skip(
                "Modèle mms-tts-ukr non téléchargé dans models/ — "
                "routing correct mais fichier manquant. "
                "Sur hermes/prod le modèle doit être présent."
            )

        assert r.status_code == 200, r.text[:300]
        assert r.headers["content-type"] == "audio/wav", (
            "UK devrait utiliser MMS local (WAV), pas Voxtral (MP3). "
            "Vérifier MISTRAL_UNSUPPORTED_LANGS dans tts/main.py"
        )
        assert len(r.content) > 1000

    @pytest.mark.parametrize("lang,text", [
        ("es", "Atascos en la autopista A6."),
        ("de", "Staus auf der Autobahn A6."),
    ])
    def test_es_and_de_use_voxtral(self, client, tts_url, lang, text):
        """ES et DE doivent aussi passer par Voxtral (bien supportés)."""
        r = client.post(f"{tts_url}/synthesize", json={"text": text, "lang": lang}, timeout=60)
        if r.status_code != 200 and "guardrail" in r.text.lower():
            pytest.skip(f"Voxtral guardrail: {r.text[:200]}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mpeg"

    def test_empty_text_returns_400(self, client, tts_url):
        r = client.post(f"{tts_url}/synthesize", json={"text": "", "lang": "en"})
        assert r.status_code == 400

    def test_whitespace_only_text_returns_400(self, client, tts_url):
        r = client.post(f"{tts_url}/synthesize", json={"text": "   \n\t  ", "lang": "en"})
        assert r.status_code == 400
