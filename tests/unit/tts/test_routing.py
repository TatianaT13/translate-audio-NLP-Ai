"""Tests unitaires — routing TTS multi-backend.

Capture le fix routing UK vers MMS local :
  Voxtral Mistral n'est pas entraîné en ukrainien → voix anglaise avec accent
  → on route UK vers Meta MMS-TTS local (mms-tts-ukr).
"""

from __future__ import annotations


class TestUnsupportedLangs:
    """Le set MISTRAL_UNSUPPORTED_LANGS contrôle le routing par langue."""

    def test_uk_is_in_unsupported_langs(self):
        from tts.main import MISTRAL_UNSUPPORTED_LANGS
        assert "uk" in MISTRAL_UNSUPPORTED_LANGS, (
            "UK doit être routé vers MMS local, pas Voxtral (accent anglais)"
        )

    def test_en_is_NOT_in_unsupported_langs(self):
        from tts.main import MISTRAL_UNSUPPORTED_LANGS
        assert "en" not in MISTRAL_UNSUPPORTED_LANGS

    def test_es_is_NOT_in_unsupported_langs(self):
        from tts.main import MISTRAL_UNSUPPORTED_LANGS
        assert "es" not in MISTRAL_UNSUPPORTED_LANGS

    def test_de_is_NOT_in_unsupported_langs(self):
        from tts.main import MISTRAL_UNSUPPORTED_LANGS
        assert "de" not in MISTRAL_UNSUPPORTED_LANGS


class TestLangModelsMapping:
    """LANG_MODELS mappe chaque langue vers son modèle MMS local."""

    def test_uk_maps_to_mms_ukr(self):
        from tts.main import LANG_MODELS
        assert "uk" in LANG_MODELS
        assert "ukr" in str(LANG_MODELS["uk"]), "UK doit pointer vers mms-tts-ukr"

    def test_en_maps_to_mms_eng(self):
        from tts.main import LANG_MODELS
        assert "en" in LANG_MODELS
        assert "eng" in str(LANG_MODELS["en"])
