"""Tests unitaires — conversion audio ffmpeg du service STT.

Capture le fix `30b2aaf` : STT accepte n'importe quel format audio
même avec une mauvaise extension, via :
  - tempfile suffix .bin (ignore l'extension client)
  - ffmpeg -probesize 100M / -analyzeduration 100M (sniffing du contenu)

Note : on lit le SOURCE du module au lieu de l'importer, pour éviter
la dépendance `faster_whisper` (500 Mo avec torch) dans les tests unit.
"""

from __future__ import annotations

from pathlib import Path


STT_MAIN_SOURCE = (
    Path(__file__).parent.parent.parent.parent
    / "backend" / "services" / "stt" / "src" / "stt" / "main.py"
).read_text()


class TestConvertAudioFlags:
    """La fonction convert_audio() doit être robuste aux mauvaises extensions."""

    def test_uses_probesize_flag_for_format_detection(self):
        """La commande ffmpeg contient -probesize pour sniffer le format."""
        assert "-probesize" in STT_MAIN_SOURCE
        assert "100M" in STT_MAIN_SOURCE, "Should probe at least 100M for robust detection"

    def test_uses_analyzeduration_flag(self):
        assert "-analyzeduration" in STT_MAIN_SOURCE

    def test_output_format_is_wav_16k_mono(self):
        """La conversion doit produire du WAV 16kHz mono (Whisper input)."""
        assert "-ar" in STT_MAIN_SOURCE and "16000" in STT_MAIN_SOURCE
        assert "-ac" in STT_MAIN_SOURCE
        assert "pcm_s16le" in STT_MAIN_SOURCE


class TestTempfileExtensionHandling:
    """Le tempfile ne doit PAS garder l'extension client (source du bug corrigé)."""

    def test_uses_generic_bin_suffix_not_client_extension(self):
        """Le tempfile utilise .bin (générique) pour que ffmpeg détecte sur le contenu."""
        assert 'suffix=".bin"' in STT_MAIN_SOURCE, (
            "Le tempfile doit utiliser .bin, pas l'extension du client — "
            "sinon ffmpeg tente de décoder selon l'extension (ex: .mp3) au "
            "lieu du vrai format du contenu (ex: M4A)."
        )

    def test_docstring_mentions_multi_format_support(self):
        """Le docstring/comment doit expliquer pourquoi on ignore l'extension."""
        # Mentionne au moins un des formats susceptibles d'être renommés
        formats_mentionnes = ["M4A", "AAC", "WebM"]
        assert any(f in STT_MAIN_SOURCE for f in formats_mentionnes), (
            "Le code doit documenter les formats concernés par le fix"
        )
