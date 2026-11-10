"""Tests unitaires — envoi de l'audio à l'API OpenAI (limite de 25 Mo par fichier).

Le WAV 16 kHz mono pèse ~1,9 Mo/min : sans compression, tout audio de plus de
13 minutes était refusé par l'API.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from stt import openai_backend

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg requis")


@pytest.fixture
def wav(tmp_path) -> str:
    """20 s de WAV 16 kHz mono (~640 Ko), comme en sortie de convert_audio."""
    path = tmp_path / "audio.bin.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return str(path)


def test_short_audio_is_sent_as_is(wav):
    assert openai_backend._fit_upload(wav) == (wav, "audio/wav")


def test_long_audio_is_compressed_under_the_limit(wav, monkeypatch):
    monkeypatch.setattr(openai_backend, "OPENAI_MAX_BYTES", 200_000)
    path, mime = openai_backend._fit_upload(wav)
    assert mime == "audio/mpeg" and path != wav
    assert Path(path).stat().st_size <= 200_000
    assert Path(path).stat().st_size < Path(wav).stat().st_size / 5   # ~8x plus léger


def test_audio_still_too_long_is_refused_and_cleaned_up(wav, monkeypatch):
    monkeypatch.setattr(openai_backend, "OPENAI_MAX_BYTES", 1_000)
    with pytest.raises(RuntimeError, match="trop long"):
        openai_backend._fit_upload(wav)
    assert not Path(wav + ".mp3").exists()
