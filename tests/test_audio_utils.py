import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.io.wavfile import read as wav_read

from flash_nlp.transcription.audio_utils import (
    convert_to_wav_16k_mono,
    ensure_ffmpeg_or_raise,
    list_input_devices,
    rms,
    save_wav,
    which_ffmpeg,
)


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

def test_rms_known_value():
    # Signal carré d'amplitude 1.0 → RMS = 1.0
    x = np.ones(1000, dtype=np.float32)
    assert abs(rms(x) - 1.0) < 1e-6


def test_rms_silence():
    x = np.zeros(500, dtype=np.float32)
    assert rms(x) == 0.0


def test_rms_empty_array():
    x = np.array([], dtype=np.float32)
    assert rms(x) == 0.0


def test_rms_known_sine():
    # RMS d'un sinus d'amplitude A = A / sqrt(2)
    t = np.linspace(0, 2 * np.pi, 10000)
    x = np.sin(t).astype(np.float32)
    expected = 1.0 / np.sqrt(2)
    assert abs(rms(x) - expected) < 1e-3


# ---------------------------------------------------------------------------
# save_wav
# ---------------------------------------------------------------------------

def test_save_wav_creates_file(tmp_path):
    path = str(tmp_path / "out.wav")
    audio = np.zeros(1600, dtype=np.float32)
    save_wav(path, audio, sr=16000)
    assert Path(path).exists()


def test_save_wav_readable_by_scipy(tmp_path):
    path = str(tmp_path / "out.wav")
    audio = np.linspace(-0.5, 0.5, 3200, dtype=np.float32)
    save_wav(path, audio, sr=16000)

    sr, data = wav_read(path)
    assert sr == 16000
    assert data.dtype == np.int16
    assert len(data) == 3200


def test_save_wav_clips_overflow(tmp_path):
    path = str(tmp_path / "clipped.wav")
    audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    save_wav(path, audio, sr=16000)

    _, data = wav_read(path)
    # Les valeurs doivent être bornées à [-32767, 32767]
    assert data[0] == 32767   # 2.0 → clip → 1.0 → 32767
    assert data[1] == -32767  # -3.0 → clip → -1.0 → -32767


# ---------------------------------------------------------------------------
# which_ffmpeg / ensure_ffmpeg_or_raise
# ---------------------------------------------------------------------------

def test_which_ffmpeg_found(mocker):
    mocker.patch(
        "flash_nlp.transcription.audio_utils.shutil.which",
        return_value="/usr/bin/ffmpeg",
    )
    assert which_ffmpeg() == "/usr/bin/ffmpeg"


def test_which_ffmpeg_not_found(mocker):
    mocker.patch(
        "flash_nlp.transcription.audio_utils.shutil.which",
        return_value=None,
    )
    assert which_ffmpeg() is None


def test_ensure_ffmpeg_raises_if_missing(mocker):
    mocker.patch(
        "flash_nlp.transcription.audio_utils.shutil.which",
        return_value=None,
    )
    with pytest.raises(RuntimeError, match="ffmpeg introuvable"):
        ensure_ffmpeg_or_raise()


def test_ensure_ffmpeg_passes_if_present(mocker):
    mocker.patch(
        "flash_nlp.transcription.audio_utils.shutil.which",
        return_value="/usr/bin/ffmpeg",
    )
    ensure_ffmpeg_or_raise()  # ne doit pas lever


# ---------------------------------------------------------------------------
# convert_to_wav_16k_mono
# ---------------------------------------------------------------------------

def test_convert_wav_calls_ffmpeg(tmp_path, mocker):
    mocker.patch(
        "flash_nlp.transcription.audio_utils.shutil.which",
        return_value="/usr/bin/ffmpeg",
    )
    mock_run = mocker.patch(
        "flash_nlp.transcription.audio_utils.subprocess.run",
        return_value=MagicMock(returncode=0, stderr=""),
    )
    convert_to_wav_16k_mono("input.mp3", "output.wav")

    args = mock_run.call_args[0][0]
    assert args[0] == "/usr/bin/ffmpeg"
    assert "-ar" in args
    assert "16000" in args
    assert "-ac" in args
    assert "1" in args


def test_convert_wav_raises_on_error(tmp_path, mocker):
    mocker.patch(
        "flash_nlp.transcription.audio_utils.shutil.which",
        return_value="/usr/bin/ffmpeg",
    )
    mocker.patch(
        "flash_nlp.transcription.audio_utils.subprocess.run",
        return_value=MagicMock(returncode=1, stderr="Invalid data found"),
    )
    with pytest.raises(RuntimeError, match="ffmpeg conversion échouée"):
        convert_to_wav_16k_mono("bad.mp3", "output.wav")


# ---------------------------------------------------------------------------
# list_input_devices
# ---------------------------------------------------------------------------

def test_list_input_devices_filters_inputs(mocker):
    fake_devices = [
        {"name": "Micro USB", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Haut-parleur", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Interface audio", "max_input_channels": 4, "max_output_channels": 4},
    ]
    mocker.patch(
        "flash_nlp.transcription.audio_utils.sd.query_devices",
        return_value=fake_devices,
    )
    result = list_input_devices()
    names = [name for _, name in result]
    assert "Micro USB" in names
    assert "Interface audio" in names
    assert "Haut-parleur" not in names


def test_list_input_devices_returns_index(mocker):
    fake_devices = [
        {"name": "Device A", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "Device B", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Device C", "max_input_channels": 2, "max_output_channels": 0},
    ]
    mocker.patch(
        "flash_nlp.transcription.audio_utils.sd.query_devices",
        return_value=fake_devices,
    )
    result = list_input_devices()
    indices = [idx for idx, _ in result]
    assert 0 in indices  # Device A → index 0
    assert 2 in indices  # Device C → index 2
    assert 1 not in indices  # Device B → pas d'input
