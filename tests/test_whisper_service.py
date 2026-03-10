from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from flash_nlp.transcription.whisper_service import WhisperService


def _make_segment(text: str, start: float = 0.0, end: float = 1.0):
    s = SimpleNamespace(text=text, start=start, end=end)
    return s


def _make_info(language: str = "fr", probability: float = 0.95, duration: float = 5.0):
    return SimpleNamespace(language=language, language_probability=probability, duration=duration)


def _mock_model(mocker, segments=None, info=None):
    """Patche WhisperModel et retourne l'instance mock."""
    if segments is None:
        segments = [_make_segment("Bonjour le monde")]
    if info is None:
        info = _make_info()

    mock_instance = MagicMock()
    mock_instance.transcribe.return_value = (iter(segments), info)

    mock_cls = mocker.patch(
        "flash_nlp.transcription.whisper_service.WhisperModel",
        return_value=mock_instance,
    )
    return mock_cls, mock_instance


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_creates_model(mocker):
    mock_cls, _ = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small", device="cpu", compute_type="int8")
    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_load_cached_no_reload(mocker):
    mock_cls, _ = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small", device="cpu", compute_type="int8")
    svc.load("small", device="cpu", compute_type="int8")
    assert mock_cls.call_count == 1


def test_load_reloads_on_model_change(mocker):
    mock_cls, _ = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small", device="cpu", compute_type="int8")
    svc.load("medium", device="cpu", compute_type="int8")
    assert mock_cls.call_count == 2


def test_load_default_compute_type_cpu(mocker):
    mock_cls, _ = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small", device="cpu")
    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8")


def test_load_default_compute_type_cuda(mocker):
    mock_cls, _ = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small", device="cuda")
    mock_cls.assert_called_once_with("small", device="cuda", compute_type="float16")


# ---------------------------------------------------------------------------
# transcribe_wav()
# ---------------------------------------------------------------------------

def test_transcribe_wav_returns_tuple(mocker):
    _, mock_instance = _mock_model(
        mocker,
        segments=[_make_segment("Bonjour"), _make_segment("le monde")],
        info=_make_info("fr", 0.97),
    )
    svc = WhisperService()
    svc.load("small")

    text, lang, prob = svc.transcribe_wav("fake.wav", language="fr", beam_size=5)

    assert text == "Bonjour le monde"
    assert lang == "fr"
    assert abs(prob - 0.97) < 1e-6


def test_transcribe_wav_empty_segments(mocker):
    _, mock_instance = _mock_model(mocker, segments=[], info=_make_info("fr", 0.5))
    svc = WhisperService()
    svc.load("small")

    text, lang, prob = svc.transcribe_wav("fake.wav", language=None, beam_size=5)

    assert text == ""
    assert lang == "fr"


def test_transcribe_wav_strips_whitespace(mocker):
    _, mock_instance = _mock_model(
        mocker,
        segments=[_make_segment("  Bonjour  "), _make_segment("  monde  ")],
        info=_make_info(),
    )
    svc = WhisperService()
    svc.load("small")

    text, _, _ = svc.transcribe_wav("fake.wav", language=None, beam_size=5)
    assert text == "Bonjour monde"


def test_transcribe_wav_not_loaded_raises():
    svc = WhisperService()
    with pytest.raises(AssertionError):
        svc.transcribe_wav("fake.wav", language=None, beam_size=5)


def test_transcribe_wav_passes_correct_args(mocker):
    _, mock_instance = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small")

    svc.transcribe_wav("audio.wav", language="fr", beam_size=3)

    mock_instance.transcribe.assert_called_once_with(
        "audio.wav",
        language="fr",
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
    )


# ---------------------------------------------------------------------------
# transcribe_wav_with_segments()
# ---------------------------------------------------------------------------

def test_transcribe_wav_with_segments_structure(mocker):
    _, mock_instance = _mock_model(
        mocker,
        segments=[_make_segment("Flash info", 0.0, 2.5)],
        info=_make_info("fr", 0.92, duration=10.0),
    )
    svc = WhisperService()
    svc.load("small")

    result = svc.transcribe_wav_with_segments("fake.wav", language="fr", beam_size=5)

    assert set(result.keys()) == {"language", "language_probability", "duration", "segments", "text"}
    assert result["language"] == "fr"
    assert result["text"] == "Flash info"
    assert len(result["segments"]) == 1
    assert result["segments"][0] == {"start": 0.0, "end": 2.5, "text": "Flash info"}


def test_transcribe_wav_with_segments_custom_silence(mocker):
    _, mock_instance = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small")

    svc.transcribe_wav_with_segments("fake.wav", language=None, beam_size=5, min_silence_ms=800)

    call_kwargs = mock_instance.transcribe.call_args[1]
    assert call_kwargs["vad_parameters"]["min_silence_duration_ms"] == 800
