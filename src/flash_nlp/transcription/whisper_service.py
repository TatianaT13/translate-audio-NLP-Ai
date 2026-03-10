from typing import List, Optional, Tuple

from faster_whisper import WhisperModel


class WhisperService:
    def __init__(self):
        self._model: Optional[WhisperModel] = None
        self._model_name: Optional[str] = None
        self._device: str = "cpu"
        self._compute_type: str = "int8"

    def load(self, model_name: str, device: str = "cpu", compute_type: Optional[str] = None):
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        if (
            self._model is not None
            and self._model_name == model_name
            and self._device == device
            and self._compute_type == compute_type
        ):
            return

        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe_wav(
        self,
        wav_path: str,
        language: Optional[str],
        beam_size: int,
    ) -> Tuple[str, str, float]:
        assert self._model is not None, "Modèle non chargé. Appeler load() d'abord."
        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
        )
        texts: List[str] = []
        for s in segments:
            t = (s.text or "").strip()
            if t:
                texts.append(t)
        return " ".join(texts), info.language, float(info.language_probability)

    def transcribe_wav_with_segments(
        self,
        wav_path: str,
        language: Optional[str],
        beam_size: int,
        min_silence_ms: int = 500,
    ) -> dict:
        assert self._model is not None, "Modèle non chargé. Appeler load() d'abord."
        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": min_silence_ms},
        )
        segs = []
        texts = []
        for s in segments:
            text = (s.text or "").strip()
            segs.append({"start": float(s.start), "end": float(s.end), "text": text})
            if text:
                texts.append(text)
        return {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
            "segments": segs,
            "text": " ".join(texts),
        }
