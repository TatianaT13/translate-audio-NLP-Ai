import shutil
import subprocess
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    a = np.clip(audio, -1.0, 1.0)
    a16 = (a * 32767.0).astype(np.int16)
    wav_write(path, sr, a16)


def which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def ensure_ffmpeg_or_raise() -> None:
    if which_ffmpeg() is None:
        raise RuntimeError(
            "ffmpeg introuvable. Installe ffmpeg (ex: brew install ffmpeg sur macOS) "
            "pour supporter tous les formats (mp3, m4a, aac, flac, ogg, webm, etc.)."
        )


def convert_to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    ensure_ffmpeg_or_raise()
    cmd = [
        which_ffmpeg(),
        "-y",
        "-i", src_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        dst_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion échouée:\n{p.stderr}")


def list_input_devices() -> List[Tuple[int, str]]:
    devices = sd.query_devices()
    return [
        (i, d.get("name", "unknown"))
        for i, d in enumerate(devices)
        if d.get("max_input_channels", 0) > 0
    ]
