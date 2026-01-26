import argparse
import queue
import sys
import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel


@dataclass
class AudioChunk:
    pcm: np.ndarray
    sample_rate: int


def list_input_devices():
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            name = d.get("name", "unknown")
            print(f"{i}: {name}")


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def save_wav(path: str, audio: np.ndarray, sr: int):
    a = np.clip(audio, -1.0, 1.0)
    a16 = (a * 32767.0).astype(np.int16)
    wav_write(path, sr, a16)


def transcribe_wav(model: WhisperModel, wav_path: str, language: Optional[str], beam_size: int):
    segments, info = model.transcribe(
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--language", default=None, help="ex: fr, en. vide = auto")
    ap.add_argument("--input-device", type=int, default=None, help="id device (voir --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    ap.add_argument("--min-rms", type=float, default=0.01, help="seuil pour ignorer le silence (0.005 à 0.02)")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    sr = args.sample_rate
    chunk_samples = int(sr * args.chunk_seconds)

    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "int8" if args.device == "cpu" else "float16"

    print(f"Chargement modèle={args.model} device={args.device} compute_type={compute_type}")
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        mono = indata[:, 0].copy()
        q.put(mono)

    stream_kwargs = {
        "samplerate": sr,
        "channels": 1,
        "dtype": "float32",
        "callback": callback,
    }
    if args.input_device is not None:
        stream_kwargs["device"] = args.input_device

    print("Démarrage capture micro. Ctrl+C pour arrêter.")
    buffer: List[np.ndarray] = []
    buffered = 0
    last_print_ts = 0.0

    try:
        with sd.InputStream(**stream_kwargs):
            while True:
                x = q.get()
                buffer.append(x)
                buffered += x.shape[0]

                if buffered >= chunk_samples:
                    chunk = np.concatenate(buffer, axis=0)[:chunk_samples]
                    rest = np.concatenate(buffer, axis=0)[chunk_samples:]
                    buffer = [rest] if rest.size else []
                    buffered = rest.shape[0] if rest.size else 0

                    level = rms(chunk)
                    if level < args.min_rms:
                        continue

                    wav_path = ".live_chunk.wav"
                    save_wav(wav_path, chunk, sr)

                    t0 = time.time()
                    text, lang, prob = transcribe_wav(model, wav_path, args.language, args.beam_size)
                    dt = time.time() - t0

                    if text:
                        now = time.time()
                        if now - last_print_ts > 0.1:
                            last_print_ts = now
                            print(f"[{time.strftime('%H:%M:%S')}] ({lang},{prob:.2f}) {text}  (chunk {dt:.2f}s)")
                            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nArrêt.")
        return


if __name__ == "__main__":
    main()