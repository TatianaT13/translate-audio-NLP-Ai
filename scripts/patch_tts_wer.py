"""
Recompute tts_wer for rows where tts_wer == -1.0.

Usage:
    .venv.nosync/bin/python scripts/patch_tts_wer.py

Requires Docker TTS service running on localhost:8003.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flash_nlp.transcription.whisper_service import WhisperService

try:
    from jiwer import wer as _jiwer_wer
    _WER_AVAILABLE = True
except ImportError:
    print("jiwer non installé — .venv.nosync/bin/python -m pip install jiwer")
    sys.exit(1)

import httpx

ROOT        = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "outputs" / "experiments" / "results.csv"
TTS_URL     = os.getenv("TTS_URL", "http://localhost:8003")


def call_tts(text: str, lang: str) -> bytes | None:
    try:
        r = httpx.post(f"{TTS_URL}/synthesize", json={"text": text, "lang": lang}, timeout=30.0)
        return r.content if r.is_success else None
    except Exception as e:
        print(f"    TTS erreur : {e}")
        return None


def transcribe(audio_bytes: bytes, lang: str, svc: WhisperService) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        text, _, _ = svc.transcribe_wav(tmp, language=lang, beam_size=1)
        return text
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    if not RESULTS_CSV.exists():
        print("results.csv introuvable.")
        return

    rows = []
    with RESULTS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    if "tts_wer" not in (fieldnames or []):
        print("Colonne tts_wer absente du CSV.")
        return

    to_patch = [r for r in rows if r.get("tts_wer", "").strip() in ("-1.0", "-1", "")]
    print(f"{len(to_patch)} runs à patcher sur {len(rows)} total.")

    if not to_patch:
        print("Rien à faire.")
        return

    # Vérifier que TTS est joignable
    try:
        httpx.get(f"{TTS_URL}/health", timeout=5.0)
    except Exception:
        print(f"TTS non joignable sur {TTS_URL} — lance docker compose up -d")
        return

    # Charger Whisper (réutilise le même modèle pour tous les runs)
    whisper_cache: dict[str, WhisperService] = {}

    for i, row in enumerate(to_patch):
        whisper_model = row["whisper_model"]
        lang          = row["target_lang"]
        translation   = row["translation"]
        run_id        = row["run_id"]

        print(f"\n[{i+1}/{len(to_patch)}] {run_id[:60]}…")

        if whisper_model not in whisper_cache:
            print(f"  Chargement Whisper {whisper_model}…")
            svc = WhisperService()
            svc.load(whisper_model, device="cpu")
            whisper_cache[whisper_model] = svc

        audio_bytes = call_tts(translation, lang)
        if not audio_bytes:
            print("  TTS échoué, skip.")
            continue

        transcript = transcribe(audio_bytes, lang, whisper_cache[whisper_model])
        if not transcript:
            print("  Transcription vide, skip.")
            continue

        tts_wer = round(_jiwer_wer(translation.lower(), transcript.lower()), 4)
        row["tts_wer"] = str(tts_wer)
        print(f"  TTS_WER = {tts_wer}")

    # Réécrire le CSV
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV mis à jour : {RESULTS_CSV}")


if __name__ == "__main__":
    main()
