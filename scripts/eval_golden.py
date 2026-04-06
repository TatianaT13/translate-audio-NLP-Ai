"""
Plan d'expérience — évaluation de toutes les combinaisons modèle x prompt.

Ce script tourne le pipeline sur chaque audio et logue les résultats dans :
    outputs/experiments/results.csv

Si des traductions de référence existent dans data/golden/references/,
les scores BLEU sont calculés automatiquement.

Usage :
    # Tous les audios de l'archive, toutes les combinaisons
    python scripts/eval_golden.py

    # Un seul audio
    python scripts/eval_golden.py --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3

    # Reprendre sans écraser les runs déjà faits
    python scripts/eval_golden.py --skip-existing
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sacrebleu

# METEOR (nltk)
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score as _meteor_score
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    _METEOR_AVAILABLE = True
except ImportError:
    _METEOR_AVAILABLE = False

# WER (jiwer)
try:
    from jiwer import wer as _jiwer_wer
    _WER_AVAILABLE = True
except ImportError:
    _WER_AVAILABLE = False

# Langfuse — initialisé après load_dotenv
_lf = None

def _init_langfuse():
    global _lf
    try:
        from langfuse import Langfuse
        if os.getenv("LANGFUSE_PUBLIC_KEY") and _lf is None:
            _lf = Langfuse()
    except Exception:
        pass

# Ajouter src/ et scripts/ au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_nlp.transcription.whisper_service import WhisperService
from scripts.run_pipeline import run as run_pipeline

# ---------------------------------------------------------------------------
# Plan d'expérience
# ---------------------------------------------------------------------------

WHISPER_MODELS = [
    "large-v3",
    "small",
]

LLM_MODELS = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
]

PROMPT_VERSIONS = ["v1.0", "v1.1", "v1.2"]

TARGET_LANGS = ["en"]   # ajouter "uk" si besoin

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
ARCHIVE_DIR  = ROOT / "data" / "flash_audio_archive"
GOLDEN_DIR   = ROOT / "data" / "golden"
REFS_DIR     = GOLDEN_DIR / "references"
RESULTS_DIR  = ROOT / "outputs" / "experiments"
RESULTS_CSV  = RESULTS_DIR / "results.csv"

TTS_URL = os.getenv("TTS_URL", "http://localhost:8003")

CSV_HEADERS = [
    "run_id", "timestamp", "audio", "zone",
    "whisper_model", "llm_model", "prompt_version", "target_lang",
    "source_text", "translation",
    "language_prob",
    "latency_conv_ms", "latency_stt_ms", "latency_llm_ms", "latency_total_ms",
    "bleu", "meteor", "wer", "tts_wer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing_run_ids() -> set:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {row["run_id"] for row in reader}


def make_run_id(audio: Path, whisper: str, llm: str, prompt: str, lang: str) -> str:
    return f"{audio.stem}__{whisper}__{llm.replace('/', '-')}__{prompt}__{lang}"


def compute_bleu(hypothesis: str, audio_stem: str, lang: str) -> float:
    ref_path = REFS_DIR / f"{audio_stem}_{lang}.txt"
    if not ref_path.exists():
        return -1.0
    reference = ref_path.read_text(encoding="utf-8").strip()
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return round(result.score, 2)


def compute_meteor(hypothesis: str, audio_stem: str, lang: str) -> float:
    """METEOR sur la traduction (mêmes refs que BLEU)."""
    if not _METEOR_AVAILABLE:
        return -1.0
    ref_path = REFS_DIR / f"{audio_stem}_{lang}.txt"
    if not ref_path.exists():
        return -1.0
    reference = ref_path.read_text(encoding="utf-8").strip()
    score = _meteor_score([reference.split()], hypothesis.split())
    return round(score, 4)


def compute_wer(source_text: str, audio_stem: str) -> float:
    """WER sur la transcription STT (refs FR : data/golden/references/*_fr.txt)."""
    if not _WER_AVAILABLE:
        return -1.0
    ref_path = REFS_DIR / f"{audio_stem}_fr.txt"
    if not ref_path.exists():
        return -1.0
    reference = ref_path.read_text(encoding="utf-8").strip()
    return round(_jiwer_wer(reference, source_text), 4)


def compute_tts_wer(translation: str, lang: str, whisper_svc: "WhisperService") -> float:
    """Round-trip WER : TTS → audio → Whisper → WER vs texte original."""
    if not _WER_AVAILABLE:
        return -1.0
    try:
        import httpx
        r = httpx.post(
            f"{TTS_URL}/synthesize",
            json={"text": translation, "lang": lang},
            timeout=30.0,
        )
        if not r.is_success:
            return -1.0
        audio_bytes = r.content
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            transcript, _, _ = whisper_svc.transcribe_wav(tmp_path, language=lang, beam_size=1)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return round(_jiwer_wer(translation.lower(), transcript.lower()), 4)
    except Exception:
        return -1.0


def write_row(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, delimiter=";")
        if new:
            writer.writeheader()
        writer.writerow(row)


def get_zone(audio: Path) -> str:
    parts = audio.parts
    # structure : .../YYYY-MM-DD/<zone>/flash_*.mp3
    try:
        return parts[-2]
    except IndexError:
        return "unknown"


def collect_audios(audio_arg: str | None) -> list[Path]:
    if audio_arg:
        return [Path(audio_arg)]
    return sorted(ARCHIVE_DIR.rglob("*.mp3"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(audio_arg: str | None, skip_existing: bool) -> None:
    audios = collect_audios(audio_arg)
    if not audios:
        print("Aucun audio trouvé.")
        return

    existing = load_existing_run_ids() if skip_existing else set()

    # Combinaisons totales
    combinations = [
        (w, llm, pv, lang)
        for w   in WHISPER_MODELS
        for llm in LLM_MODELS
        for pv  in PROMPT_VERSIONS
        for lang in TARGET_LANGS
    ]
    total = len(audios) * len(combinations)
    print(f"\nPlan d'expérience : {len(audios)} audios x {len(combinations)} combinaisons = {total} runs")
    print(f"Résultats -> {RESULTS_CSV}\n")

    # Charger Whisper une seule fois par taille de modèle
    whisper_cache: dict[str, WhisperService] = {}

    done = 0
    skipped = 0

    for audio in audios:
        zone = get_zone(audio)
        print(f"\n{'='*60}")
        print(f"Audio : {audio.name}  (zone={zone})")
        print(f"{'='*60}")

        for (whisper_model, llm_model, prompt_version, target_lang) in combinations:
            run_id = make_run_id(audio, whisper_model, llm_model, prompt_version, target_lang)

            if run_id in existing:
                print(f"  [skip] {run_id}")
                skipped += 1
                continue

            print(f"\n  [{done+1}/{total}] {whisper_model} | {llm_model} | {prompt_version} | {target_lang}")

            # Récupérer ou créer le service Whisper
            if whisper_model not in whisper_cache:
                svc = WhisperService()
                svc.load(whisper_model, device="cpu")
                whisper_cache[whisper_model] = svc

            try:
                result = run_pipeline(
                    audio_path=audio,
                    model=llm_model,
                    target_lang=target_lang,
                    prompt_version=prompt_version,
                    whisper_model=whisper_model,
                    _svc=whisper_cache[whisper_model],
                )
            except Exception as e:
                print(f"  ERREUR : {e}")
                continue

            bleu    = compute_bleu(result["translation"], audio.stem, target_lang)
            meteor  = compute_meteor(result["translation"], audio.stem, target_lang)
            wer     = compute_wer(result["source_text"], audio.stem)
            tts_wer = compute_tts_wer(result["translation"], target_lang, whisper_cache[whisper_model])

            row = {
                "run_id":           run_id,
                "timestamp":        datetime.utcnow().isoformat(),
                "audio":            audio.name,
                "zone":             zone,
                "whisper_model":    result["whisper_model"],
                "llm_model":        result["llm_model"],
                "prompt_version":   result["prompt_version"],
                "target_lang":      result["target_lang"],
                "source_text":      result["source_text"],
                "translation":      result["translation"],
                "language_prob":    result["language_prob"],
                "latency_conv_ms":  result["latency_conv_ms"],
                "latency_stt_ms":   result["latency_stt_ms"],
                "latency_llm_ms":   result["latency_llm_ms"],
                "latency_total_ms": result["latency_total_ms"],
                "bleu":             bleu,
                "meteor":           meteor,
                "wer":              wer,
                "tts_wer":          tts_wer,
            }
            write_row(row)
            done += 1

            # Langfuse — log trace complète
            _init_langfuse()
            if _lf:
                try:
                    tid = _lf.create_trace_id()
                    scores = [
                        ("latency_total_ms", result["latency_total_ms"]),
                        ("latency_stt_ms",   result["latency_stt_ms"]),
                        ("latency_llm_ms",   result["latency_llm_ms"]),
                        ("language_prob",    result["language_prob"]),
                    ]
                    if bleu >= 0:
                        scores.append(("bleu", bleu))
                    if meteor >= 0:
                        scores.append(("meteor", meteor))
                    if wer >= 0:
                        scores.append(("wer", wer))
                    if tts_wer >= 0:
                        scores.append(("tts_wer", tts_wer))
                    for name, value in scores:
                        _lf.create_score(trace_id=tid, name=name, value=float(value))
                    _lf.flush()
                    print(f"  [Langfuse] ✓ trace={tid[:8]}...")
                except Exception as e:
                    print(f"  [Langfuse] {e}")

            parts = []
            parts.append(f"BLEU={bleu}"         if bleu    >= 0 else "BLEU=n/a")
            parts.append(f"METEOR={meteor}"     if meteor  >= 0 else "METEOR=n/a")
            parts.append(f"WER={wer}"           if wer     >= 0 else "WER=n/a (crée data/golden/references/{audio.stem}_fr.txt)")
            parts.append(f"TTS_WER={tts_wer}"   if tts_wer >= 0 else "TTS_WER=n/a (TTS non joignable)")
            print(f"  -> {' | '.join(parts)}  |  total={result['latency_total_ms']}ms  [sauvegardé]")

    print(f"\n{'='*60}")
    print(f"Terminé : {done} runs effectués, {skipped} ignorés")
    print(f"Résultats : {RESULTS_CSV}")

    if done > 0:
        _print_summary()


def _print_summary() -> None:
    print(f"\n--- Résumé ---")
    rows = []
    with RESULTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    if not rows:
        return

    # Trier par latence totale
    rows_with_bleu = [r for r in rows if float(r["bleu"]) >= 0]
    rows_by_latency = sorted(rows, key=lambda r: int(r["latency_total_ms"]))

    print(f"\nTop 3 plus rapides :")
    for r in rows_by_latency[:3]:
        print(f"  {r['whisper_model']:10s} | {r['llm_model']:35s} | {r['prompt_version']} | {r['latency_total_ms']}ms")

    if rows_with_bleu:
        rows_by_bleu = sorted(rows_with_bleu, key=lambda r: float(r["bleu"]), reverse=True)
        print(f"\nTop 3 meilleur BLEU :")
        for r in rows_by_bleu[:3]:
            meteor_str = f" | METEOR={r['meteor']}" if float(r.get("meteor", -1)) >= 0 else ""
            print(f"  {r['whisper_model']:10s} | {r['llm_model']:35s} | {r['prompt_version']} | BLEU={r['bleu']}{meteor_str}")

    rows_with_wer = [r for r in rows if float(r.get("wer", -1)) >= 0]
    if rows_with_wer:
        rows_by_wer = sorted(rows_with_wer, key=lambda r: float(r["wer"]))
        print(f"\nTop 3 meilleur WER (STT) :")
        for r in rows_by_wer[:3]:
            print(f"  {r['whisper_model']:10s} | {r['llm_model']:35s} | {r['prompt_version']} | WER={r['wer']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation plan d'expérience")
    parser.add_argument("--audio",         default=None,        help="Tester un seul audio (optionnel)")
    parser.add_argument("--skip-existing", action="store_true", help="Ignorer les runs déjà dans results.csv")
    args = parser.parse_args()

    main(audio_arg=args.audio, skip_existing=args.skip_existing)
