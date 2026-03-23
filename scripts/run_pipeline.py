"""
Premier run end-to-end :
    MP3 -> WhisperService (STT) -> texte -> LLM (GROQ ou Ollama) -> traduction

Prérequis GROQ :
    Créer un fichier .env à la racine du projet :
        GROQ_API_KEY=gsk_xxxxxxxxxxxx
    Obtenir une clé gratuite sur https://console.groq.com

Usage :
    # GROQ (cloud, aucune installation locale)
    python scripts/run_pipeline.py \\
        --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3 \\
        --model groq/llama3-8b-8192

    # Ollama (local, nécessite `ollama serve` + `ollama pull phi3:mini`)
    python scripts/run_pipeline.py \\
        --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3 \\
        --model ollama/phi3:mini

    # Tester tous les prompts sur un audio
    python scripts/run_pipeline.py \\
        --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3 \\
        --model groq/llama3-8b-8192 \\
        --all-prompts
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

# Charger .env si présent
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import litellm

from flash_nlp.transcription.audio_utils import convert_to_wav_16k_mono, ensure_ffmpeg_or_raise
from flash_nlp.transcription.whisper_service import WhisperService

# Désactiver les logs verbeux de litellm
litellm.set_verbose = False

# ---------------------------------------------------------------------------
# Prompts versionnés
# ---------------------------------------------------------------------------

PROMPTS = {
    "v1.0": (
        "Translate the following French text to {lang}. "
        "Output only the translation, nothing else.\n\n{text}"
    ),
    "v1.1": (
        "You are a professional translator specializing in traffic and road safety announcements. "
        "Translate the following French traffic bulletin to {lang}. "
        "Keep road names (A6, N7...) as-is. Output only the translation.\n\n{text}"
    ),
    "v1.2": (
        "You are a professional translator. "
        "Translate this French road traffic report to {lang}. "
        "Preserve road identifiers (A6, N7, D roads). "
        "Use clear, concise language suitable for a radio broadcast. "
        "Output only the translated text.\n\n{text}"
    ),
}

LANG_LABELS = {
    "en": "English",
    "uk": "Ukrainian",
    "es": "Spanish",
    "de": "German",
}


# ---------------------------------------------------------------------------
# Appel LLM via LiteLLM (GROQ, Ollama, Anthropic...)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str, timeout: int = 60) -> tuple[str, float]:
    """
    model examples :
        "groq/llama3-8b-8192"
        "groq/mixtral-8x7b-32768"
        "ollama/phi3:mini"
        "anthropic/claude-haiku-4-5-20251001"
    """
    t0 = time.perf_counter()
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        translation = response.choices[0].message.content.strip()
        latency_ms = (time.perf_counter() - t0) * 1000
        return translation, latency_ms
    except Exception as e:
        raise RuntimeError(f"Erreur LLM ({model}) : {e}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(
    audio_path: Path,
    model: str,
    target_lang: str,
    prompt_version: str,
    whisper_model: str,
    _svc: WhisperService = None,
) -> dict:
    ensure_ffmpeg_or_raise()

    lang_label = LANG_LABELS.get(target_lang, target_lang)

    print(f"\n{'─'*50}")
    print(f"Audio         : {audio_path.name}")
    print(f"Whisper model : {whisper_model}")
    print(f"LLM model     : {model}")
    print(f"Prompt        : {prompt_version}")
    print(f"Target lang   : {lang_label}")
    print(f"{'─'*50}")

    # 1. Conversion MP3 -> WAV 16kHz mono
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    t0 = time.perf_counter()
    convert_to_wav_16k_mono(str(audio_path), wav_path)
    conv_ms = (time.perf_counter() - t0) * 1000
    print(f"[1/3] Conversion     : {conv_ms:.0f}ms")

    # 2. Transcription Whisper (réutilise le service si fourni)
    svc = _svc or WhisperService()
    if not svc._model:
        svc.load(whisper_model, device="cpu")

    t0 = time.perf_counter()
    text, lang, lang_prob = svc.transcribe_wav(wav_path, language="fr", beam_size=5)
    stt_ms = (time.perf_counter() - t0) * 1000
    print(f"[2/3] Transcription  : {stt_ms:.0f}ms  |  {lang}  {lang_prob:.0%}")
    print(f"      Texte          : {text}")

    # 3. Traduction LLM
    prompt = PROMPTS[prompt_version].format(lang=lang_label, text=text)
    translation, llm_ms = call_llm(prompt, model=model)
    print(f"[3/3] Traduction     : {llm_ms:.0f}ms")
    print(f"      Résultat       : {translation}")

    total_ms = conv_ms + stt_ms + llm_ms
    print(f"\n  Total : {total_ms/1000:.1f}s  (conv={conv_ms:.0f}ms  stt={stt_ms:.0f}ms  llm={llm_ms:.0f}ms)")

    return {
        "audio":            audio_path.name,
        "whisper_model":    whisper_model,
        "llm_model":        model,
        "prompt_version":   prompt_version,
        "target_lang":      target_lang,
        "source_text":      text,
        "language":         lang,
        "language_prob":    round(lang_prob, 4),
        "translation":      translation,
        "latency_conv_ms":  round(conv_ms),
        "latency_stt_ms":   round(stt_ms),
        "latency_llm_ms":   round(llm_ms),
        "latency_total_ms": round(total_ms),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline STT + traduction")
    parser.add_argument("--audio",          required=True,              help="Chemin vers le fichier MP3/WAV")
    parser.add_argument("--model",          default="groq/llama3-8b-8192", help="Modèle LiteLLM (ex: groq/llama3-8b-8192, ollama/phi3:mini)")
    parser.add_argument("--target-lang",    default="en",               help="Langue cible : en, uk, es, de")
    parser.add_argument("--prompt-version", default="v1.1",             help="Version du prompt : v1.0, v1.1, v1.2")
    parser.add_argument("--whisper-model",  default="small",            help="Modèle Whisper : tiny, small, medium, large-v3")
    parser.add_argument("--all-prompts",    action="store_true",        help="Tester tous les prompts sur cet audio")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Fichier introuvable : {audio_path}")
        exit(1)

    # Vérifier clé GROQ si nécessaire
    if "groq" in args.model and not os.getenv("GROQ_API_KEY"):
        print("\nERREUR : GROQ_API_KEY manquante.")
        print("  1. Crée un compte sur https://console.groq.com")
        print("  2. Crée un fichier .env à la racine du projet :")
        print("       GROQ_API_KEY=gsk_xxxxxxxxxxxx")
        exit(1)

    # Charger Whisper une seule fois
    svc = WhisperService()
    svc.load(args.whisper_model, device="cpu")

    if args.all_prompts:
        print(f"\nMode comparaison — {len(PROMPTS)} prompts x 1 modèle")
        results = []
        for pv in PROMPTS:
            r = run(
                audio_path=audio_path,
                model=args.model,
                target_lang=args.target_lang,
                prompt_version=pv,
                whisper_model=args.whisper_model,
                _svc=svc,
            )
            results.append(r)

        print(f"\n{'='*50}")
        print("COMPARAISON DES PROMPTS")
        print(f"{'='*50}")
        for r in results:
            print(f"\n[{r['prompt_version']}] {r['llm_model']}  ({r['latency_llm_ms']}ms)")
            print(f"  {r['translation']}")
    else:
        run(
            audio_path=audio_path,
            model=args.model,
            target_lang=args.target_lang,
            prompt_version=args.prompt_version,
            whisper_model=args.whisper_model,
            _svc=svc,
        )
