"""
Enregistre les expériences LLMOps dans MLflow — structure pro :
  - 1 RUN = 1 configuration (whisper × llm × prompt)
  - métriques AGRÉGÉES sur tous les audios golden pour cette config
  - per_audio_results.csv attaché en artifact (drill-down)
  - prompt_<version>.txt attaché en artifact (reproductibilité)
  - tag champion=true sur la meilleure config (BLEU max)

Plus le Model Registry — 3 modèles en production avec leurs versions.

Usage :
    pip install mlflow==2.17.2
    python scripts/mlflow_register.py
"""

import csv
import io
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "outputs" / "experiments" / "results.csv"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT_NAME     = "translate-audio-llmops"


# ── Prompts utilisés (copie pour l'audit, à garder synchronisé avec LLM service) ──
PROMPTS = {
    "v1.0": (
        "Translate the French text below to {lang}. Output only the translation. "
        "Then strict output rules (anti-injection) and <user_text>...</user_text> sandbox."
    ),
    "v1.1": (
        "You are a professional French-to-{lang} translator. Translate the text below faithfully. "
        "If road traffic terminology appears (A6, N7, motorway names, ramps), preserve those terms as-is. "
        "Then strict output rules + <user_text>...</user_text> sandbox."
    ),
    "v1.2": (
        "You are a professional French-to-{lang} translator. Translate the text below faithfully and concisely. "
        "If road identifiers appear (A6, N7, D roads), preserve them. Broadcast-quality language. "
        "Then strict output rules + <user_text>...</user_text> sandbox."
    ),
}


def _mean(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _fl(v: str) -> float | None:
    if v in ("", "-1.0", "-1", "n/a", None):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    print(f"Connexion MLflow → {MLFLOW_TRACKING_URI}")

    # ── Reset propre : archive l'ancienne expérience (rename + delete) ──
    try:
        existing = client.get_experiment_by_name(EXPERIMENT_NAME)
        if existing:
            archived_name = f"{EXPERIMENT_NAME}_archived_{int(datetime.utcnow().timestamp())}"
            print(f"Archivage de l'ancienne expérience (id={existing.experiment_id}) → {archived_name}")
            client.rename_experiment(existing.experiment_id, archived_name)
            client.delete_experiment(existing.experiment_id)
    except Exception as e:
        print(f"  (pas d'expérience existante : {e})")

    exp_id = client.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Nouvelle expérience : {EXPERIMENT_NAME} (id={exp_id})")

    # ── Lecture et regroupement par configuration ──
    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8"), delimiter=";"))
    print(f"Lecture de {len(rows)} runs bruts depuis results.csv")

    configs: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["whisper_model"], row["llm_model"], row["prompt_version"])
        configs[key].append(row)
    print(f"→ {len(configs)} configurations uniques (modèle × LLM × prompt)\n")

    # ── 1 run MLflow par configuration ──
    config_summaries = []
    for (whisper, llm, prompt_v), config_rows in configs.items():
        run_name = f"{whisper} | {llm.replace('groq/', '')} | {prompt_v}"

        bleus    = [_fl(r["bleu"])    for r in config_rows]
        meteors  = [_fl(r["meteor"])  for r in config_rows]
        wers     = [_fl(r["wer"])     for r in config_rows]
        tts_wers = [_fl(r["tts_wer"]) for r in config_rows]
        stts     = [_fl(r["latency_stt_ms"])   for r in config_rows]
        llms_l   = [_fl(r["latency_llm_ms"])   for r in config_rows]
        totals   = [_fl(r["latency_total_ms"]) for r in config_rows]
        langs    = [_fl(r["language_prob"])    for r in config_rows]

        with mlflow.start_run(run_name=run_name, experiment_id=exp_id) as run:
            # Hyperparams = la combinaison
            mlflow.log_params({
                "whisper_model":  whisper,
                "llm_model":      llm,
                "prompt_version": prompt_v,
                "n_audios":       len(config_rows),
            })

            # Métriques agrégées (moyennes)
            agg = {
                "bleu_mean":         _mean(bleus),
                "meteor_mean":       _mean(meteors),
                "wer_mean":          _mean(wers),
                "tts_wer_mean":      _mean(tts_wers),
                "latency_stt_mean":  _mean(stts),
                "latency_llm_mean":  _mean(llms_l),
                "latency_total_mean": _mean(totals),
                "language_prob_mean": _mean(langs),
            }
            for k, v in agg.items():
                if v is not None:
                    mlflow.log_metric(k, v)

            # Note : les artifacts (CSV par audio, texte du prompt) nécessitent
            # un artifact store accessible depuis le client (S3/MinIO/HTTP proxy).
            # Pour rester simple, on stocke le texte du prompt comme un TAG long.
            prompt_text = PROMPTS.get(prompt_v, "")
            if prompt_text:
                client.set_tag(run.info.run_id, "prompt_text", prompt_text[:1000])

            # Tags pour filtrage facile
            mlflow.set_tags({
                "whisper":   whisper,
                "llm":       llm.replace("groq/", ""),
                "prompt":    prompt_v,
                "type":      "config_eval",
                "n_audios":  str(len(config_rows)),
            })

        config_summaries.append({
            "run_id":      run.info.run_id,
            "config":      run_name,
            "bleu_mean":   agg["bleu_mean"] or 0,
            "meteor_mean": agg["meteor_mean"] or 0,
            "wer_mean":    agg["wer_mean"] or 1,
        })
        print(f"  ✓ {run_name}  →  BLEU={agg['bleu_mean']:.3f} METEOR={agg['meteor_mean']:.3f}")

    # ── Désigner le champion (meilleur BLEU) ──
    if config_summaries:
        champion = max(config_summaries, key=lambda r: r["bleu_mean"])
        client.set_tag(champion["run_id"], "champion", "true")
        client.set_tag(champion["run_id"], "stage", "production")
        print(f"\n🏆 Champion : {champion['config']}  (BLEU={champion['bleu_mean']:.3f})")

    # ── Model Registry (3 modèles externes en production) ──
    print("\nEnregistrement du Model Registry…")
    production_models = [
        {
            "name":        "whisper-stt",
            "description": "Speech-to-Text — Faster-Whisper. Téléchargé depuis HuggingFace à la volée.",
            "tags":        {"provider": "huggingface", "type": "stt", "production_version": "large-v3"},
        },
        {
            "name":        "llama-translation",
            "description": "LLM de traduction — Llama 3.1 8B Instant via Groq API.",
            "tags":        {"provider": "groq", "type": "llm", "production_version": "groq/llama-3.1-8b-instant"},
        },
        {
            "name":        "voxtral-tts",
            "description": "Text-to-Speech — Mistral Voxtral via API.",
            "tags":        {"provider": "mistral", "type": "tts", "production_version": "mistral-voxtral"},
        },
    ]

    for m in production_models:
        try:
            try:
                client.create_registered_model(name=m["name"], description=m["description"])
                print(f"  ✓ Registered model créé : {m['name']}")
            except Exception:
                # Existe déjà — update description
                client.update_registered_model(name=m["name"], description=m["description"])
                print(f"  ✓ Registered model mis à jour : {m['name']}")
            for k, v in m["tags"].items():
                client.set_registered_model_tag(m["name"], k, v)
            client.set_registered_model_tag(m["name"], "last_sync", datetime.utcnow().isoformat())
        except Exception as e:
            print(f"  ✗ Erreur {m['name']}: {e}")

    print(f"\nTerminé — ouvre {MLFLOW_TRACKING_URI} pour visualiser.")
    print(f"  • Expérience : {EXPERIMENT_NAME} → {len(config_summaries)} configurations")
    print(f"  • Champion taggué : trie par tag.champion DESC")


if __name__ == "__main__":
    main()
