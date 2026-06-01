"""
Importe les 84 runs historiques (results.csv) dans MLflow et enregistre
les modèles en production dans le Model Registry.

Usage:
    pip install mlflow==2.17.2
    python scripts/mlflow_register.py

Pré-requis :
    docker compose up mlflow -d
"""

import csv
import os
import sys
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


def main():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Connexion MLflow → {MLFLOW_TRACKING_URI}")

    # Création / récupération de l'expérience
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Expérience : {exp.name} (id={exp.experiment_id})")

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8"), delimiter=";"))
    print(f"Import de {len(rows)} runs vers MLflow…")

    ok = 0
    for i, row in enumerate(rows, 1):
        try:
            run_name = row.get("run_id") or f"run_{i}"
            with mlflow.start_run(run_name=run_name, experiment_id=exp.experiment_id):
                # Paramètres = combinaison testée
                mlflow.log_params({
                    "whisper_model":  row.get("whisper_model", ""),
                    "llm_model":      row.get("llm_model", ""),
                    "prompt_version": row.get("prompt_version", ""),
                    "target_lang":    row.get("target_lang", ""),
                    "audio":          row.get("audio", ""),
                    "zone":           row.get("zone", ""),
                })

                # Métriques
                for metric in (
                    "language_prob", "latency_conv_ms", "latency_stt_ms",
                    "latency_llm_ms", "latency_total_ms",
                    "bleu", "meteor", "wer", "tts_wer",
                ):
                    val = row.get(metric, "").strip()
                    if val in ("", "-1.0", "-1", "n/a"):
                        continue
                    try:
                        mlflow.log_metric(metric, float(val))
                    except ValueError:
                        pass

                # Texte source + traduction comme tags (recherche facile)
                mlflow.set_tag("source_text",  row.get("source_text",  "")[:500])
                mlflow.set_tag("translation",  row.get("translation",  "")[:500])
                mlflow.set_tag("timestamp",    row.get("timestamp",    ""))

            ok += 1
            if i % 10 == 0:
                print(f"  {i}/{len(rows)} runs importés…")
        except Exception as e:
            print(f"  Erreur run {i}: {e}")

    print(f"\n{ok}/{len(rows)} runs enregistrés dans MLflow.")

    # ── Enregistrement des modèles en production (Model Registry) ────────────
    # On enregistre des "external references" car nos modèles sont externes
    # (Groq API, Mistral API) ou téléchargés à la volée (Whisper depuis HF).
    print("\nEnregistrement des modèles dans le Model Registry…")

    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    production_models = [
        {
            "name":        "whisper-stt",
            "description": "Speech-to-Text — Faster-Whisper. Téléchargé depuis HuggingFace à la volée.",
            "version_tag": "large-v3",
            "tags":        {"provider": "huggingface", "type": "stt", "language": "fr"},
        },
        {
            "name":        "llama-translation",
            "description": "LLM de traduction — Llama 3.1 8B Instant via Groq API.",
            "version_tag": "groq/llama-3.1-8b-instant",
            "tags":        {"provider": "groq", "type": "llm", "prompt_version": "v1.1"},
        },
        {
            "name":        "voxtral-tts",
            "description": "Text-to-Speech — Mistral Voxtral via API.",
            "version_tag": "mistral-voxtral",
            "tags":        {"provider": "mistral", "type": "tts"},
        },
    ]

    for m in production_models:
        try:
            # Crée le registered model si pas existant
            try:
                client.create_registered_model(
                    name=m["name"],
                    description=m["description"],
                    tags=m["tags"],
                )
                print(f"  ✓ Registered model créé : {m['name']}")
            except Exception:
                # Existe déjà
                pass
            # Note : pour créer une version, il faut un run avec un artifact.
            # On set juste les tags pour identifier le modèle production.
            client.set_registered_model_tag(m["name"], "production_version", m["version_tag"])
            client.set_registered_model_tag(m["name"], "registered_at", datetime.utcnow().isoformat())
            print(f"    → tag production_version={m['version_tag']}")
        except Exception as e:
            print(f"  ✗ Erreur {m['name']}: {e}")

    print(f"\nTerminé — ouvre {MLFLOW_TRACKING_URI} pour visualiser.")


if __name__ == "__main__":
    main()
