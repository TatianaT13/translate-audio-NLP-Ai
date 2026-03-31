"""
Import des 84 runs existants (results.csv) dans Langfuse.
Utilise l'API REST /api/public/ingestion (compatible toutes versions SDK).

Usage :
    python3 scripts/langfuse_import.py
"""

import csv
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "outputs" / "experiments" / "results.csv"


def main():
    import httpx

    pub_key  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec_key  = os.getenv("LANGFUSE_SECRET_KEY", "")
    host     = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not pub_key or not sec_key:
        print("Erreur : LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY manquantes dans .env")
        sys.exit(1)

    # Vérifie l'auth
    r = httpx.get(f"{host}/api/public/health", auth=(pub_key, sec_key), timeout=10)
    if not r.is_success:
        print(f"Erreur d'authentification Langfuse : HTTP {r.status_code}")
        sys.exit(1)
    print("Connexion Langfuse OK")

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8"), delimiter=";"))
    print(f"Import de {len(rows)} runs vers Langfuse...")

    ok = 0
    with httpx.Client(auth=(pub_key, sec_key), timeout=30) as client:
        for i, row in enumerate(rows, 1):
            try:
                tid = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                comment = f"{row['audio']} | {row['whisper_model']} | {row['llm_model']} | {row['prompt_version']}"

                scores_data = [
                    ("latency_total_ms", float(row["latency_total_ms"])),
                    ("latency_stt_ms",   float(row["latency_stt_ms"])),
                    ("latency_llm_ms",   float(row["latency_llm_ms"])),
                    ("language_prob",    float(row["language_prob"])),
                ]
                bleu = float(row["bleu"])
                if bleu >= 0:
                    scores_data.append(("bleu", bleu))

                batch = [
                    # Créer la trace
                    {
                        "id":        str(uuid.uuid4()),
                        "type":      "trace-create",
                        "timestamp": now,
                        "body": {
                            "id":        tid,
                            "name":      "translation",
                            "metadata": {
                                "audio":          row["audio"],
                                "whisper_model":  row["whisper_model"],
                                "llm_model":      row["llm_model"],
                                "prompt_version": row["prompt_version"],
                            },
                        },
                    },
                    # Scores
                    *[
                        {
                            "id":        str(uuid.uuid4()),
                            "type":      "score-create",
                            "timestamp": now,
                            "body": {
                                "id":       str(uuid.uuid4()),
                                "traceId":  tid,
                                "name":     name,
                                "value":    value,
                                "dataType": "NUMERIC",
                                "comment":  comment,
                            },
                        }
                        for name, value in scores_data
                    ],
                ]

                resp = client.post(f"{host}/api/public/ingestion", json={"batch": batch})
                if resp.is_success:
                    ok += 1
                else:
                    print(f"  Erreur run {i}: HTTP {resp.status_code} — {resp.text[:80]}")

                if i % 10 == 0:
                    print(f"  {i}/{len(rows)} importés...")

            except Exception as e:
                print(f"  Erreur run {i}: {e}")

    print(f"\nTerminé — {ok}/{len(rows)} traces envoyées à Langfuse.")
    print("Les données apparaissent dans le dashboard sous quelques secondes.")


if __name__ == "__main__":
    main()
