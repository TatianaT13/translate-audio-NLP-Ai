"""
Import des 84 runs existants (results.csv) dans Langfuse.
Ne relance aucun LLM — lit juste le CSV et crée les traces.

Usage :
    python scripts/langfuse_import.py
"""

import csv
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "outputs" / "experiments" / "results.csv"


def main():
    from langfuse import Langfuse
    lf = Langfuse()

    if not lf.auth_check():
        print("Erreur d'authentification Langfuse. Vérifie LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans .env")
        sys.exit(1)

    rows = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8"), delimiter=";"))
    print(f"Import de {len(rows)} runs vers Langfuse...")

    for i, row in enumerate(rows, 1):
        try:
            tid = lf.create_trace_id()

            scores = [
                ("latency_total_ms", float(row["latency_total_ms"])),
                ("latency_stt_ms",   float(row["latency_stt_ms"])),
                ("latency_llm_ms",   float(row["latency_llm_ms"])),
                ("language_prob",    float(row["language_prob"])),
            ]
            bleu = float(row["bleu"])
            if bleu >= 0:
                scores.append(("bleu", bleu))

            for name, value in scores:
                lf.create_score(
                    trace_id=tid,
                    name=name,
                    value=value,
                    comment=f"{row['audio']} | {row['whisper_model']} | {row['llm_model']} | {row['prompt_version']}",
                )

            if i % 10 == 0:
                lf.flush()
                print(f"  {i}/{len(rows)} importés...")

        except Exception as e:
            print(f"  Erreur run {i}: {e}")

    lf.flush()
    print(f"\nTerminé — {len(rows)} traces dans Langfuse.")
    print("Ouvre https://cloud.langfuse.com -> Scores pour visualiser.")


if __name__ == "__main__":
    main()
