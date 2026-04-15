"""
Importe toutes les métriques du results.csv vers Langfuse.
Chaque ligne du CSV devient une trace Langfuse avec scores :
BLEU, METEOR, WER, TTS_WER, latency_*, language_prob.

Usage :
    .venv.nosync/bin/python scripts/import_metrics_to_langfuse.py
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

from langfuse import Langfuse

ROOT        = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "outputs" / "experiments" / "results.csv"


def _parse(v: str) -> float | None:
    v = v.strip()
    if v in ("", "-1.0", "-1", "n/a"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY manquants dans .env")
        sys.exit(1)

    if not RESULTS_CSV.exists():
        print("results.csv introuvable.")
        sys.exit(1)

    lf = Langfuse()
    with RESULTS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    print(f"Importation de {len(rows)} runs vers Langfuse…")

    score_names = ["bleu", "meteor", "wer", "tts_wer",
                   "latency_total_ms", "latency_stt_ms", "latency_llm_ms",
                   "language_prob"]

    total_scores = 0
    for i, row in enumerate(rows, 1):
        tid = lf.create_trace_id()
        comment = f"{row['audio']} | {row['whisper_model']} | {row['llm_model']} | {row['prompt_version']}"

        pushed = 0
        for name in score_names:
            v = _parse(row.get(name, ""))
            if v is None:
                continue
            lf.create_score(trace_id=tid, name=name, value=v, comment=comment)
            pushed += 1

        total_scores += pushed
        if i % 5 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] {row['run_id'][:60]} → {pushed} scores")

    lf.flush()
    print(f"\nTerminé : {total_scores} scores envoyés vers Langfuse.")


if __name__ == "__main__":
    main()
