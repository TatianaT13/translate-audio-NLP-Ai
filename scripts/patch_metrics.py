"""
Recalcule BLEU / METEOR / WER pour les runs où ces valeurs sont à -1
en réutilisant source_text et translation déjà stockés dans results.csv.
Aucun appel STT/LLM — lecture seule + réécriture CSV.

Usage :
    .venv.nosync/bin/python scripts/patch_metrics.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_golden import compute_bleu, compute_meteor, compute_wer

ROOT        = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "outputs" / "experiments" / "results.csv"


def main() -> None:
    if not RESULTS_CSV.exists():
        print("results.csv introuvable.")
        return

    with RESULTS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        rows = list(reader)

    patched = 0
    for row in rows:
        audio_stem = Path(row["audio"]).stem
        lang       = row["target_lang"]

        def _is_missing(v: str) -> bool:
            return v.strip() in ("-1.0", "-1", "")

        if _is_missing(row.get("bleu", "")):
            b = compute_bleu(row["translation"], audio_stem, lang)
            if b >= 0:
                row["bleu"] = str(b)
                patched += 1

        if _is_missing(row.get("meteor", "")):
            m = compute_meteor(row["translation"], audio_stem, lang)
            if m >= 0:
                row["meteor"] = str(m)
                patched += 1

        if _is_missing(row.get("wer", "")):
            w = compute_wer(row["source_text"], audio_stem)
            if w >= 0:
                row["wer"] = str(w)
                patched += 1

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{patched} métriques recalculées sur {len(rows)} runs.")
    print(f"CSV mis à jour : {RESULTS_CSV}")


if __name__ == "__main__":
    main()
