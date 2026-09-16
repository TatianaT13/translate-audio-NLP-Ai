"""
Benchmark de l'economie unitaire — pull Langfuse et sort un rapport
par type de job pour calibrer le pricing en credits.

Sortie :
  - Rapport en console (Markdown-friendly)
  - CSV detaille : outputs/benchmarks/costs_YYYYMMDD.csv
  - Recommandations de prix (formule marge cible 75/80 %)

Le script utilise l'API publique Langfuse et n'a pas besoin d'acces
serveur direct : il tourne depuis n'importe ou avec les credentials
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.

Usage :
    .venv.nosync/bin/python scripts/benchmark_costs.py
    .venv.nosync/bin/python scripts/benchmark_costs.py --days 30
    .venv.nosync/bin/python scripts/benchmark_costs.py --days 90 --min-jobs 10

Pre-requis : rejouer au moins 20 jobs varies (1 min, 5 min, 30 min, 1 h)
via l'app ou via un script de charge avant de lancer ce benchmark.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "outputs" / "benchmarks"

LANGFUSE_HOST   = os.getenv("LANGFUSE_HOST",   "https://cloud.langfuse.com")
LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")

# Marge cible pour calculer le prix recommande selon la formule du plan tarifaire :
#   prix = cout_reel / (1 - marge_cible)
TARGET_MARGIN = 0.80  # 80 % — objectif SaaS

# Cout d'infrastructure alloue par heure d'audio traitee (STT self-hosted +
# gateway + observability + amortissement Whisper large-v3 telecharge).
# Estimation grossiere a affiner apres 3 mois d'exploitation reelle.
INFRA_COST_PER_HOUR_EUR = 0.10


# ── Client Langfuse REST ─────────────────────────────────────────────────────
def _langfuse_get(path: str, params: dict | None = None) -> dict:
    if not LANGFUSE_PUBLIC or not LANGFUSE_SECRET:
        print("ERREUR : LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY manquants dans .env")
        sys.exit(1)
    url = f"{LANGFUSE_HOST}{path}"
    r = httpx.get(url, auth=(LANGFUSE_PUBLIC, LANGFUSE_SECRET), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_traces(days: int) -> list[dict]:
    """Recupere toutes les traces 'translation' des N derniers jours."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    traces: list[dict] = []
    page = 1
    while True:
        resp = _langfuse_get("/api/public/traces", params={
            "name": "translation",
            "fromTimestamp": since,
            "limit": 100,
            "page": page,
        })
        batch = resp.get("data", [])
        traces.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        if page > 50:  # garde-fou
            break
    return traces


# ── Extraction cout + duree par trace ────────────────────────────────────────
def extract_job_row(trace: dict) -> dict | None:
    """Extrait les champs interessants d'une trace pour l'analyse.

    Le pipeline logue le cout dans metadata.cost_usd et les latences aussi.
    On calcule la duree approximative de l'audio d'apres audio_kb (16kHz
    mono, ~32 KB/s) — grossier mais suffisant pour bencher.
    """
    meta = trace.get("metadata") or {}
    cost_usd = meta.get("cost_usd") or 0
    if cost_usd is None:
        cost_usd = 0

    audio_kb = None
    inp = trace.get("input") or {}
    if isinstance(inp, dict):
        audio_kb = inp.get("audio_kb")

    # Estimation grossiere : WAV 16kHz mono ~32 KB/s
    duration_sec = None
    if audio_kb:
        duration_sec = round(audio_kb / 32, 1)

    return {
        "trace_id":    trace.get("id"),
        "created_at":  trace.get("timestamp") or trace.get("createdAt"),
        "target_lang": meta.get("target_lang"),
        "whisper":     meta.get("whisper_model"),
        "llm":         meta.get("llm_model"),
        "audio_kb":    audio_kb,
        "duration_sec": duration_sec,
        "latency_ms":  meta.get("total_latency_ms"),
        "cost_usd":    float(cost_usd),
    }


# ── Agregation + statistiques ────────────────────────────────────────────────
def bucket_duration(sec: float | None) -> str:
    if sec is None:
        return "unknown"
    if sec < 60:
        return "< 1 min"
    if sec < 300:
        return "1-5 min"
    if sec < 1800:
        return "5-30 min"
    if sec < 3600:
        return "30-60 min"
    return "> 1 h"


def summarize_bucket(rows: list[dict], usd_to_eur: float) -> dict:
    """Retourne mediane / P90 / max pour un bucket."""
    costs_usd = [r["cost_usd"] for r in rows if r["cost_usd"] > 0]
    durations = [r["duration_sec"] for r in rows if r["duration_sec"]]

    if not costs_usd:
        return {"n": 0}

    cost_median_usd = statistics.median(costs_usd)
    cost_p90_usd = _percentile(costs_usd, 90)
    cost_max_usd = max(costs_usd)

    # Cout par minute d'audio (moyenne pondérée)
    total_cost = sum(costs_usd)
    total_min = sum(d / 60 for d in durations) if durations else None
    cost_per_min_usd = (total_cost / total_min) if total_min else None

    return {
        "n":                 len(costs_usd),
        "duration_median":   statistics.median(durations) if durations else None,
        "cost_median_usd":   cost_median_usd,
        "cost_median_eur":   cost_median_usd * usd_to_eur,
        "cost_p90_usd":      cost_p90_usd,
        "cost_p90_eur":      cost_p90_usd * usd_to_eur,
        "cost_max_usd":      cost_max_usd,
        "cost_per_min_usd":  cost_per_min_usd,
        "cost_per_min_eur":  (cost_per_min_usd * usd_to_eur) if cost_per_min_usd else None,
    }


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── Rapport ──────────────────────────────────────────────────────────────────
def print_report(buckets: dict, total_jobs: int, days: int, usd_to_eur: float) -> None:
    print()
    print(f"{'='*72}")
    print(f"  Benchmark economie unitaire — {total_jobs} jobs sur les {days} derniers jours")
    print(f"  Taux USD/EUR retenu : {usd_to_eur:.4f}")
    print(f"{'='*72}")
    print()

    if total_jobs == 0:
        print("Aucune trace 'translation' trouvee. Rejouer quelques jobs via l'app.")
        return

    # Tableau par bucket de duree
    print("## Cout par bucket de duree (audio traite)")
    print()
    header = f"| {'Bucket':<12} | {'n':>3} | {'Duree med':>10} | {'Cout med':>10} | {'Cout P90':>10} | {'Cout/min':>10} |"
    sep    = f"|{'-'*14}|{'-'*5}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|"
    print(header)
    print(sep)
    order = ["< 1 min", "1-5 min", "5-30 min", "30-60 min", "> 1 h", "unknown"]
    for bucket in order:
        s = buckets.get(bucket, {})
        if not s or s.get("n", 0) == 0:
            continue
        dm  = f"{s['duration_median']:>7.0f} s" if s.get('duration_median') else "     —  "
        cm  = f"{s['cost_median_eur']*100:>7.3f} ct" if s.get('cost_median_eur') else "     —  "
        c90 = f"{s['cost_p90_eur']*100:>7.3f} ct"    if s.get('cost_p90_eur')    else "     —  "
        cpm = f"{s['cost_per_min_eur']*100:>7.3f} ct" if s.get('cost_per_min_eur') else "     —  "
        print(f"| {bucket:<12} | {s['n']:>3} | {dm} | {cm} | {c90} | {cpm} |")

    # Recommandations de prix
    print()
    print("## Recommandations tarifaires (marge cible 80 %)")
    print()
    print("Formule : prix = cout_reel / (1 - 0,80) = cout_reel × 5")
    print()

    # Cout moyen par minute agrégé (toutes durees confondues)
    all_costs_per_min = [
        s.get("cost_per_min_eur", 0) or 0
        for s in buckets.values() if s.get("n", 0) > 0
    ]
    if all_costs_per_min:
        median_cost_per_min = statistics.median([c for c in all_costs_per_min if c > 0])
        # Ajout du cout infra alloue
        infra_per_min = INFRA_COST_PER_HOUR_EUR / 60
        total_cost_per_min = median_cost_per_min + infra_per_min
        recommended_price = total_cost_per_min / (1 - TARGET_MARGIN)

        print(f"  Cout API median par minute d'audio  : {median_cost_per_min*100:.3f} ct EUR")
        print(f"  Cout infra alloue par minute        : {infra_per_min*100:.3f} ct EUR "
              f"(base {INFRA_COST_PER_HOUR_EUR}€/h)")
        print(f"  Cout total median par minute        : {total_cost_per_min*100:.3f} ct EUR")
        print(f"  → Prix recommande par minute        : {recommended_price*100:.2f} ct EUR")
        print()

        # Calibrage des offres
        print("### Calibrage des offres Pro / Business (Traduction+voix = 2 credits/min)")
        print()
        for name, credits in [("Pro (19,90 €)", 200), ("Business (39,90 €)", 600)]:
            minutes_audio = credits / 2  # traduction + voix
            cost_full_usage_eur = minutes_audio * total_cost_per_min
            print(f"  {name}")
            print(f"    Credits inclus : {credits}")
            print(f"    Si 100 % utilises en audio traduit : {minutes_audio:.0f} min = "
                  f"{cost_full_usage_eur:.2f} € de cout API")
            price_eur = 19.90 if credits == 200 else 39.90
            worst_case_margin = (price_eur - cost_full_usage_eur) / price_eur * 100
            print(f"    Marge pire cas (100 % utilisation)  : {worst_case_margin:.1f} %")
            print()


def export_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark economie unitaire depuis Langfuse")
    p.add_argument("--days", type=int, default=30, help="Fenetre de temps (defaut 30)")
    p.add_argument("--usd-eur", type=float, default=0.92, help="Taux de conversion USD->EUR (defaut 0.92)")
    p.add_argument("--min-jobs", type=int, default=5, help="Seuil d'alerte 'trop peu de donnees'")
    args = p.parse_args()

    print(f"Fetch des traces 'translation' des {args.days} derniers jours…")
    traces = fetch_traces(args.days)
    print(f"  → {len(traces)} traces recuperees")

    rows = [r for r in (extract_job_row(t) for t in traces) if r is not None]

    if len(rows) < args.min_jobs:
        print(f"\n⚠️  Seulement {len(rows)} jobs, seuil recommande = {args.min_jobs}.")
        print("   Rejouer quelques traductions varies (1 min, 5 min, 30 min, 1 h) et")
        print("   relancer ce script pour un rapport plus fiable.")

    # Groupement par bucket
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[bucket_duration(r["duration_sec"])].append(r)

    # Summary
    bucket_stats = {name: summarize_bucket(rs, args.usd_eur) for name, rs in buckets.items()}
    print_report(bucket_stats, len(rows), args.days, args.usd_eur)

    # Export CSV
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = OUT_DIR / f"costs_{stamp}.csv"
    export_csv(rows, csv_path)
    print(f"CSV detaille ecrit : {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
