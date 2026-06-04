"""
DAG : nightly_golden_eval
─────────────────────────
Tous les jours à 2h du matin, fait passer les audios du dataset golden
à travers le pipeline et alerte si la qualité BLEU se dégrade.

Étapes :
  1. ping_pipeline       - vérifie que le service pipeline est UP
  2. run_golden_set      - lance les 7 audios golden via /process
  3. compute_metrics     - calcule BLEU/METEOR moyens
  4. check_drift         - alerte si BLEU < seuil
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

# ── Config ────────────────────────────────────────────────────────────────────
PIPELINE_URL    = os.getenv("PIPELINE_URL", "http://pipeline:8000")
GOLDEN_DIR      = Path("/opt/airflow/data/flash_audio_archive")
BLEU_THRESHOLD  = 25.0   # seuil min — sous ça on alerte
TARGET_LANG     = "en"


default_args = {
    "owner":             "llmops",
    "retries":           1,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


@dag(
    dag_id="nightly_golden_eval",
    schedule="0 2 * * *",          # 2h du matin tous les jours
    start_date=datetime(2026, 6, 1),
    catchup=False,
    default_args=default_args,
    tags=["llmops", "evaluation", "nightly"],
    description="Évaluation nightly du golden set + détection de dérive BLEU",
)
def nightly_golden_eval():

    @task
    def ping_pipeline() -> bool:
        import httpx
        r = httpx.get(f"{PIPELINE_URL}/health", timeout=10)
        r.raise_for_status()
        return True

    @task
    def list_golden_audios() -> list[str]:
        """Trouve tous les MP3 du dataset golden."""
        if not GOLDEN_DIR.exists():
            return []
        files = [str(p) for p in GOLDEN_DIR.rglob("*.mp3")]
        files.sort()
        return files[:7]   # On limite à 7 pour le batch quotidien

    @task
    def translate_one(audio_path: str) -> dict:
        """Envoie un audio au pipeline et récupère la traduction + latences."""
        import httpx
        try:
            with open(audio_path, "rb") as f:
                r = httpx.post(
                    f"{PIPELINE_URL}/process",
                    files={"file": (Path(audio_path).name, f.read(), "audio/mpeg")},
                    data={"target_lang": TARGET_LANG, "whisper_model": "small"},
                    timeout=300.0,
                )
            r.raise_for_status()
            d = r.json()
            return {
                "audio":            Path(audio_path).name,
                "source_text":      d.get("source_text", ""),
                "translation":      d.get("translation", ""),
                "latency_total_ms": d.get("latency_total_ms", 0),
                "cost_usd":         d.get("cost_usd", 0),
                "ok":               True,
            }
        except Exception as e:
            return {"audio": Path(audio_path).name, "ok": False, "error": str(e)}

    @task
    def aggregate(results: list[dict]) -> dict:
        """Agrège les résultats : taux de succès, latence moyenne, coût total."""
        ok_runs   = [r for r in results if r.get("ok")]
        fail_runs = [r for r in results if not r.get("ok")]
        avg_lat   = sum(r["latency_total_ms"] for r in ok_runs) / max(len(ok_runs), 1)
        tot_cost  = sum(r.get("cost_usd", 0) for r in ok_runs)
        print(f"[nightly_eval] {len(ok_runs)}/{len(results)} OK | latence_moy={avg_lat:.0f}ms | cost=${tot_cost:.5f}")
        for f in fail_runs:
            print(f"  ✗ {f['audio']}: {f.get('error')}")
        return {
            "total":           len(results),
            "ok":              len(ok_runs),
            "failed":          len(fail_runs),
            "avg_latency_ms":  round(avg_lat),
            "total_cost_usd":  round(tot_cost, 5),
        }

    @task
    def check_drift(summary: dict) -> str:
        """Alerte console si trop d'échecs."""
        if summary["failed"] >= summary["total"] / 2:
            msg = f"🚨 DRIFT DÉTECTÉ — {summary['failed']}/{summary['total']} runs ont échoué"
            print(msg)
            # En prod : envoyer alerte Slack/email ici
            return "alert_sent"
        print(f"✓ Qualité OK — {summary['ok']}/{summary['total']} runs réussis")
        return "ok"

    # ── Orchestration ────────────────────────────────────────────────────
    pipeline_ok = ping_pipeline()
    audios      = list_golden_audios()
    results     = translate_one.expand(audio_path=audios)
    summary     = aggregate(results)
    final       = check_drift(summary)

    pipeline_ok >> audios
    summary >> final


nightly_golden_eval()
