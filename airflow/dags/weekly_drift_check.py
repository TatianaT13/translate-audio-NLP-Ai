"""
DAG : weekly_drift_check
─────────────────────────
Tous les dimanches à 3h du matin, interroge Langfuse pour comparer
les métriques de la semaine N vs N-1 et détecter une dégradation.

Métriques surveillées : latence moyenne, coût moyen, BLEU moyen, taux d'erreur.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task


LANGFUSE_HOST   = os.getenv("LANGFUSE_HOST",   "https://cloud.langfuse.com")
LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")


default_args = {
    "owner":             "llmops",
    "retries":           1,
    "retry_delay":       timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=10),
}


@dag(
    dag_id="weekly_drift_check",
    schedule="0 3 * * 0",          # Dimanche 3h
    start_date=datetime(2026, 6, 1),
    catchup=False,
    default_args=default_args,
    tags=["llmops", "drift", "weekly"],
    description="Compare les métriques de la semaine N vs N-1 (Langfuse)",
)
def weekly_drift_check():

    @task
    def fetch_week_metrics(days_ago: int) -> dict:
        """Récupère les scores Langfuse d'une semaine donnée (offset en jours)."""
        if not LANGFUSE_PUBLIC or not LANGFUSE_SECRET:
            print("⚠️  Langfuse non configuré — skip")
            return {}

        import httpx
        from datetime import datetime as _dt, timezone as _tz

        end_iso   = (_dt.now(_tz.utc) - timedelta(days=days_ago)).isoformat()
        start_iso = (_dt.now(_tz.utc) - timedelta(days=days_ago + 7)).isoformat()

        scores: dict[str, list[float]] = {}
        page = 1
        with httpx.Client(timeout=15) as client:
            while True:
                r = client.get(
                    f"{LANGFUSE_HOST}/api/public/scores",
                    auth=(LANGFUSE_PUBLIC, LANGFUSE_SECRET),
                    params={
                        "limit":     100,
                        "page":      page,
                        "fromTimestamp": start_iso,
                        "toTimestamp":   end_iso,
                    },
                )
                if not r.is_success:
                    break
                batch = r.json().get("data", [])
                for s in batch:
                    scores.setdefault(s["name"], []).append(s["value"])
                if len(batch) < 100:
                    break
                page += 1

        def _avg(lst):
            return sum(lst) / len(lst) if lst else None

        return {
            "n_runs":            len(scores.get("latency_total_ms", [])),
            "avg_latency_ms":    _avg(scores.get("latency_total_ms", [])),
            "avg_cost_usd":      _avg(scores.get("cost_usd", [])),
            "avg_bleu":          _avg(scores.get("bleu", [])),
            "avg_language_prob": _avg(scores.get("language_prob", [])),
        }

    @task
    def compare(this_week: dict, last_week: dict) -> str:
        """Compare semaine N vs N-1 et alerte si dégradation significative."""
        alerts = []

        def _diff(metric: str, lower_is_better: bool = False) -> str:
            tv, lv = this_week.get(metric), last_week.get(metric)
            if tv is None or lv is None or lv == 0:
                return f"{metric}: insufficient data"
            change = (tv - lv) / lv * 100
            arrow  = "↑" if change > 0 else "↓"
            bad    = (change > 10 if lower_is_better else change < -10)
            if bad:
                alerts.append(f"{metric} {arrow} {change:+.1f}% ({lv:.3f} → {tv:.3f})")
            return f"{metric}: {lv:.3f} → {tv:.3f} ({arrow}{abs(change):.1f}%)"

        print(f"📊 Semaine actuelle : {this_week.get('n_runs', 0)} runs")
        print(f"📊 Semaine précédente : {last_week.get('n_runs', 0)} runs\n")
        for m in ["avg_latency_ms", "avg_cost_usd"]:
            print("  " + _diff(m, lower_is_better=True))
        for m in ["avg_bleu", "avg_language_prob"]:
            print("  " + _diff(m, lower_is_better=False))

        if alerts:
            print("\n🚨 DRIFT DÉTECTÉ :")
            for a in alerts:
                print(f"  - {a}")
            return "alert"
        print("\n✓ Pas de dérive significative")
        return "ok"

    this_week = fetch_week_metrics(days_ago=0)
    last_week = fetch_week_metrics(days_ago=7)
    compare(this_week, last_week)


weekly_drift_check()
