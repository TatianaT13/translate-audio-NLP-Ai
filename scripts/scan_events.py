"""
scan_events.py — Scanner d'événements trafic.

Lit les transcriptions Whisper, détecte les incidents par regex
et notifie via console, macOS et/ou webhook HTTP.

Usage :
  python scripts/scan_events.py --once --min-severity low
  python scripts/scan_events.py --watch --interval 60 --macos
  python scripts/scan_events.py --watch --webhook-url http://localhost:5000/alert
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from flash_nlp.analysis.event_extractor import extract_events, severity_rank
from flash_nlp.analysis.notifier import dispatch
from flash_nlp.io import ensure_dir, load_json, save_json

# Regex pour extraire le timestamp depuis le chemin du fichier audio
# Exemple : 2026-01-23/nord/flash_nord_20260123_164916.mp3 → 20260123_164916
_TS_RE = re.compile(r"flash_\w+?_(\d{8}_\d{4,6})\.mp3")


def _parse_timestamp(source_file: str) -> str:
    m = _TS_RE.search(source_file)
    return m.group(1) if m else "?"


def _parse_zone(source_file: str) -> str:
    parts = Path(source_file).parts
    return parts[1] if len(parts) >= 2 else "?"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scan_once(
    transcripts_dir: Path,
    state_path: Path,
    alerts_dir: Path,
    min_severity: str,
    macos: bool,
    webhook_url: str,
    verbose: bool = True,
) -> int:
    """
    Traite les nouveaux fichiers depuis le dernier scan.
    Retourne le nombre d'événements émis.
    """
    index_path = transcripts_dir / "index.json"
    if not index_path.exists():
        if verbose:
            print(f"[scan] index introuvable : {index_path}")
        return 0

    index: dict = load_json(index_path)
    state: dict = load_json(state_path)
    already_done: set = set(state.get("processed", []))

    min_rank = severity_rank(min_severity)
    total_events = 0

    for source_rel, meta in index.items():
        if source_rel in already_done:
            continue

        txt_filename = meta.get("output_txt")
        if not txt_filename:
            continue

        txt_path = transcripts_dir / txt_filename
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            already_done.add(source_rel)
            continue

        zone = _parse_zone(source_rel)
        timestamp = _parse_timestamp(source_rel)

        events = extract_events(text, zone=zone, source_file=source_rel, timestamp=timestamp)
        filtered = [e for e in events if severity_rank(e.severity) >= min_rank]

        for event in filtered:
            dispatch(event, alerts_dir=alerts_dir, macos=macos, webhook_url=webhook_url or None)
            total_events += 1

        already_done.add(source_rel)

    # Sauvegarde l'état
    state["processed"] = sorted(already_done)
    state["last_scan_utc"] = _utc_now()
    ensure_dir(state_path.parent)
    save_json(state_path, state)

    return total_events


def main():
    ap = argparse.ArgumentParser(description="Scanner d'événements trafic")
    ap.add_argument(
        "--input", default="outputs/transcripts_whisper",
        help="Dossier des transcriptions Whisper",
    )
    ap.add_argument(
        "--alerts-dir", default="outputs",
        help="Dossier pour alerts.jsonl et scan_state.json",
    )
    ap.add_argument(
        "--watch", action="store_true",
        help="Polling continu",
    )
    ap.add_argument(
        "--once", action="store_true",
        help="Une seule passe puis quitter",
    )
    ap.add_argument(
        "--interval", type=int, default=60,
        help="Secondes entre chaque scan en mode --watch (défaut : 60)",
    )
    ap.add_argument(
        "--min-severity", default="low", choices=["low", "medium", "high"],
        help="Niveau minimum pour déclencher une alerte",
    )
    ap.add_argument(
        "--webhook-url", default=os.environ.get("SCAN_WEBHOOK_URL", ""),
        help="URL POST pour les alertes webhook (env: SCAN_WEBHOOK_URL)",
    )
    ap.add_argument(
        "--macos", action="store_true",
        default=bool(os.environ.get("SCAN_MACOS_NOTIFY")),
        help="Activer les notifications macOS (env: SCAN_MACOS_NOTIFY=1)",
    )
    args = ap.parse_args()

    transcripts_dir = Path(args.input)
    alerts_dir = Path(args.alerts_dir)
    state_path = alerts_dir / "scan_state.json"

    def run_scan():
        n = scan_once(
            transcripts_dir=transcripts_dir,
            state_path=state_path,
            alerts_dir=alerts_dir,
            min_severity=args.min_severity,
            macos=args.macos,
            webhook_url=args.webhook_url,
        )
        if n == 0:
            print(f"[{_utc_now()}] Aucun nouvel événement.")
        else:
            print(f"[{_utc_now()}] {n} événement(s) émis.")

    if args.once or not args.watch:
        run_scan()
        return

    print(f"[scan] Mode watch — intervalle {args.interval}s | sévérité min={args.min_severity}")
    try:
        while True:
            run_scan()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[scan] Arrêt.")


if __name__ == "__main__":
    main()
