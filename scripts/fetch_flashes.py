import time
import argparse
from pathlib import Path

from flash_nlp.acquisition import (
    DEFAULT_TZ_NAME,
    get_tz,
    now_str_local,
    fetch_once_conditional,
    fetch_once_legacy,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="flash_audio_archive", help="Dossier de sortie")
    ap.add_argument("--every", type=int, default=15, help="Périodicité en minutes (mode fixe)")
    ap.add_argument("--keep-days", type=int, default=30, help="Rétention en jours")
    ap.add_argument("--once", action="store_true", help="Télécharger une seule fois et quitter")
    ap.add_argument("--watch", action="store_true",
                    help="Mode surveillance: polling adaptatif avec requêtes conditionnelles")
    ap.add_argument("--min-every", type=int, default=10,
                    help="Intervalle minimum en secondes pour --watch (défaut: 10)")
    ap.add_argument("--max-every", type=int, default=900,
                    help="Intervalle maximum en secondes pour --watch (défaut: 900 = 15 min)")
    ap.add_argument("--tz", default=DEFAULT_TZ_NAME,
                    help="Fuseau IANA (ex: Europe/Paris) ou 'local' pour le fuseau système")
    args = ap.parse_args()

    try:
        tz = get_tz(args.tz)
    except Exception as e:
        raise SystemExit(f"Fuseau horaire invalide '{args.tz}': {e}")

    root = Path(args.root)

    if args.once:
        n = fetch_once_legacy(root, args.keep_days, tz)
        print(f"{now_str_local(tz)} | nouveaux fichiers: {n}")
        return

    if args.watch:
        sleep_s = max(1, int(args.min_every))
        cond_state = {}
        while True:
            try:
                n = fetch_once_conditional(root, args.keep_days, tz, cond_state)
                print(f"{now_str_local(tz)} | nouveaux fichiers: {n} | prochain scan dans {sleep_s}s")
                if n > 0:
                    sleep_s = max(1, int(args.min_every))
                else:
                    next_s = int(sleep_s * 1.5)
                    sleep_s = min(args.max_every, max(args.min_every, next_s))
            except Exception as e:
                print(f"{now_str_local(tz)} | ERREUR: {e}")
                sleep_s = max(1, int(args.min_every))
            time.sleep(sleep_s)
    else:
        every_s = max(1, args.every * 60)
        while True:
            try:
                n = fetch_once_legacy(root, args.keep_days, tz)
                print(f"{now_str_local(tz)} | nouveaux fichiers: {n}")
            except Exception as e:
                print(f"{now_str_local(tz)} | ERREUR: {e}")
            time.sleep(every_s)


if __name__ == "__main__":
    main()
