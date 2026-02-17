import os, time, hashlib, requests, csv, argparse, datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
 
URLS = {
    "nord":  "https://audio.autorouteinfo.fr/flash_nord.mp3",
    "sud":   "https://audio.autorouteinfo.fr/flash_sud.mp3",
    "ouest": "https://audio.autorouteinfo.fr/flash_ouest.mp3",
}
 
# --- Fuseau par défaut ---
DEFAULT_TZ_NAME = "Europe/Paris"
 
def get_tz(tz_name: str):
    if tz_name == "local":
        return dt.datetime.now().astimezone().tzinfo
    return ZoneInfo(tz_name)
 
def now_local(tz) -> dt.datetime:
    return dt.datetime.now(tz)
 
def now_str_local(tz) -> str:
    # Inclut l'offset pour lever l'ambiguïté dans les logs et l'index
    return now_local(tz).strftime("%Y-%m-%d %H:%M:%S%z")
 
def now_str_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
 
def ts_compact_local(tz) -> str:
    """
    Timestamp lisible en heure locale sans offset, format: YYYYmmdd_HHMM
    (Ex: 20260217_1744)
    """
    return now_local(tz).strftime("%Y%m%d_%H%M")
 
def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()
 
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
 
def parse_stamp_to_dt(stamp: str, tz) -> dt.datetime:
    """
    Convertit un stamp de type:
      - 'YYYYmmdd_HHMM+HHMM' (ancien format, avec offset)
      - 'YYYYmmdd_HHMM'      (nouveau format, sans offset, interprété en heure locale)
    en datetime aware pour comparaison.
    """
    # Ancien format avec offset
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d_%H%M%z")
    except Exception:
        pass
    # Nouveau format sans offset -> interprété en heure locale (aware)
    d_naive = dt.datetime.strptime(stamp, "%Y%m%d_%H%M")
    return d_naive.replace(tzinfo=tz)
 
def rotate(path: Path, keep_days: int, tz):
    if keep_days <= 0:
        return
    limit = now_local(tz) - dt.timedelta(days=keep_days)
    for f in path.glob("**/*.mp3"):
        # attend formats:
        #   flash_zone_YYYYmmdd_HHMM+HHMM.mp3 (legacy)
        #   flash_zone_YYYYmmdd_HHMM.mp3      (nouveau)
        try:
            parts = f.stem.rsplit("_", 2)  # [prefix..., YYYYmmdd, HHMM(+HHMM)]
            if len(parts) < 3:
                continue
            stamp = parts[-2] + "_" + parts[-1]
            d = parse_stamp_to_dt(stamp, tz)
            if d < limit:
                f.unlink(missing_ok=True)
        except Exception:
            # On ignore les noms inattendus
            pass
 
def write_index_row(index_csv: Path, row: list):
    new = not index_csv.exists()
    with index_csv.open("a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter=";")
        if new:
            # On enregistre heure locale (avec offset) ET UTC pour exploitation externe
            w.writerow(["datetime_local", "datetime_utc", "zone", "filename", "filesize", "md5"])
        w.writerow(row)
 
def dedupe_by_md5(day_dir: Path, new_md5: str) -> bool:
    """
    Retourne True si le dernier fichier du jour a déjà le même MD5 (donc doublon).
    """
    existing = sorted(day_dir.glob("flash_*.mp3"))
    latest = existing[-1] if existing else None
    if latest is not None and latest.exists():
        try:
            with latest.open("rb") as lf:
                if md5_bytes(lf.read()) == new_md5:
                    return True
        except Exception:
            pass
    return False
 
def save_audio(root_dir: Path, zone: str, content: bytes, tz) -> Path:
    day_dir = root_dir / now_local(tz).strftime("%Y-%m-%d") / zone
    ensure_dir(day_dir)
    stamp = ts_compact_local(tz)  # ex: 20260217_1744 (sans offset dans le nom)
    fpath = day_dir / f"flash_{zone}_{stamp}.mp3"
 
    # Optionnel: anti-collision de nom si deux fichiers la même minute
    # Décommente pour éviter tout écrasement sans changer le nommage demandé.
    #i = 1
    #base = fpath
    #while fpath.exists():
    #    fpath = base.with_name(base.stem + f"-{i}" + base.suffix)
    #    i += 1
 
    with fpath.open("wb") as fp:
        fp.write(content)
    return fpath
 
def fetch_once_conditional(root_dir: Path, keep_days: int, tz, cond_state: dict) -> int:
    """
    Fait une passe sur toutes les zones avec GET conditionnel si possible.
    cond_state: dict persistant { zone: {"etag": str|None, "lm": str|None} }
    """
    ensure_dir(root_dir)
    index_csv = root_dir / "index.csv"
    added = 0
 
    for zone, url in URLS.items():
        st = cond_state.setdefault(zone, {"etag": None, "lm": None})
 
        headers = {}
        if st.get("etag"):
            headers["If-None-Match"] = st["etag"]
        if st.get("lm"):
            headers["If-Modified-Since"] = st["lm"]
 
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except Exception as e:
            # Log et passe à la zone suivante
            print(f"{now_str_local(tz)} | {zone} | ERREUR requête: {e}")
            continue
 
        # Mémorise les en-têtes si présents (pour les prochains cycles)
        etag = r.headers.get("ETag")
        lm = r.headers.get("Last-Modified")
        if etag:
            st["etag"] = etag
        if lm:
            st["lm"] = lm
 
        if r.status_code == 304:
            # Pas de changement côté serveur
            continue
 
        if r.status_code != 200:
            print(f"{now_str_local(tz)} | {zone} | HTTP {r.status_code} (ignoré)")
            continue
 
        content = r.content
        if len(content) < 10_000:
            # Réponses non audio ou erreurs
            continue
 
        m = md5_bytes(content)
 
        # Dédoublonnage de sécurité (même si 200, au cas où)
        day_dir = root_dir / now_local(tz).strftime("%Y-%m-%d") / zone
        ensure_dir(day_dir)
        if dedupe_by_md5(day_dir, m):
            continue
 
        fpath = save_audio(root_dir, zone, content, tz)
        write_index_row(
            index_csv,
            [
                now_str_local(tz),   # datetime_local (avec offset)
                now_str_utc(),       # datetime_utc
                zone,
                str(fpath.relative_to(root_dir)),
                fpath.stat().st_size,
                m,
            ],
        )
        added += 1
 
    rotate(root_dir, keep_days, tz)
    return added
 
def fetch_once_legacy(root_dir: Path, keep_days: int, tz) -> int:
    """
    Mode historique: GET direct (sans conditionnel), dédup par MD5.
    """
    cond_state_dummy = {}
    return fetch_once_conditional(root_dir, keep_days, tz, cond_state_dummy)
 
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
        cond_state = {}  # persiste en mémoire du processus
        while True:
            try:
                n = fetch_once_conditional(root, args.keep_days, tz, cond_state)
                print(f"{now_str_local(tz)} | nouveaux fichiers: {n} | prochain scan dans {sleep_s}s")
                # Adaptation de l'intervalle
                if n > 0:
                    sleep_s = max(1, int(args.min_every))  # on repart au minimum
                else:
                    # backoff progressif jusqu'au plafond
                    next_s = int(sleep_s * 1.5)
                    sleep_s = min(args.max_every, max(args.min_every, next_s))
            except Exception as e:
                print(f"{now_str_local(tz)} | ERREUR: {e}")
                # en cas d'erreur, dors au minimum
                sleep_s = max(1, int(args.min_every))
            time.sleep(sleep_s)
    else:
        # Mode périodique fixe (compat)
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
