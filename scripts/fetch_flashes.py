import time, json, csv, argparse, datetime as dt
from pathlib import Path
import hashlib
import requests

URLS = {
    "nord":  "https://audio.autorouteinfo.fr/flash_nord.mp3",
    "sud":   "https://audio.autorouteinfo.fr/flash_sud.mp3",
    "ouest": "https://audio.autorouteinfo.fr/flash_ouest.mp3",
}

def now_utc():
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def ts_compact():
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))

def save_state(state_path: Path, state: dict):
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def write_index_row(index_csv: Path, row: list):
    new = not index_csv.exists()
    with index_csv.open("a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter=";")
        if new:
            w.writerow(["datetime_utc","zone","filename","filesize","md5","etag","last_modified"])
        w.writerow(row)

def rotate(root_dir: Path, keep_days: int):
    if keep_days <= 0:
        return
    limit = dt.datetime.utcnow() - dt.timedelta(days=keep_days)
    for day_dir in root_dir.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            d = dt.datetime.strptime(day_dir.name, "%Y-%m-%d")
        except Exception:
            continue
        if d < limit:
            for f in day_dir.glob("**/*.mp3"):
                f.unlink(missing_ok=True)

def head_meta(url: str) -> dict:
    r = requests.head(url, timeout=15, allow_redirects=True)
    r.raise_for_status()
    return {
        "etag": r.headers.get("ETag"),
        "last_modified": r.headers.get("Last-Modified"),
        "content_length": r.headers.get("Content-Length"),
    }

def fetch_content(url: str) -> bytes:
    r = requests.get(url, timeout=30, stream=True)
    r.raise_for_status()
    chunks = []
    size = 0
    for chunk in r.iter_content(chunk_size=1024 * 256):
        if not chunk:
            continue
        chunks.append(chunk)
        size += len(chunk)
        if size > 50 * 1024 * 1024:
            raise RuntimeError("File too large, abort")
    return b"".join(chunks)

def fetch_once(root_dir: Path, keep_days: int = 30) -> int:
    ensure_dir(root_dir)
    index_csv = root_dir / "index.csv"
    state_path = root_dir / "state.json"
    state = load_state(state_path)

    added = 0
    for zone, url in URLS.items():
        meta = head_meta(url)

        prev = state.get(zone, {})
        if prev.get("etag") and meta.get("etag") and prev["etag"] == meta["etag"]:
            continue
        if prev.get("last_modified") and meta.get("last_modified") and prev["last_modified"] == meta["last_modified"]:
            continue

        content = fetch_content(url)
        if len(content) < 10_000:
            continue

        stamp = ts_compact()
        day_dir = root_dir / dt.datetime.utcnow().strftime("%Y-%m-%d") / zone
        ensure_dir(day_dir)
        fname = f"flash_{zone}_{stamp}.mp3"
        fpath = day_dir / fname

        fpath.write_bytes(content)
        m = md5_file(fpath)

        if prev.get("md5") == m:
            fpath.unlink(missing_ok=True)
            state[zone] = {**meta, "md5": prev.get("md5")}
            continue

        rel = str(fpath.relative_to(root_dir))
        write_index_row(index_csv, [now_utc(), zone, rel, fpath.stat().st_size, m, meta.get("etag"), meta.get("last_modified")])

        state[zone] = {**meta, "md5": m}
        added += 1

    save_state(state_path, state)
    rotate(root_dir, keep_days)
    return added

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="flash_audio_archive")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--keep-days", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)

    while True:
        try:
            n = fetch_once(root, args.keep_days)
            print(f"{now_utc()} | nouveaux fichiers: {n}")
        except Exception as e:
            print(f"{now_utc()} | ERREUR: {e}")
        if args.once:
            break
        time.sleep(args.every * 60)