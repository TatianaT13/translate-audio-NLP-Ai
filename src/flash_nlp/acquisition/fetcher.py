import csv
import datetime as dt
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

URLS = {
    "nord":  "https://audio.autorouteinfo.fr/flash_nord.mp3",
    "sud":   "https://audio.autorouteinfo.fr/flash_sud.mp3",
    "ouest": "https://audio.autorouteinfo.fr/flash_ouest.mp3",
}

DEFAULT_TZ_NAME = "Europe/Paris"


def get_tz(tz_name: str):
    if tz_name == "local":
        return dt.datetime.now().astimezone().tzinfo
    return ZoneInfo(tz_name)


def now_local(tz) -> dt.datetime:
    return dt.datetime.now(tz)


def now_str_local(tz) -> str:
    return now_local(tz).strftime("%Y-%m-%d %H:%M:%S%z")


def now_str_utc() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def ts_compact_local(tz) -> str:
    return now_local(tz).strftime("%Y%m%d_%H%M")


def md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_stamp_to_dt(stamp: str, tz) -> dt.datetime:
    try:
        return dt.datetime.strptime(stamp, "%Y%m%d_%H%M%z")
    except Exception:
        pass
    d_naive = dt.datetime.strptime(stamp, "%Y%m%d_%H%M")
    return d_naive.replace(tzinfo=tz)


def rotate(path: Path, keep_days: int, tz) -> None:
    if keep_days <= 0:
        return
    limit = now_local(tz) - dt.timedelta(days=keep_days)
    for f in path.glob("**/*.mp3"):
        try:
            parts = f.stem.rsplit("_", 2)
            if len(parts) < 3:
                continue
            stamp = parts[-2] + "_" + parts[-1]
            d = parse_stamp_to_dt(stamp, tz)
            if d < limit:
                f.unlink(missing_ok=True)
        except Exception:
            pass


def write_index_row(index_csv: Path, row: list) -> None:
    new = not index_csv.exists()
    with index_csv.open("a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter=";")
        if new:
            w.writerow(["datetime_local", "datetime_utc", "zone", "filename", "filesize", "md5"])
        w.writerow(row)


def dedupe_by_md5(day_dir: Path, new_md5: str) -> bool:
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
    stamp = ts_compact_local(tz)
    fpath = day_dir / f"flash_{zone}_{stamp}.mp3"
    with fpath.open("wb") as fp:
        fp.write(content)
    return fpath


def fetch_once_conditional(root_dir: Path, keep_days: int, tz, cond_state: dict) -> int:
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
            print(f"{now_str_local(tz)} | {zone} | ERREUR requête: {e}")
            continue

        etag = r.headers.get("ETag")
        lm = r.headers.get("Last-Modified")
        if etag:
            st["etag"] = etag
        if lm:
            st["lm"] = lm

        if r.status_code == 304:
            continue

        if r.status_code != 200:
            print(f"{now_str_local(tz)} | {zone} | HTTP {r.status_code} (ignoré)")
            continue

        content = r.content
        if len(content) < 10_000:
            continue

        m = md5_bytes(content)

        day_dir = root_dir / now_local(tz).strftime("%Y-%m-%d") / zone
        ensure_dir(day_dir)
        if dedupe_by_md5(day_dir, m):
            continue

        fpath = save_audio(root_dir, zone, content, tz)
        write_index_row(
            index_csv,
            [
                now_str_local(tz),
                now_str_utc(),
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
    return fetch_once_conditional(root_dir, keep_days, tz, {})
