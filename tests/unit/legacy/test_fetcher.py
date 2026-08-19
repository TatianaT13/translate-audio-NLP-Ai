import csv
import datetime as dt
import hashlib
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from flash_nlp.acquisition.fetcher import (
    DEFAULT_TZ_NAME,
    dedupe_by_md5,
    fetch_once_conditional,
    get_tz,
    md5_bytes,
    parse_stamp_to_dt,
    rotate,
    write_index_row,
)

TZ_PARIS = ZoneInfo("Europe/Paris")


# ---------------------------------------------------------------------------
# md5_bytes
# ---------------------------------------------------------------------------

def test_md5_bytes_known_value():
    expected = hashlib.md5(b"hello").hexdigest()
    assert md5_bytes(b"hello") == expected


def test_md5_bytes_empty():
    assert md5_bytes(b"") == hashlib.md5(b"").hexdigest()


# ---------------------------------------------------------------------------
# parse_stamp_to_dt
# ---------------------------------------------------------------------------

def test_parse_stamp_with_offset():
    result = parse_stamp_to_dt("20260217_1744+0100", TZ_PARIS)
    assert result.year == 2026
    assert result.month == 2
    assert result.day == 17
    assert result.hour == 17
    assert result.minute == 44


def test_parse_stamp_without_offset():
    result = parse_stamp_to_dt("20260217_1744", TZ_PARIS)
    assert result.year == 2026
    assert result.tzinfo == TZ_PARIS


# ---------------------------------------------------------------------------
# get_tz
# ---------------------------------------------------------------------------

def test_get_tz_paris():
    tz = get_tz("Europe/Paris")
    assert str(tz) == "Europe/Paris"


def test_get_tz_local():
    tz = get_tz("local")
    assert tz is not None


def test_get_tz_invalid():
    with pytest.raises(Exception):
        get_tz("Pays/Imaginaire")


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

def _make_mp3(directory: Path, stamp: str) -> Path:
    """Crée un faux fichier MP3 avec un nom de stamp donné."""
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"flash_nord_{stamp}.mp3"
    p.write_bytes(b"\xff" * 100)
    return p


def test_rotate_removes_old_files(tmp_path):
    old_stamp = "20200101_1200"
    recent_stamp = dt.datetime.now(TZ_PARIS).strftime("%Y%m%d_%H%M")
    old_file = _make_mp3(tmp_path, old_stamp)
    recent_file = _make_mp3(tmp_path, recent_stamp)

    rotate(tmp_path, keep_days=30, tz=TZ_PARIS)

    assert not old_file.exists()
    assert recent_file.exists()


def test_rotate_keeps_recent_files(tmp_path):
    recent_stamp = dt.datetime.now(TZ_PARIS).strftime("%Y%m%d_%H%M")
    recent_file = _make_mp3(tmp_path, recent_stamp)

    rotate(tmp_path, keep_days=30, tz=TZ_PARIS)

    assert recent_file.exists()


def test_rotate_keep_days_zero_noop(tmp_path):
    old_stamp = "20200101_1200"
    old_file = _make_mp3(tmp_path, old_stamp)

    rotate(tmp_path, keep_days=0, tz=TZ_PARIS)

    assert old_file.exists()


# ---------------------------------------------------------------------------
# write_index_row
# ---------------------------------------------------------------------------

def test_write_index_row_creates_header(tmp_path):
    csv_path = tmp_path / "index.csv"
    write_index_row(csv_path, ["2026-01-22 10:00:00+0100", "2026-01-22 09:00:00", "nord", "file.mp3", 1234, "abc"])

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    assert rows[0] == ["datetime_local", "datetime_utc", "zone", "filename", "filesize", "md5"]
    assert rows[1][2] == "nord"


def test_write_index_row_appends(tmp_path):
    csv_path = tmp_path / "index.csv"
    row1 = ["2026-01-22 10:00:00+0100", "2026-01-22 09:00:00", "nord", "f1.mp3", 100, "aa"]
    row2 = ["2026-01-22 10:15:00+0100", "2026-01-22 09:15:00", "sud", "f2.mp3", 200, "bb"]

    write_index_row(csv_path, row1)
    write_index_row(csv_path, row2)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    # 1 header + 2 lignes de données
    assert len(rows) == 3
    assert rows[1][2] == "nord"
    assert rows[2][2] == "sud"


# ---------------------------------------------------------------------------
# dedupe_by_md5
# ---------------------------------------------------------------------------

def test_dedupe_by_md5_detects_duplicate(tmp_path):
    content = b"\xff" * 500
    existing = tmp_path / "flash_nord_20260122_1000.mp3"
    existing.write_bytes(content)

    assert dedupe_by_md5(tmp_path, md5_bytes(content)) is True


def test_dedupe_by_md5_new_content(tmp_path):
    existing = tmp_path / "flash_nord_20260122_1000.mp3"
    existing.write_bytes(b"\xff" * 500)

    assert dedupe_by_md5(tmp_path, md5_bytes(b"\xaa" * 500)) is False


def test_dedupe_by_md5_empty_dir(tmp_path):
    assert dedupe_by_md5(tmp_path, "anymd5") is False


# ---------------------------------------------------------------------------
# fetch_once_conditional (avec mock requests)
# ---------------------------------------------------------------------------

def _make_response(status: int, content: bytes = b"", headers: dict = None):
    r = MagicMock()
    r.status_code = status
    r.content = content
    r.headers = headers or {}
    return r


def test_fetch_once_conditional_304(tmp_path, mocker):
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        return_value=_make_response(304),
    )
    cond_state = {}
    added = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    assert added == 0


def test_fetch_once_conditional_200_saves(tmp_path, mocker):
    content = b"\xff" * 15_000  # > 10_000 bytes
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        return_value=_make_response(200, content, {"ETag": '"abc"'}),
    )
    cond_state = {}
    added = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    # 3 zones → 3 fichiers sauvegardés
    assert added == 3
    assert (tmp_path / "index.csv").exists()


def test_fetch_once_conditional_small_content(tmp_path, mocker):
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        return_value=_make_response(200, b"\xff" * 100),  # < 10_000 bytes
    )
    cond_state = {}
    added = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    assert added == 0


def test_fetch_once_conditional_network_error(tmp_path, mocker):
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        side_effect=ConnectionError("réseau indisponible"),
    )
    cond_state = {}
    # Ne doit pas lever d'exception
    added = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    assert added == 0


def test_fetch_once_conditional_updates_etag(tmp_path, mocker):
    content = b"\xff" * 15_000
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        return_value=_make_response(200, content, {"ETag": '"etag-xyz"', "Last-Modified": "Fri, 22 Jan 2026 10:00:00 GMT"}),
    )
    cond_state = {}
    fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)

    assert cond_state["nord"]["etag"] == '"etag-xyz"'
    assert cond_state["nord"]["lm"] == "Fri, 22 Jan 2026 10:00:00 GMT"


def test_fetch_once_conditional_deduplicates(tmp_path, mocker):
    content = b"\xff" * 15_000
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        return_value=_make_response(200, content),
    )
    cond_state = {}
    # Première passe → 3 fichiers
    added1 = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    # Deuxième passe avec le même contenu → 0 (doublon MD5)
    added2 = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    assert added1 == 3
    assert added2 == 0
