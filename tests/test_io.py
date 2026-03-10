import json
from pathlib import Path

import pytest

from flash_nlp.io import ensure_dir, load_json, save_json, list_audio_files, AUDIO_EXTS


# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------

def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    ensure_dir(target)
    assert target.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    ensure_dir(target)  # ne doit pas lever d'exception
    assert target.is_dir()


# ---------------------------------------------------------------------------
# load_json / save_json
# ---------------------------------------------------------------------------

def test_save_load_json_roundtrip(tmp_path):
    data = {"key": "valeur", "liste": [1, 2, 3], "unicode": "été"}
    p = tmp_path / "data.json"
    save_json(p, data)
    result = load_json(p)
    assert result == data


def test_load_json_missing_file(tmp_path):
    p = tmp_path / "inexistant.json"
    assert load_json(p) == {}


def test_save_json_unicode_preserved(tmp_path):
    data = {"message": "Île-de-France: état d'urgence"}
    p = tmp_path / "fr.json"
    save_json(p, data)
    raw = p.read_text(encoding="utf-8")
    assert "Île-de-France" in raw  # ensure_ascii=False


# ---------------------------------------------------------------------------
# list_audio_files
# ---------------------------------------------------------------------------

def test_list_audio_files_extensions(tmp_path):
    for ext in AUDIO_EXTS:
        (tmp_path / f"file{ext}").touch()

    found = {f.suffix.lower() for f in list_audio_files(tmp_path)}
    assert found == AUDIO_EXTS


def test_list_audio_files_ignores_others(tmp_path):
    for name in ["doc.txt", "data.json", "script.py", "image.png"]:
        (tmp_path / name).touch()

    found = list(list_audio_files(tmp_path))
    assert found == []


def test_list_audio_files_recursive(tmp_path):
    sub = tmp_path / "2026-01-22" / "nord"
    sub.mkdir(parents=True)
    (sub / "flash_nord_20260122_1000.mp3").touch()
    (tmp_path / "autre.wav").touch()

    found = {f.name for f in list_audio_files(tmp_path)}
    assert "flash_nord_20260122_1000.mp3" in found
    assert "autre.wav" in found


def test_list_audio_files_case_insensitive(tmp_path):
    (tmp_path / "AUDIO.MP3").touch()
    (tmp_path / "track.WAV").touch()

    found = list(list_audio_files(tmp_path))
    assert len(found) == 2
