"""Skip automatique des tests legacy si les modules pré-microservices ne sont plus installés.

Ces tests couvrent l'ancien package `flash_nlp` (avant la migration en microservices).
Ils restent utiles quand on installe les deps historiques, mais ne doivent pas
bloquer une CI normale."""

import importlib

import pytest


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


collect_ignore_glob: list[str] = []

if not _module_available("faster_whisper"):
    collect_ignore_glob.append("test_whisper_service.py")
    collect_ignore_glob.append("test_audio_utils.py")

if not _module_available("flash_nlp"):
    collect_ignore_glob.extend([
        "test_event_extractor.py",
        "test_fetcher.py",
        "test_io.py",
    ])
