"""Fixtures globales — chargées automatiquement par pytest pour tous les tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Racine du repo
ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Chemin absolu de la racine du projet."""
    return ROOT


@pytest.fixture(scope="session")
def sample_audio_path(repo_root: Path) -> Path:
    """Chemin vers un fichier audio de test (flash trafic réel, FR).
    Cherche dans plusieurs emplacements possibles selon l'organisation du repo."""
    candidates = [
        repo_root / "flash_audio_archive" / "2026-04-14" / "nord" / "flash_nord_20260414_1818.mp3",
        repo_root / "data" / "golden" / "flash_audio_archive" / "2026-04-14" / "nord" / "flash_nord_20260414_1818.mp3",
        repo_root / "frontend" / "public" / "demo.mp3",
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip(f"Aucun audio de test trouvé (cherché : {[str(c) for c in candidates]})")


def _add_service_to_path(service_name: str) -> None:
    """Rend un service backend importable en tant que package (pour les tests unitaires)."""
    src = ROOT / "backend" / "services" / service_name / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# Rend les modules internes des services importables dans les tests unitaires
for _svc in ("pipeline", "gateway", "llm", "stt", "tts", "watcher"):
    _add_service_to_path(_svc)
