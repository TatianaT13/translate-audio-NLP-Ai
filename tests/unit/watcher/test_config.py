"""Tests unitaires — configuration du watcher.

Note :
  - event_extractor est déjà largement couvert par tests/unit/legacy/
    test_event_extractor.py
  - On lit le source de main.py au lieu de l'importer, pour éviter la
    dépendance `faster_whisper` (500 Mo avec torch) dans les tests unit.
"""

from __future__ import annotations

from pathlib import Path


WATCHER_MAIN_SOURCE = (
    Path(__file__).parent.parent.parent.parent
    / "backend" / "services" / "watcher" / "src" / "watcher" / "main.py"
).read_text()


class TestWatcherConfig:
    """Config env vars du watcher."""

    def test_poll_interval_configurable_via_env(self):
        """POLL_INTERVAL_S doit être lu depuis l'env pour flexibilité prod."""
        assert 'POLL_INTERVAL_S' in WATCHER_MAIN_SOURCE
        assert 'os.getenv("POLL_INTERVAL_S"' in WATCHER_MAIN_SOURCE

    def test_whisper_model_configurable_via_env(self):
        assert 'WHISPER_MODEL' in WATCHER_MAIN_SOURCE
        assert 'os.getenv("WHISPER_MODEL"' in WATCHER_MAIN_SOURCE

    def test_langfuse_v4_client_initialized(self):
        """Le watcher doit init le SDK v4 (pas le raw /api/public/ingestion)."""
        assert "from langfuse import Langfuse" in WATCHER_MAIN_SOURCE, (
            "Watcher doit utiliser le SDK v4, pas juste du raw HTTP"
        )

    def test_no_legacy_ingestion_http_call(self):
        """Le watcher NE doit PLUS faire de POST HTTP vers /api/public/ingestion.

        Vérifie qu'aucun `client.post(...ingestion...)` n'existe. Les mentions
        en commentaires/docstrings sont autorisées (historique de migration).
        """
        # Signature du bug legacy : client.post + LANGFUSE_HOST + /ingestion
        # On check l'absence de la combinaison HTTP + endpoint ingestion
        assert 'client.post' not in WATCHER_MAIN_SOURCE or (
            'client.post' in WATCHER_MAIN_SOURCE
            and '/api/public/ingestion' not in WATCHER_MAIN_SOURCE.split('client.post')[1][:500]
        ), "Un client.post(...ingestion...) subsiste — migrer vers SDK v4"


class TestEventExtractorImport:
    """Vérifie que le module event_extractor est bien importable (pas de dep lourde)."""

    def test_extract_events_callable(self):
        from watcher.event_extractor import extract_events
        assert callable(extract_events)

    def test_traffic_event_class_exists(self):
        from watcher.event_extractor import TrafficEvent
        assert TrafficEvent.__name__ == "TrafficEvent"

    def test_severity_rank_ordering(self):
        """high > medium > low."""
        from watcher.event_extractor import severity_rank
        assert severity_rank("high") > severity_rank("medium") > severity_rank("low")
