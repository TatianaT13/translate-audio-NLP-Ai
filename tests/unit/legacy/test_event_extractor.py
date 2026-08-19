import pytest

from flash_nlp.analysis.event_extractor import (
    TrafficEvent,
    extract_events,
    severity_rank,
)

# Textes de test représentatifs des vraies transcriptions autoroutières
_TEXT_ACCIDENT = (
    "Autoroute Info, bonjour. Un accident sur l'A6 sens Paris au niveau du km 34, "
    "deux véhicules impliqués, circulation très ralentie. Comptez 20 minutes de retard."
)
_TEXT_BOUCHON = (
    "Bouchon de 8 km sur l'A43 direction Chambéry entre les sorties 20 et 23. "
    "Prévoyez 35 minutes supplémentaires."
)
_TEXT_RALENTISSEMENT = (
    "Circulation ralentie sur la N7 vers Lyon. Trafic chargé mais fluide. "
    "Comptez quelques minutes de retard seulement."
)
_TEXT_INTEMPERIES = (
    "Attention, verglas et neige sur l'A40 en Haute-Savoie. "
    "Chaînes ou pneus hiver obligatoires. Alerte orange en vigueur."
)
_TEXT_ANIMAL = (
    "Un sanglier errant sur la chaussée de l'A71 entre Vichy et Moulins. "
    "Soyez vigilants."
)
_TEXT_TRAVAUX = (
    "Travaux de nuit sur la rocade de Bordeaux, rétrécissement de voie "
    "jusqu'à vendredi matin."
)
_TEXT_FERMETURE = (
    "Bretelle fermée sur l'A1 sens province à hauteur de Senlis. "
    "Déviation en place par la N17."
)
_TEXT_NEUTRE = (
    "Autoroute Info. Le réseau autoroutier est fluide sur l'ensemble du territoire. "
    "Bonne route à tous."
)
_TEXT_MULTIPLE = _TEXT_ACCIDENT + " " + _TEXT_INTEMPERIES


# ---------------------------------------------------------------------------
# Détection par type
# ---------------------------------------------------------------------------

def test_detects_accident():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "accident" in types


def test_detects_bouchon():
    events = extract_events(_TEXT_BOUCHON, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "bouchon" in types


def test_detects_ralentissement():
    events = extract_events(_TEXT_RALENTISSEMENT, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "ralentissement" in types


def test_detects_intemperies():
    events = extract_events(_TEXT_INTEMPERIES, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "intemperies" in types


def test_detects_animal():
    events = extract_events(_TEXT_ANIMAL, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "animal" in types


def test_detects_travaux():
    events = extract_events(_TEXT_TRAVAUX, zone="ouest", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "travaux" in types


def test_detects_fermeture():
    events = extract_events(_TEXT_FERMETURE, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    types = [e.type for e in events]
    assert "fermeture" in types


# ---------------------------------------------------------------------------
# Extraction de routes
# ---------------------------------------------------------------------------

def test_extracts_routes_a6():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert events
    routes = events[0].routes
    assert any("A6" in r for r in routes)


def test_extracts_multiple_routes():
    events = extract_events(_TEXT_BOUCHON, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    assert events
    routes = events[0].routes
    assert any("A43" in r for r in routes)


def test_extracts_rocade():
    events = extract_events(_TEXT_TRAVAUX, zone="ouest", source_file="f.mp3", timestamp="20260122_1000")
    assert events
    routes = events[0].routes
    assert any("rocade" in r.lower() for r in routes)


# ---------------------------------------------------------------------------
# Extraction de direction
# ---------------------------------------------------------------------------

def test_extracts_direction_sens_paris():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert events
    assert "paris" in events[0].direction.lower()


def test_extracts_direction_vers():
    events = extract_events(_TEXT_RALENTISSEMENT, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    assert events
    assert events[0].direction != ""


# ---------------------------------------------------------------------------
# Absence de faux positifs
# ---------------------------------------------------------------------------

def test_no_false_positive_neutral_text():
    events = extract_events(_TEXT_NEUTRE, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert events == []


def test_empty_text_returns_no_events():
    events = extract_events("", zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert events == []


# ---------------------------------------------------------------------------
# Sévérité
# ---------------------------------------------------------------------------

def test_severity_accident_is_high():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    acc = next(e for e in events if e.type == "accident")
    assert acc.severity == "high"


def test_severity_ralentissement_is_low():
    events = extract_events(_TEXT_RALENTISSEMENT, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    r = next(e for e in events if e.type == "ralentissement")
    assert r.severity == "low"


def test_severity_bouchon_is_medium():
    events = extract_events(_TEXT_BOUCHON, zone="sud", source_file="f.mp3", timestamp="20260122_1000")
    b = next(e for e in events if e.type == "bouchon")
    assert b.severity == "medium"


def test_severity_rank_ordering():
    assert severity_rank("high") > severity_rank("medium") > severity_rank("low")


# ---------------------------------------------------------------------------
# Texte avec plusieurs événements
# ---------------------------------------------------------------------------

def test_multiple_events_detected():
    events = extract_events(_TEXT_MULTIPLE, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    types = {e.type for e in events}
    assert "accident" in types
    assert "intemperies" in types


# ---------------------------------------------------------------------------
# Métadonnées propagées
# ---------------------------------------------------------------------------

def test_zone_propagated():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert all(e.zone == "nord" for e in events)


def test_timestamp_propagated():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260123_1649")
    assert all(e.timestamp == "20260123_1649" for e in events)


def test_source_file_propagated():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="2026/nord/flash.mp3", timestamp="?")
    assert all(e.source_file == "2026/nord/flash.mp3" for e in events)


def test_location_hint_non_empty():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="?")
    acc = next(e for e in events if e.type == "accident")
    assert len(acc.location_hint) > 0


def test_as_dict_has_expected_keys():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert events
    d = events[0].as_dict()
    expected_keys = {"type", "severity", "routes", "direction", "location_hint", "zone", "timestamp", "source_file", "delay_hint"}
    assert expected_keys == set(d.keys())
