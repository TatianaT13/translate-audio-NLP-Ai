import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Patterns de détection
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, re.Pattern] = {
    "accident": re.compile(
        r'\b(accident|collision|accrochage|carambolage|heurt(?:é|er)?|percuté|renversé)\b',
        re.IGNORECASE,
    ),
    "bouchon": re.compile(
        r'\b(bouchon|embouteillage|file\s+d[\'e]\s*attente|congestion|saturé|saturation)\b',
        re.IGNORECASE,
    ),
    "ralentissement": re.compile(
        r'\b(ralentissement|ralenti|circulation\s+ralentie|trafic\s+dense|trafic\s+charg[eé]|'
        r'circulation\s+difficile|trafic\s+difficile|fort\s+trafic|fluide\s+mais)\b',
        re.IGNORECASE,
    ),
    "vehicule_panne": re.compile(
        r'\b(v[eé]hicule?\s+en\s+panne|voiture\s+en\s+panne|poids.lourd\s+en\s+panne|'
        r'camion\s+en\s+panne|d[eé]pannage|obstacle\s+sur\s+la\s+chauss[eé]e)\b',
        re.IGNORECASE,
    ),
    "animal": re.compile(
        r'\b(animal|animaux|sanglier|vache|cheval|troupeau|biche|chevreuil|cerf|'
        r'errant|sur\s+la\s+chaussée|traverse)\b',
        re.IGNORECASE,
    ),
    "travaux": re.compile(
        r'\b(travaux|chantier|zone\s+de\s+travaux|r[eé]tr[eé]cissement|basculement)\b',
        re.IGNORECASE,
    ),
    "fermeture": re.compile(
        r'\b(ferm[eé]|fermeture|interdit|d[eé]viation|voie\s+ferm[eé]e|bretelle\s+ferm[eé]e|'
        r'tunnel\s+ferm[eé])\b',
        re.IGNORECASE,
    ),
    "intemperies": re.compile(
        r'\b(neige|verglas|brouillard|pluie\s+vergla[cç]ante|givr[eé]|black.ice|'
        r'alerte\s+orange|alerte\s+rouge|conditions\s+hivernales|chaine|pneus?\s+hiver)\b',
        re.IGNORECASE,
    ),
}

_SEVERITY: dict[str, str] = {
    "accident": "high",
    "fermeture": "high",
    "bouchon": "medium",
    "animal": "medium",
    "intemperies": "medium",
    "ralentissement": "low",
    "travaux": "low",
    "vehicule_panne": "low",
}

_SEVERITY_RANK: dict[str, int] = {"high": 2, "medium": 1, "low": 0}

_ROUTE_RE = re.compile(
    r'\b(A\d+|N\d+|D\d+|RN\d+|RD\d+|p[eé]riph[eé]rique|rocade|boulevard\s+p[eé]riph[eé]rique)\b',
    re.IGNORECASE,
)

_DIRECTION_RE = re.compile(
    r'\b(sens\s+\w+(?:\s+\w+)?|direction\s+\w+(?:\s+\w+)?|vers\s+\w+(?:\s+\w+)?|'
    r'entre\s+\w+\s+et\s+\w+)\b',
    re.IGNORECASE,
)

_DELAY_RE = re.compile(
    r'(\d+)\s*(?:minute|min|km\s+de\s+bouchon|kilomètre)',
    re.IGNORECASE,
)

_CONTEXT_RADIUS = 80  # caractères autour du match pour le location_hint


# ---------------------------------------------------------------------------
# Modèle de données
# ---------------------------------------------------------------------------

@dataclass
class TrafficEvent:
    type: str
    severity: str
    routes: List[str]
    direction: str
    location_hint: str
    zone: str
    timestamp: str
    source_file: str
    delay_hint: str = ""

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "routes": self.routes,
            "direction": self.direction,
            "location_hint": self.location_hint,
            "zone": self.zone,
            "timestamp": self.timestamp,
            "source_file": self.source_file,
            "delay_hint": self.delay_hint,
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_context(text: str, match_start: int, match_end: int) -> str:
    start = max(0, match_start - _CONTEXT_RADIUS)
    end = min(len(text), match_end + _CONTEXT_RADIUS)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def _extract_routes(text: str) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for m in _ROUTE_RE.finditer(text):
        key = m.group(0).upper().replace(" ", "")
        if key not in seen:
            seen.add(key)
            result.append(m.group(0))
    return result


def _extract_direction(text: str) -> str:
    m = _DIRECTION_RE.search(text)
    return m.group(0).strip() if m else ""


def _extract_delay(text: str) -> str:
    m = _DELAY_RE.search(text)
    return m.group(0).strip() if m else ""


def extract_events(
    text: str,
    zone: str,
    source_file: str,
    timestamp: str,
) -> List[TrafficEvent]:
    """
    Analyse le texte d'une transcription et retourne la liste des événements
    trafic détectés (un par type de perturbation trouvé).
    """
    events: List[TrafficEvent] = []
    routes = _extract_routes(text)
    direction = _extract_direction(text)
    delay = _extract_delay(text)

    for event_type, pattern in _PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue

        hint = _extract_context(text, match.start(), match.end())

        events.append(
            TrafficEvent(
                type=event_type,
                severity=_SEVERITY[event_type],
                routes=routes,
                direction=direction,
                location_hint=hint,
                zone=zone,
                timestamp=timestamp,
                source_file=source_file,
                delay_hint=delay,
            )
        )

    return events


def severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(severity, 0)
