"""
Watcher Service — Port 8005

Polling adaptatif sur les 3 flux autorouteinfo.fr (nord / sud / ouest).
- Fetch toutes les ~15s par zone (adaptatif via ETag/Last-Modified)
- STT Whisper en mémoire (pas de fichier conservé)
- Extraction d'événements trafic (event_extractor)
- Ring buffer deque(maxlen=4) par zone → les vieux tombent automatiquement
- SSE /stream  → dashboard admin en temps réel
- GET /events  → snapshot actuel
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from watcher.whisper_service import WhisperService
from watcher.event_extractor import extract_events, TrafficEvent

# ── Config ───────────────────────────────────────────────────────────────────
URLS = {
    "nord":  "https://audio.autorouteinfo.fr/flash_nord.mp3",
    "sud":   "https://audio.autorouteinfo.fr/flash_sud.mp3",
    "ouest": "https://audio.autorouteinfo.fr/flash_ouest.mp3",
}

WHISPER_MODEL  = os.getenv("WHISPER_MODEL", "small")
POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL_S", "15"))   # secondes entre chaque poll
MAX_PER_ZONE   = int(os.getenv("MAX_EVENTS_PER_ZONE", "10"))
LLM_URL        = os.getenv("LLM_URL", "http://llm:8002")
TRANSLATE_LANGS = os.getenv("TRANSLATE_LANGS", "en,uk,es").split(",")

# ── État global persisté sur disque (survit aux restarts container) ─────────
STATE_FILE = Path(os.getenv("STATE_FILE", "/app/state/events.json"))

def _load_state() -> dict[str, deque]:
    state = {
        "nord":  deque(maxlen=MAX_PER_ZONE),
        "sud":   deque(maxlen=MAX_PER_ZONE),
        "ouest": deque(maxlen=MAX_PER_ZONE),
    }
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            kept, dropped = 0, 0
            for zone in state:
                for ev in data.get(zone, [])[-MAX_PER_ZONE:]:
                    if _is_fresh(ev):
                        state[zone].append(ev)
                        kept += 1
                    else:
                        dropped += 1
            print(f"[watcher] état restauré : {kept} gardés, {dropped} périmés (>12h)", flush=True)
        except Exception as e:
            print(f"[watcher] état corrompu, démarre vide : {e}", flush=True)
    return state

def _save_state() -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({z: list(d) for z, d in _events.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[watcher] save state erreur : {e}", flush=True)

_events: dict[str, deque] = _load_state()

# Clients SSE connectés
_sse_clients: list[asyncio.Queue] = []

# ETag / Last-Modified par zone pour requêtes conditionnelles
_cond_state: dict[str, dict] = {
    "nord":  {"etag": None, "lm": None, "hash": None},
    "sud":   {"etag": None, "lm": None, "hash": None},
    "ouest": {"etag": None, "lm": None, "hash": None},
}

# Whisper chargé une seule fois
_whisper: WhisperService | None = None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Watcher — Trafic Live", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator(excluded_handlers=["/health", "/metrics"]).instrument(app).expose(app)

# ── Métriques business custom (au-delà du HTTP) ───────────────────────────────
from prometheus_client import Counter

WATCHER_POLLS = Counter(
    "watcher_polls_total",
    "Nombre de cycles de polling watcher",
    labelnames=["zone", "status"],   # status: error, no_change, fresh, no_events, ok
)
WATCHER_EVENTS = Counter(
    "watcher_events_extracted_total",
    "Nombre d'événements trafic extraits par sévérité et type",
    labelnames=["zone", "severity", "type"],
)
WATCHER_COST = Counter(
    "watcher_translation_cost_usd_total",
    "Coût LLM cumulé des traductions du watcher (USD)",
    labelnames=["lang"],
)
WATCHER_TOKENS = Counter(
    "watcher_translation_tokens_total",
    "Tokens LLM cumulés des traductions du watcher",
    labelnames=["lang"],
)

# ── Langfuse tracing (best-effort, ne bloque jamais) ─────────────────────────
LANGFUSE_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST   = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_whisper() -> WhisperService:
    global _whisper
    if _whisper is None:
        _whisper = WhisperService()
        _whisper.load(WHISPER_MODEL, device="cpu")
        print(f"[watcher] Whisper {WHISPER_MODEL} chargé", flush=True)
    return _whisper


def _mp3_to_text(mp3_bytes: bytes) -> tuple[str, float]:
    """Transcrit les bytes MP3 en texte FR via Whisper. Pas de fichier conservé."""
    # Écrire en tmp, convertir en WAV, transcrire, supprimer immédiatement
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
        f_mp3.write(mp3_bytes)
        mp3_path = f_mp3.name

    wav_path = mp3_path.replace(".mp3", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        svc = _get_whisper()
        text, lang, lang_prob = svc.transcribe_wav(wav_path, language="fr", beam_size=5)
        return text, lang_prob
    finally:
        Path(mp3_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)


EVENT_TTL_SECONDS = int(os.getenv("EVENT_TTL_SECONDS", "43200"))  # 12h

def _event_to_dict(ev: TrafficEvent, translations: dict[str, str] | None = None) -> dict:
    return {
        "type":          ev.type,
        "severity":      ev.severity,
        "routes":        ev.routes,
        "direction":     ev.direction,
        "location_hint": ev.location_hint,
        "zone":          ev.zone,
        "timestamp":     ev.timestamp,
        "delay_hint":    ev.delay_hint,
        "translations":  translations or {},
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }


def _is_fresh(ev: dict) -> bool:
    """Vrai si l'événement a moins de EVENT_TTL_SECONDS secondes."""
    try:
        created = datetime.fromisoformat(ev["created_at"])
        age = (datetime.now(timezone.utc) - created).total_seconds()
        return age < EVENT_TTL_SECONDS
    except (KeyError, ValueError):
        return False  # événements sans created_at (anciens) → drop


async def _translate_batch(text: str, client: httpx.AsyncClient) -> tuple[dict[str, str], dict[str, dict]]:
    """Traduit le texte FR dans toutes les langues cibles en parallèle.
    Retourne (translations, llm_meta) où llm_meta[lang] = {tokens, cost, model}."""
    async def _one(lang: str) -> tuple[str, str, dict]:
        try:
            r = await client.post(
                f"{LLM_URL}/translate",
                json={"text": text, "target_lang": lang},
                timeout=30.0,
            )
            if r.is_success:
                data = r.json()
                # Reporter coût + tokens en métriques Prometheus
                cost   = float(data.get("cost_usd", 0) or 0)
                tokens = int(data.get("total_tokens", 0) or 0)
                if cost > 0:
                    WATCHER_COST.labels(lang=lang).inc(cost)
                if tokens > 0:
                    WATCHER_TOKENS.labels(lang=lang).inc(tokens)
                return lang, data["translation"], {
                    "model":             data.get("model", ""),
                    "prompt_tokens":     int(data.get("prompt_tokens", 0)),
                    "completion_tokens": int(data.get("completion_tokens", 0)),
                    "total_tokens":      tokens,
                    "cost_usd":          cost,
                }
        except Exception as e:
            print(f"[watcher] translate {lang} erreur: {e}", flush=True)
        return lang, "", {}

    results = await asyncio.gather(*[_one(l.strip()) for l in TRANSLATE_LANGS if l.strip()])
    translations = {lang: tr for lang, tr, _ in results if tr}
    llm_meta     = {lang: meta for lang, tr, meta in results if tr}
    return translations, llm_meta


def _broadcast(zone: str, events: list[dict]) -> None:
    """Envoie une mise à jour SSE à tous les clients connectés."""
    payload = json.dumps({"zone": zone, "events": events}, ensure_ascii=False)
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _sse_clients.remove(q)


# ── Tracing Langfuse end-to-end d'un cycle watcher ───────────────────────────

async def _langfuse_trace_flash(
    client: httpx.AsyncClient,
    zone: str,
    text: str,
    lang_prob: float,
    events: list,
    translations: dict,
    llm_meta: dict,
) -> None:
    """Envoie 1 trace Langfuse par flash polled avec spans STT + extraction + N generations."""
    import uuid as _uuid
    from datetime import datetime as _dt, timezone as _tz

    tid = str(_uuid.uuid4())
    now_iso = _dt.now(_tz.utc).isoformat()

    # Compteurs par sévérité pour les scores
    sev_counts = {"high": 0, "medium": 0, "low": 0}
    for ev in events:
        sev_counts[ev.severity] = sev_counts.get(ev.severity, 0) + 1
    total_cost   = sum(m.get("cost_usd", 0)   for m in llm_meta.values())
    total_tokens = sum(m.get("total_tokens", 0) for m in llm_meta.values())

    batch = [
        # Trace racine
        {
            "id": str(_uuid.uuid4()),
            "type": "trace-create",
            "timestamp": now_iso,
            "body": {
                "id": tid,
                "name": "watcher_flash",
                "input":  {"zone": zone},
                "output": {"events_count": len(events), "translations": list(translations.keys())},
                "metadata": {
                    "zone":            zone,
                    "events_total":    len(events),
                    "events_high":     sev_counts["high"],
                    "events_medium":   sev_counts["medium"],
                    "events_low":      sev_counts["low"],
                    "cost_usd_total":  round(total_cost, 6),
                    "tokens_total":    total_tokens,
                },
            },
        },
        # Span STT
        {
            "id": str(_uuid.uuid4()),
            "type": "span-create",
            "timestamp": now_iso,
            "body": {
                "id":      str(_uuid.uuid4()),
                "traceId": tid,
                "name":    "stt",
                "input":   {"source": "autorouteinfo.fr/" + zone},
                "output":  {"text": text[:1000], "language_prob": lang_prob},
            },
        },
        # Span event extraction
        {
            "id": str(_uuid.uuid4()),
            "type": "span-create",
            "timestamp": now_iso,
            "body": {
                "id":      str(_uuid.uuid4()),
                "traceId": tid,
                "name":    "event_extraction",
                "input":   {"text": text[:500]},
                "output":  {
                    "events": [
                        {"type": ev.type, "severity": ev.severity,
                         "routes": ev.routes, "direction": ev.direction}
                        for ev in events
                    ],
                },
            },
        },
        # Generations pour chaque traduction
        *[
            {
                "id": str(_uuid.uuid4()),
                "type": "generation-create",
                "timestamp": now_iso,
                "body": {
                    "id":      str(_uuid.uuid4()),
                    "traceId": tid,
                    "name":    f"translate_to_{lang}",
                    "model":   meta.get("model", os.getenv("LLM_MODEL", "")),
                    "input":   text[:1500],
                    "output":  translations.get(lang, "")[:1500],
                    "usage": {
                        "input":     meta.get("prompt_tokens", 0),
                        "output":    meta.get("completion_tokens", 0),
                        "total":     meta.get("total_tokens", 0),
                        "unit":      "TOKENS",
                        "totalCost": meta.get("cost_usd", 0),
                    },
                },
            }
            for lang, meta in llm_meta.items()
        ],
        # Scores agrégés
        *[
            {
                "id": str(_uuid.uuid4()),
                "type": "score-create",
                "timestamp": now_iso,
                "body": {
                    "id":       str(_uuid.uuid4()),
                    "traceId":  tid,
                    "name":     name,
                    "value":    float(value),
                    "dataType": "NUMERIC",
                    "comment":  f"watcher | zone={zone}",
                },
            }
            for name, value in [
                ("events_total",   len(events)),
                ("events_high",    sev_counts["high"]),
                ("events_medium",  sev_counts["medium"]),
                ("language_prob",  lang_prob),
                ("cost_usd",       total_cost),
                ("total_tokens",   total_tokens),
            ]
        ],
    ]

    await client.post(
        f"{LANGFUSE_HOST}/api/public/ingestion",
        auth=(LANGFUSE_PUBLIC, LANGFUSE_SECRET),
        json={"batch": batch},
        timeout=8.0,
    )


# ── Boucle de polling par zone ────────────────────────────────────────────────

async def _poll_zone(zone: str, client: httpx.AsyncClient) -> None:
    """Fetch une zone, transcrit si nouveau contenu, extrait les événements."""
    st = _cond_state[zone]
    headers = {}
    if st["etag"]:
        headers["If-None-Match"] = st["etag"]
    if st["lm"]:
        headers["If-Modified-Since"] = st["lm"]

    try:
        r = await client.get(URLS[zone], headers=headers, timeout=20.0)
    except Exception as e:
        # autorouteinfo.fr serveur instable : timeouts/connect errors fréquents → silencieux sauf si nouveau type
        msg = repr(e) if not str(e) else str(e)
        print(f"[watcher] {zone} | fetch erreur ({type(e).__name__}): {msg}", flush=True)
        WATCHER_POLLS.labels(zone=zone, status="error").inc()
        return

    # Mise à jour ETag/Last-Modified
    if r.headers.get("ETag"):
        st["etag"] = r.headers["ETag"]
    if r.headers.get("Last-Modified"):
        st["lm"] = r.headers["Last-Modified"]

    if r.status_code == 304:
        WATCHER_POLLS.labels(zone=zone, status="no_change").inc()
        return

    if r.status_code != 200 or len(r.content) < 10_000:
        WATCHER_POLLS.labels(zone=zone, status="error").inc()
        return

    # Hash du contenu pour éviter de re-transcrire un MP3 identique
    # (fallback si le serveur ne respecte pas ETag/Last-Modified)
    import hashlib
    content_hash = hashlib.md5(r.content).hexdigest()
    if content_hash == st.get("hash"):
        return
    st["hash"] = content_hash

    # Transcription en thread pour ne pas bloquer l'event loop
    loop = asyncio.get_event_loop()
    try:
        text, lang_prob = await loop.run_in_executor(None, _mp3_to_text, r.content)
    except Exception as e:
        print(f"[watcher] {zone} | STT erreur: {e}", flush=True)
        return

    if not text.strip():
        return

    ts = datetime.now(ZoneInfo(os.getenv("TZ", "Europe/Paris"))).strftime("%H:%M")
    events = extract_events(text, zone=zone, source_file="live", timestamp=ts)

    if not events:
        print(f"[watcher] {zone} | transcrit, aucun événement détecté", flush=True)
        WATCHER_POLLS.labels(zone=zone, status="no_events").inc()
        return

    # Comptage Prometheus par sévérité/type
    for ev in events:
        WATCHER_EVENTS.labels(zone=zone, severity=ev.severity, type=ev.type).inc()

    # Fusion : les events sur la même portion (mêmes routes + même direction)
    # sont regroupés en UNE carte affichée, avec :
    #  - tous les types listés (ex: ["Ralentissement", "Bouchon"])
    #  - la sévérité MAX
    # Garde tous les niveaux (low/medium/high) — le filtre se fait côté frontend.
    _SEV_ORDER = {"high": 3, "medium": 2, "low": 1}
    grouped: dict[tuple, dict] = {}
    for ev in events:
        key = (tuple(ev.routes), ev.direction)
        d = _event_to_dict(ev, None)
        if key not in grouped:
            d["types"] = [ev.type]
            grouped[key] = d
        else:
            existing = grouped[key]
            if ev.type not in existing["types"]:
                existing["types"].append(ev.type)
            # Garder la sévérité max
            if _SEV_ORDER.get(ev.severity, 0) > _SEV_ORDER.get(existing["severity"], 0):
                existing["severity"] = ev.severity
                existing["type"] = ev.type   # type principal = le plus sévère

    # Traduction auto du texte complet en parallèle vers toutes les langues cibles
    translations, llm_meta = await _translate_batch(text, client)
    if translations:
        print(f"[watcher] {zone} | traduit en {list(translations.keys())}", flush=True)
        # Injecter les traductions dans chaque event groupé
        for d in grouped.values():
            d["translations"] = translations

    for d in grouped.values():
        _events[zone].append(d)
        print(
            f"[watcher] {zone} | {d['severity'].upper():6s} {'+'.join(d['types']):30s} "
            f"{', '.join(d['routes']) or '—'}  {d['direction']}",
            flush=True,
        )

    # Persister sur disque puis broadcast
    _save_state()
    _broadcast(zone, list(_events[zone]))
    WATCHER_POLLS.labels(zone=zone, status="ok").inc()

    # ── Trace Langfuse end-to-end pour ce cycle ────────────────────────────
    if LANGFUSE_PUBLIC and LANGFUSE_SECRET:
        try:
            await _langfuse_trace_flash(client, zone, text, lang_prob, events, translations, llm_meta)
        except Exception as e:
            print(f"[watcher] {zone} | langfuse trace erreur: {e}", flush=True)


async def _watcher_loop() -> None:
    """Boucle principale : poll les 3 zones en parallèle toutes les POLL_INTERVAL s."""
    print(f"[watcher] démarrage — interval={POLL_INTERVAL}s model={WHISPER_MODEL}", flush=True)

    # Précharger Whisper au démarrage (évite la latence au 1er flash)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_whisper)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        while True:
            t0 = time.perf_counter()
            await asyncio.gather(
                _poll_zone("nord",  client),
                _poll_zone("sud",   client),
                _poll_zone("ouest", client),
            )
            elapsed = time.perf_counter() - t0
            wait = max(0.0, POLL_INTERVAL - elapsed)
            await asyncio.sleep(wait)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "watcher",
        "whisper_model": WHISPER_MODEL,
        "poll_interval_s": POLL_INTERVAL,
        "events_per_zone": {z: len(d) for z, d in _events.items()},
    }


@app.get("/events")
def get_events():
    """Snapshot des événements actuels (max MAX_PER_ZONE par zone, filtrés par TTL)."""
    return JSONResponse({
        zone: [ev for ev in buf if _is_fresh(ev)] for zone, buf in _events.items()
    })


@app.get("/stream")
async def sse_stream():
    """
    Server-Sent Events — le dashboard s'abonne et reçoit les mises à jour en temps réel.
    Format : data: {"zone": "nord", "events": [...]}
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=20)
    _sse_clients.append(q)

    async def generator() -> AsyncIterator[str]:
        # Envoyer l'état actuel immédiatement à la connexion
        snapshot = {zone: list(buf) for zone, buf in _events.items()}
        yield f"data: {json.dumps({'snapshot': snapshot}, ensure_ascii=False)}\n\n"

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive ping
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(_watcher_loop())
