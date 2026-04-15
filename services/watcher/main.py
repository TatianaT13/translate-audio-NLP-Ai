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

from flash_nlp.transcription.whisper_service import WhisperService
from flash_nlp.analysis.event_extractor import extract_events, TrafficEvent

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

# ── État global (en RAM uniquement) ──────────────────────────────────────────
# Ring buffer : quand le 5e arrive, le 1er tombe automatiquement
_events: dict[str, deque] = {
    "nord":  deque(maxlen=MAX_PER_ZONE),
    "sud":   deque(maxlen=MAX_PER_ZONE),
    "ouest": deque(maxlen=MAX_PER_ZONE),
}

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
    }


async def _translate_batch(text: str, client: httpx.AsyncClient) -> dict[str, str]:
    """Traduit le texte FR dans toutes les langues cibles en parallèle."""
    async def _one(lang: str) -> tuple[str, str]:
        try:
            r = await client.post(
                f"{LLM_URL}/translate",
                json={"text": text, "target_lang": lang},
                timeout=30.0,
            )
            if r.is_success:
                return lang, r.json()["translation"]
        except Exception as e:
            print(f"[watcher] translate {lang} erreur: {e}", flush=True)
        return lang, ""

    results = await asyncio.gather(*[_one(l.strip()) for l in TRANSLATE_LANGS if l.strip()])
    return {lang: tr for lang, tr in results if tr}


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
        print(f"[watcher] {zone} | fetch erreur: {e}", flush=True)
        return

    # Mise à jour ETag/Last-Modified
    if r.headers.get("ETag"):
        st["etag"] = r.headers["ETag"]
    if r.headers.get("Last-Modified"):
        st["lm"] = r.headers["Last-Modified"]

    if r.status_code == 304:
        # Pas de changement
        return

    if r.status_code != 200 or len(r.content) < 10_000:
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
        return

    # Filtrer severity >= medium (pas les "low" / ralentissements banals)
    filtered = [ev for ev in events if ev.severity in ("high", "medium")]
    if not filtered:
        print(f"[watcher] {zone} | {len(events)} event(s) low severity ignorés", flush=True)
        return

    # Traduction auto du texte complet en parallèle vers toutes les langues cibles
    translations = await _translate_batch(text, client)
    if translations:
        print(f"[watcher] {zone} | traduit en {list(translations.keys())}", flush=True)

    for ev in filtered:
        _events[zone].append(_event_to_dict(ev, translations))
        print(
            f"[watcher] {zone} | {ev.severity.upper():6s} {ev.type:20s} "
            f"{', '.join(ev.routes) or '—'}  {ev.direction}",
            flush=True,
        )

    # Broadcast SSE
    _broadcast(zone, list(_events[zone]))


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
def health():
    return {
        "status": "ok",
        "service": "watcher",
        "whisper_model": WHISPER_MODEL,
        "poll_interval_s": POLL_INTERVAL,
        "events_per_zone": {z: len(d) for z, d in _events.items()},
    }


@app.get("/events")
def get_events():
    """Snapshot des événements actuels (max MAX_PER_ZONE par zone)."""
    return JSONResponse({
        zone: list(buf) for zone, buf in _events.items()
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
