# flash-nlp

Pipeline Python de transcription audio et d'analyse NLP pour la détection automatique d'événements trafic à partir de flux audio publics.

---

## Vue d'ensemble

```
Flux audio (MP3)
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│  Acquisition │────▶│ Transcription │────▶│  Analyse NLP      │────▶│ Notification │
│  HTTP + CSV  │     │  Whisper ASR  │     │  Extraction regex │     │ Console/Hook │
└─────────────┘     └──────────────┘     └───────────────────┘     └──────────────┘
```

Le projet est structuré comme un package Python installable (`flash-nlp`) avec quatre couches indépendantes et une suite de tests complète.

---

## Fonctionnalités

- **Acquisition** : téléchargement conditionnel (ETag / If-Modified-Since), déduplification MD5, archivage horodaté, rotation automatique
- **Transcription** : conversion audio via ffmpeg → WAV 16kHz mono, transcription avec `faster-whisper` + VAD intégré
- **Analyse NLP** : détection de 8 types d'événements (accident, bouchon, travaux, intempéries…) par expressions régulières, extraction de routes, directions et délais
- **Notification** : dispatch console, notification macOS native, webhook HTTP, log JSONL
- **Tests** : 5 fichiers pytest couvrant chaque couche avec mocks

---

## Installation

**Prérequis :** Python 3.11+, ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
apt install ffmpeg
```

```bash
git clone <repo>
cd translate-audio-NLP-Ai

python -m venv .venv
source .venv/bin/activate

pip install -e .
# ou avec l'interface graphique (PySide6)
pip install -e ".[gui]"
```

---

## Structure

```
.
├── pyproject.toml
├── src/
│   └── flash_nlp/
│       ├── acquisition/
│       │   └── fetcher.py          # Téléchargement + déduplification + archivage
│       ├── transcription/
│       │   ├── whisper_service.py  # ASR via faster-whisper
│       │   └── audio_utils.py      # Conversion ffmpeg, RMS, WAV
│       ├── analysis/
│       │   ├── event_extractor.py  # Détection NLP + dataclass TrafficEvent
│       │   └── notifier.py         # Console, macOS, webhook, JSONL
│       └── io/
│           └── file_utils.py       # JSON, listing audio
├── tests/
│   ├── test_fetcher.py
│   ├── test_whisper_service.py
│   ├── test_audio_utils.py
│   ├── test_event_extractor.py
│   └── test_io.py
├── data/
│   └── flash_audio_archive/        # Archive MP3 + index.csv + state.json
└── models/
    ├── mms-tts-eng/                # Modèle TTS anglais (Meta MMS)
    └── mms-tts-ukr/                # Modèle TTS ukrainien (Meta MMS)
```

---

## Utilisation

### Acquisition

```python
from pathlib import Path
from flash_nlp.acquisition.fetcher import fetch_once_conditional, get_tz

tz = get_tz("Europe/Paris")
root = Path("data/flash_audio_archive")
cond_state = {}  # persister entre les appels pour activer le cache HTTP

added = fetch_once_conditional(root, keep_days=30, tz=tz, cond_state=cond_state)
print(f"{added} nouveaux fichiers sauvegardés")
```

### Transcription

```python
from flash_nlp.transcription.whisper_service import WhisperService
from flash_nlp.transcription.audio_utils import convert_to_wav_16k_mono

# Conversion MP3 → WAV
convert_to_wav_16k_mono("audio.mp3", "/tmp/audio.wav")

# Transcription
svc = WhisperService()
svc.load("small", device="cpu")  # ou "medium", "large-v3"

text, lang, prob = svc.transcribe_wav("/tmp/audio.wav", language="fr", beam_size=5)
print(f"[{lang} {prob:.0%}] {text}")

# Mode segmenté (avec timestamps)
result = svc.transcribe_wav_with_segments("/tmp/audio.wav", language="fr", beam_size=5)
for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}")
```

### Analyse NLP

```python
from flash_nlp.analysis.event_extractor import extract_events, severity_rank

text = "Un accident sur l'A6 sens Paris au km 34. Comptez 20 minutes de retard."

events = extract_events(text, zone="nord", source_file="flash.mp3", timestamp="20260123_1649")

for e in sorted(events, key=lambda x: severity_rank(x.severity), reverse=True):
    print(f"[{e.severity.upper()}] {e.type} | {e.routes} | {e.direction}")
    print(f"  → {e.location_hint}")
```

Sortie :
```
[HIGH] accident | ['A6'] | sens Paris
  → ...Un accident sur l'A6 sens Paris au km 34...
```

### Notification

```python
from pathlib import Path
from flash_nlp.analysis.notifier import dispatch

dispatch(
    event=events[0],
    alerts_dir=Path("outputs/alerts"),
    macos=True,                                 # notification macOS native
    webhook_url="https://hooks.slack.com/...",  # optionnel
)
```

---

## Événements détectés

| Type | Sévérité | Mots-clés |
|------|----------|-----------|
| `accident` | 🔴 high | accident, collision, accrochage, carambolage… |
| `fermeture` | 🔴 high | fermé, fermeture, déviation, voie fermée… |
| `bouchon` | 🟠 medium | bouchon, embouteillage, congestion… |
| `animal` | 🟠 medium | sanglier, animal errant, sur la chaussée… |
| `intemperies` | 🟠 medium | verglas, neige, brouillard, alerte orange… |
| `ralentissement` | 🟡 low | ralentissement, trafic dense, circulation difficile… |
| `travaux` | 🟡 low | travaux, chantier, rétrécissement… |
| `vehicule_panne` | 🟡 low | véhicule en panne, dépannage… |

---

## Format de sortie

### Index CSV (`data/flash_audio_archive/index.csv`)

```
datetime_local;datetime_utc;zone;filename;filesize;md5
2026-01-23 16:49:16+0100;2026-01-23 15:49:16;nord;2026-01-23/nord/flash_nord_20260123_164916.mp3;1237850;5ad694...
```

### Alertes JSONL (`outputs/alerts/alerts.jsonl`)

```json
{
  "type": "accident",
  "severity": "high",
  "routes": ["A6"],
  "direction": "sens Paris",
  "location_hint": "...accident sur l'A6 sens Paris au km 34...",
  "zone": "nord",
  "timestamp": "20260123_1649",
  "source_file": "2026-01-23/nord/flash.mp3",
  "delay_hint": "20 minutes",
  "alerted_at": "2026-01-23T15:49:16+00:00"
}
```

---

## Tests

```bash
pip install pytest pytest-mock
pytest
```

| Fichier | Ce qui est testé |
|---------|-----------------|
| `test_fetcher.py` | Déduplification MD5, rotation, CSV, mock HTTP (304, 200, erreur réseau) |
| `test_whisper_service.py` | Cache modèle, compute_type auto, VAD, segments, mock WhisperModel |
| `test_audio_utils.py` | RMS, save_wav, clip int16, ffmpeg presence, conversion |
| `test_event_extractor.py` | Détection par type, faux positifs, routes, direction, sévérité |
| `test_io.py` | JSON roundtrip, listing audio, ensure_dir |

---

## Dépendances

| Package | Rôle |
|---------|------|
| `faster-whisper` | ASR (CTranslate2, quantification int8/float16) |
| `numpy` | Traitement du signal audio |
| `scipy` | Lecture / écriture WAV |
| `sounddevice` | Listing des périphériques audio |
| `requests` | Téléchargement HTTP conditionnel |
| `ffmpeg` *(externe)* | Conversion audio multi-format |

---

## Roadmap

- [ ] Script `main.py` — pipeline bout-en-bout (acquisition → transcription → NLP → alerte)
- [ ] Traduction automatique (MarianMT ou LLM)
- [ ] Synthèse vocale multilingue via modèles MMS embarqués
- [ ] Interface graphique (PySide6)
- [ ] Synchronisation `requirements.txt` ↔ `pyproject.toml`

---

## Licence

[LICENSE](LICENSE)
