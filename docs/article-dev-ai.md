# Flash NLP — Pipeline de transcription et d'analyse NLP de flux audio publics en Python

> Article technique — publié sur [dev-ai.fr](https://dev-ai.fr)

---

## Introduction

Ce projet implémente un pipeline complet de traitement automatique de la parole (ASR) et d'analyse NLP appliqué à des flux audio publics diffusés en continu. L'objectif : capturer périodiquement des fichiers audio, les transcrire automatiquement avec un modèle de reconnaissance vocale, extraire des événements structurés du texte transcrit, et déclencher des alertes multi-canaux en fonction de la sévérité détectée.

Le stack est entièrement open-source, tourne en local (CPU ou GPU), ne nécessite aucune clé d'API, et est packagé comme un module Python installable.

---

## Architecture globale

```
┌──────────────────────────────────────────────────────────────────┐
│                        flash-nlp pipeline                        │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │  Acquisition │───▶│ Transcription│───▶│  Analyse NLP      │   │
│  │  (fetcher)  │    │  (Whisper)   │    │ (event_extractor) │   │
│  └─────────────┘    └──────────────┘    └────────┬──────────┘   │
│                                                   │              │
│                                          ┌────────▼──────────┐  │
│                                          │  Notification     │  │
│                                          │  (notifier)       │  │
│                                          └───────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Le projet est structuré en quatre couches indépendantes et testables :

| Couche | Module | Rôle |
|--------|--------|------|
| Acquisition | `flash_nlp.acquisition.fetcher` | Téléchargement conditionnel de MP3, déduplification, archivage |
| Transcription | `flash_nlp.transcription` | Conversion audio + ASR Whisper |
| Analyse | `flash_nlp.analysis.event_extractor` | Extraction d'événements par NLP regex |
| Notification | `flash_nlp.analysis.notifier` | Dispatch console / macOS / webhook / JSONL |

---

## Packaging et dépendances

Le projet est un package Python installable via `pyproject.toml` (PEP 517/518) :

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "flash-nlp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "faster-whisper>=1.0.3",
    "numpy>=1.26",
    "scipy>=1.11",
    "sounddevice>=0.5",
    "requests>=2.31",
]

[project.optional-dependencies]
gui = ["PySide6>=6.7"]

[tool.setuptools.packages.find]
where = ["src"]
```

Installation en mode développement :

```bash
pip install -e ".[gui]"
```

Python 3.11+ est requis pour `zoneinfo` (stdlib) et les annotations de type modernes.

---

## Couche 1 — Acquisition de flux audio

### Philosophie : téléchargement conditionnel HTTP

Le module `fetcher.py` implémente une stratégie de polling intelligent qui évite de re-télécharger des contenus inchangés grâce aux headers HTTP conditionnels.

```python
# Mécanisme de cache HTTP conditionnel
headers = {}
if st.get("etag"):
    headers["If-None-Match"] = st["etag"]
if st.get("lm"):
    headers["If-Modified-Since"] = st["lm"]

r = requests.get(url, headers=headers, timeout=30)

if r.status_code == 304:
    continue  # contenu inchangé, on skip
```

**`ETag`** : identifiant opaque côté serveur représentant une version du contenu.
**`If-Modified-Since`** : le serveur répond `304 Not Modified` si le contenu n'a pas changé depuis cette date.

Ce pattern réduit drastiquement la bande passante et la charge réseau dans un contexte de polling fréquent.

### Déduplification par hash MD5

Même si le serveur ne supporte pas les headers conditionnels, une seconde ligne de défense compare le hash MD5 du contenu téléchargé avec le dernier fichier enregistré :

```python
def dedupe_by_md5(day_dir: Path, new_md5: str) -> bool:
    existing = sorted(day_dir.glob("flash_*.mp3"))
    latest = existing[-1] if existing else None
    if latest is not None and latest.exists():
        with latest.open("rb") as lf:
            if md5_bytes(lf.read()) == new_md5:
                return True
    return False
```

La fonction compare uniquement avec le **dernier fichier** (tri alphabétique du timestamp dans le nom), ce qui est suffisant car les doublons sont consécutifs par nature.

### Filtrage des contenus invalides

Un seuil minimum de taille est appliqué pour rejeter les réponses tronquées ou les pages d'erreur HTML déguisées en 200 :

```python
content = r.content
if len(content) < 10_000:
    continue  # moins de 10 Ko = contenu invalide
```

### Organisation de l'archive

Les fichiers sont organisés selon une hiérarchie `date/zone/` :

```
data/flash_audio_archive/
├── index.csv
├── state.json
└── 2026-01-23/
    ├── nord/
    │   └── flash_nord_20260123_164916.mp3
    ├── sud/
    │   └── flash_sud_20260123_164917.mp3
    └── ouest/
        └── flash_ouest_20260123_164918.mp3
```

Le nom de fichier encode la zone et un timestamp compact local (`YYYYMMDD_HHMM`), ce qui permet de reconstituer la date à partir du nom seul.

### Index CSV et traçabilité

Chaque fichier sauvegardé est logué dans un `index.csv` avec les colonnes :

```
datetime_local;datetime_utc;zone;filename;filesize;md5
2026-01-23 16:49:16+0100;2026-01-23 15:49:16;nord;2026-01-23/nord/flash_nord_20260123_164916.mp3;1237850;5ad694...
```

Deux timestamps sont stockés : local (Europe/Paris) et UTC, ce qui évite toute ambiguïté lors de l'analyse ultérieure.

### Rotation automatique des fichiers

Une fonction de rotation supprime les fichiers plus vieux que `keep_days` jours :

```python
def rotate(path: Path, keep_days: int, tz) -> None:
    if keep_days <= 0:
        return
    limit = now_local(tz) - dt.timedelta(days=keep_days)
    for f in path.glob("**/*.mp3"):
        parts = f.stem.rsplit("_", 2)
        stamp = parts[-2] + "_" + parts[-1]
        d = parse_stamp_to_dt(stamp, tz)
        if d < limit:
            f.unlink(missing_ok=True)
```

Le parsing du timestamp depuis le nom de fichier supporte deux formats : avec et sans offset timezone (`%Y%m%d_%H%M%z` et `%Y%m%d_%H%M`).

---

## Couche 2 — Transcription audio (ASR)

### Conversion audio avec ffmpeg

Whisper attend de l'audio WAV mono 16 kHz PCM 16-bit. Le module `audio_utils.py` délègue cette conversion à `ffmpeg` via subprocess :

```python
def convert_to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    ensure_ffmpeg_or_raise()
    cmd = [
        which_ffmpeg(),
        "-y",           # overwrite sans confirmation
        "-i", src_path,
        "-vn",          # pas de vidéo
        "-ac", "1",     # mono
        "-ar", "16000", # 16 kHz
        "-c:a", "pcm_s16le",  # PCM 16-bit little-endian
        dst_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion échouée:\n{p.stderr}")
```

Cette approche supporte nativement tous les formats gérés par ffmpeg : mp3, m4a, aac, flac, ogg, webm, etc.

La fonction `rms()` permet de mesurer le niveau RMS d'un signal audio pour détecter les silences avant transcription :

```python
def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
```

### WhisperService — encapsulation de faster-whisper

`faster-whisper` est une réimplémentation de Whisper basée sur CTranslate2, significativement plus rapide que l'implémentation originale d'OpenAI, avec un support natif de la quantification int8.

La classe `WhisperService` implémente un cache de modèle — le modèle n'est rechargé que si les paramètres changent :

```python
class WhisperService:
    def load(self, model_name: str, device: str = "cpu", compute_type: Optional[str] = None):
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        # Cache : ne recharge que si les paramètres changent
        if (
            self._model is not None
            and self._model_name == model_name
            and self._device == device
            and self._compute_type == compute_type
        ):
            return

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
```

**Compute types :**
- CPU → `int8` : quantification 8-bit, réduit la mémoire et accélère l'inférence
- GPU → `float16` : précision demi-flottant, optimal pour CUDA

### VAD — Voice Activity Detection

La transcription utilise le filtre VAD intégré de faster-whisper pour ignorer les segments silencieux :

```python
segments, info = self._model.transcribe(
    wav_path,
    language=language,
    beam_size=beam_size,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 350},
)
```

Le paramètre `min_silence_duration_ms=350` définit la durée minimale de silence pour couper un segment. Cela évite les hallucinations de Whisper sur les silences prolongés (un problème connu du modèle).

### Deux modes de sortie

**Mode simple** — retourne un tuple `(texte, langue, probabilité)` :

```python
def transcribe_wav(self, wav_path, language, beam_size) -> Tuple[str, str, float]:
    ...
    return " ".join(texts), info.language, float(info.language_probability)
```

**Mode segmenté** — retourne un dict structuré avec les timestamps de chaque segment :

```python
def transcribe_wav_with_segments(self, wav_path, language, beam_size, min_silence_ms=500) -> dict:
    ...
    return {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "segments": [{"start": s.start, "end": s.end, "text": text} for s in segs],
        "text": " ".join(texts),
    }
```

Le mode segmenté est utile pour localiser précisément un événement dans l'audio ou générer des sous-titres.

### Choix du modèle Whisper

| Modèle | Taille | VRAM | WER approximatif FR | Vitesse (CPU) |
|--------|--------|------|---------------------|---------------|
| `tiny` | 39M | ~1 GB | ~14% | très rapide |
| `base` | 74M | ~1 GB | ~10% | rapide |
| `small` | 244M | ~2 GB | ~7% | modéré |
| `medium` | 769M | ~5 GB | ~4% | lent |
| `large-v3` | 1550M | ~10 GB | ~2% | très lent |

Pour de la parole radio française bien articulée, `small` ou `medium` offrent le meilleur compromis précision/vitesse sur CPU.

---

## Couche 3 — Analyse NLP par extraction de motifs

### Approche : NLP symbolique par expressions régulières

Le choix d'utiliser des regex plutôt qu'un modèle de NLP entraîné (NER, classification) est délibéré : les domaines métier bien définis avec un vocabulaire contraint se prêtent parfaitement aux approches symboliques, qui offrent une précision prévisible et une latence nulle.

### Taxonomie des événements

Huit types d'événements sont détectés avec trois niveaux de sévérité :

```python
_SEVERITY: dict[str, str] = {
    "accident":        "high",
    "fermeture":       "high",
    "bouchon":         "medium",
    "animal":          "medium",
    "intemperies":     "medium",
    "ralentissement":  "low",
    "travaux":         "low",
    "vehicule_panne":  "low",
}
```

### Patterns de détection

Chaque type est détecté par une regex compilée couvrant les variantes orthographiques et morphologiques françaises :

```python
_PATTERNS: dict[str, re.Pattern] = {
    "accident": re.compile(
        r'\b(accident|collision|accrochage|carambolage|heurt(?:é|er)?|percuté|renversé)\b',
        re.IGNORECASE,
    ),
    "intemperies": re.compile(
        r'\b(neige|verglas|brouillard|pluie\s+vergla[cç]ante|givr[eé]|black.ice|'
        r'alerte\s+orange|alerte\s+rouge|conditions\s+hivernales|chaine|pneus?\s+hiver)\b',
        re.IGNORECASE,
    ),
    # ...
}
```

Points techniques notables :
- `\b` : word boundary, évite les faux positifs dans des mots composés
- `(?:é|er)?` : groupe non-capturant pour les variantes de conjugaison
- `[cç]` : classe de caractères pour les accents alternants
- `\s+` : tolérance sur les espaces multiples dans les expressions composées
- `re.IGNORECASE` : insensibilité à la casse (transcriptions parfois en majuscules)

### Extraction contextuelle

Pour chaque événement détecté, un extrait de contexte de ±80 caractères autour du match est capturé :

```python
_CONTEXT_RADIUS = 80

def _extract_context(text: str, match_start: int, match_end: int) -> str:
    start = max(0, match_start - _CONTEXT_RADIUS)
    end = min(len(text), match_end + _CONTEXT_RADIUS)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet
```

### Extraction de routes, directions et délais

```python
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
```

### Modèle de données

Les événements sont représentés par un dataclass :

```python
@dataclass
class TrafficEvent:
    type: str           # "accident", "bouchon", etc.
    severity: str       # "high", "medium", "low"
    routes: List[str]   # ["A6", "N7"]
    direction: str      # "sens Paris"
    location_hint: str  # extrait de contexte
    zone: str           # "nord", "sud", "ouest"
    timestamp: str      # "20260123_1649"
    source_file: str    # chemin relatif du MP3
    delay_hint: str     # "20 minutes"

    def as_dict(self) -> dict: ...
```

L'interface `as_dict()` permet la sérialisation JSON directe.

### Fonction principale d'extraction

```python
def extract_events(text: str, zone: str, source_file: str, timestamp: str) -> List[TrafficEvent]:
    events = []
    routes = _extract_routes(text)
    direction = _extract_direction(text)
    delay = _extract_delay(text)

    for event_type, pattern in _PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        hint = _extract_context(text, match.start(), match.end())
        events.append(TrafficEvent(
            type=event_type,
            severity=_SEVERITY[event_type],
            routes=routes,
            direction=direction,
            location_hint=hint,
            zone=zone,
            timestamp=timestamp,
            source_file=source_file,
            delay_hint=delay,
        ))

    return events
```

Les routes, direction et délai sont extraits **une seule fois** en global sur le texte, puis partagés entre tous les événements détectés — ce qui est correct car un flash audio porte généralement sur une zone géographique cohérente.

---

## Couche 4 — Notification multi-canaux

Le module `notifier.py` implémente un dispatch vers quatre cibles :

### Console

```python
def notify_console(event: TrafficEvent) -> None:
    icon = _SEVERITY_ICON.get(event.severity, "⚪")  # 🔴 🟠 🟡
    print(
        f"{icon} [{event.timestamp}] {event.zone.upper():5s} | "
        f"{event.type.upper():18s} | {routes}{direction}{delay}\n"
        f"   {event.location_hint}",
        flush=True,
    )
```

### Notification macOS native (osascript)

```python
def notify_macos(event: TrafficEvent) -> None:
    script = f'display notification "{body_safe}" with title "{title_safe}"'
    subprocess.run(["osascript", "-e", script], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

L'échappement des guillemets dans le corps AppleScript est géré manuellement pour éviter l'injection de commandes. La fonction est silencieuse si `osascript` est absent (Linux, CI).

### Webhook HTTP

```python
def notify_webhook(event: TrafficEvent, url: str) -> None:
    payload = event.as_dict()
    payload["alerted_at"] = datetime.now(timezone.utc).isoformat()
    _requests.post(url, json=payload, timeout=5)
```

Le payload JSON est horodaté UTC et contient tous les champs du `TrafficEvent`. Compatible avec n'importe quel endpoint HTTP (Slack, Discord, n8n, Make, etc.).

### Log JSONL

```python
def log_to_file(event: TrafficEvent, alerts_dir: Path) -> None:
    line = event.as_dict()
    line["alerted_at"] = datetime.now(timezone.utc).isoformat()
    with (alerts_dir / "alerts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
```

Le format JSONL (une ligne = un objet JSON) est optimal pour les logs : append-only, facilement parsable avec `jq`, `pandas`, ou DuckDB.

### Dispatch centralisé

```python
def dispatch(event, alerts_dir, macos=False, webhook_url=None) -> None:
    notify_console(event)
    log_to_file(event, alerts_dir)
    if macos:
        notify_macos(event)
    if webhook_url:
        notify_webhook(event, webhook_url)
```

---

## Stratégie de tests

Le projet dispose de **5 fichiers de tests** couvrant chaque couche indépendamment avec `pytest` et `pytest-mock`.

### Test de l'extracteur NLP

L'extracteur est testé avec des textes représentatifs de la parole radio :

```python
_TEXT_ACCIDENT = (
    "Autoroute Info, bonjour. Un accident sur l'A6 sens Paris au niveau du km 34, "
    "deux véhicules impliqués, circulation très ralentie. Comptez 20 minutes de retard."
)

def test_detects_accident():
    events = extract_events(_TEXT_ACCIDENT, zone="nord", source_file="f.mp3", timestamp="20260122_1000")
    assert "accident" in [e.type for e in events]

def test_no_false_positive_neutral_text():
    events = extract_events(
        "Le réseau autoroutier est fluide sur l'ensemble du territoire.",
        zone="nord", source_file="f.mp3", timestamp="20260122_1000"
    )
    assert events == []
```

Les tests de faux positifs sont aussi importants que les vrais positifs.

### Test du fetcher avec mock HTTP

```python
def test_fetch_once_conditional_deduplicates(tmp_path, mocker):
    content = b"\xff" * 15_000
    mocker.patch(
        "flash_nlp.acquisition.fetcher.requests.get",
        return_value=_make_response(200, content),
    )
    cond_state = {}
    added1 = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    added2 = fetch_once_conditional(tmp_path, keep_days=30, tz=TZ_PARIS, cond_state=cond_state)
    assert added1 == 3  # 3 zones → 3 fichiers
    assert added2 == 0  # doublon MD5 → rien sauvegardé
```

### Test de WhisperService avec mock modèle

```python
def test_transcribe_wav_passes_correct_args(mocker):
    _, mock_instance = _mock_model(mocker)
    svc = WhisperService()
    svc.load("small")
    svc.transcribe_wav("audio.wav", language="fr", beam_size=3)

    mock_instance.transcribe.assert_called_once_with(
        "audio.wav",
        language="fr",
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
    )
```

Le modèle Whisper est mocké entièrement — les tests ne chargent aucun modèle réel.

### Test des utilitaires audio

```python
def test_rms_known_sine():
    # RMS d'un sinus d'amplitude A = A / sqrt(2)
    t = np.linspace(0, 2 * np.pi, 10000)
    x = np.sin(t).astype(np.float32)
    assert abs(rms(x) - 1.0 / np.sqrt(2)) < 1e-3

def test_save_wav_clips_overflow(tmp_path):
    audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    save_wav(str(tmp_path / "out.wav"), audio, sr=16000)
    _, data = wav_read(str(tmp_path / "out.wav"))
    assert data[0] == 32767   # clip positif
    assert data[1] == -32767  # clip négatif
```

### Configuration pytest

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

Lancer les tests :

```bash
pip install pytest pytest-mock
pytest
```

---

## Modèles TTS embarqués

Le projet inclut des modèles TTS (Text-to-Speech) de Meta MMS (Massively Multilingual Speech) pour l'anglais et l'ukrainien :

```
models/
├── mms-tts-eng/   # TTS anglais
│   ├── config.json
│   ├── model.safetensors
│   ├── vocab.json
│   └── tokenizer_config.json
└── mms-tts-ukr/   # TTS ukrainien
    └── ...
```

MMS est une architecture Vits (Variational Inference with adversarial learning for end-to-end Text-to-Speech) fine-tunée sur 1100+ langues. Ces modèles permettent d'envisager une synthèse vocale des alertes générées — par exemple pour produire une version audio traduite en ukrainien.

---

## Extension possible : pipeline complet ASR → NLP → TTS

L'architecture permet d'enchaîner les couches pour un pipeline bout-en-bout :

```
Audio source (FR)
      ↓
[Conversion WAV 16k mono] — ffmpeg
      ↓
[Transcription ASR] — faster-whisper
      ↓
[Extraction événements] — regex NLP
      ↓
[Génération texte traduit] — LLM ou MarianMT
      ↓
[Synthèse vocale] — MMS TTS (ukr, eng...)
      ↓
Audio cible (UKR / ENG)
```

La partie traduction (LLM ou MarianMT) et la synthèse TTS via MMS sont les extensions naturelles du projet.

---

## Dépendances externes requises

| Outil | Usage | Installation |
|-------|-------|--------------|
| `ffmpeg` | Conversion audio | `brew install ffmpeg` / `apt install ffmpeg` |
| Python 3.11+ | Runtime | [python.org](https://python.org) |
| `faster-whisper` | ASR | `pip install faster-whisper` |
| `scipy` / `numpy` | Audio processing | inclus dans les deps |
| `sounddevice` | Listing périphériques audio | inclus dans les deps |

---

## Structure du projet

```
.
├── pyproject.toml
├── requirements.txt
├── src/
│   └── flash_nlp/
│       ├── acquisition/
│       │   └── fetcher.py
│       ├── transcription/
│       │   ├── whisper_service.py
│       │   └── audio_utils.py
│       ├── analysis/
│       │   ├── event_extractor.py
│       │   └── notifier.py
│       └── io/
│           └── file_utils.py
├── tests/
│   ├── test_fetcher.py
│   ├── test_whisper_service.py
│   ├── test_audio_utils.py
│   ├── test_event_extractor.py
│   └── test_io.py
├── data/
│   └── flash_audio_archive/
│       ├── index.csv
│       └── state.json
├── models/
│   ├── mms-tts-eng/
│   └── mms-tts-ukr/
└── outputs/
```

---

## Conclusion

Ce projet démontre comment combiner plusieurs briques open-source — faster-whisper, ffmpeg, scipy, Meta MMS — en un pipeline NLP cohérent pour le traitement automatique de flux audio publics. Les choix techniques (polling conditionnel HTTP, déduplification MD5, VAD Whisper, NLP symbolique par regex, format JSONL) privilégient la robustesse et la maintenabilité sur la complexité.

L'architecture modulaire permet d'étendre le pipeline vers la traduction automatique et la synthèse vocale multilingue, ouvrant la voie à des applications d'accessibilité linguistique en temps réel.

---

*Article rédigé pour [dev-ai.fr](https://dev-ai.fr) — mars 2026*
