# Plan de développement — flash-nlp -> LLMOps Pipeline
> Soutenance : 4 septembre 2026

---

## Ce qu'on a déjà (la 1ère moitié)

| Composant | Fichier | Statut |
|---|---|---|
| Acquisition audio HTTP (ETag + MD5) | `acquisition/fetcher.py` | OK |
| Conversion MP3 -> WAV 16kHz mono (ffmpeg) | `transcription/audio_utils.py` | OK |
| Transcription Whisper + VAD | `transcription/whisper_service.py` | OK |
| Extraction NLP regex (8 types d'événements) | `analysis/event_extractor.py` | OK |
| Notifications (console, macOS, webhook, JSONL) | `analysis/notifier.py` | OK |
| Suite de tests pytest complète (5 fichiers) | `tests/` | OK |
| Packaging installable | `pyproject.toml` | OK |
| Modèles TTS locaux (MMS eng + ukr) | `models/` | OK |

---

## Ce qui manque (la 2ème moitié)

### Expérimentation & traçabilité
- Aucun run comparatif enregistré (pas de Langfuse / MLflow actif)
- Pas de dataset golden (données de référence pour évaluer)
- Pas de métriques objectives (BLEU, latence par étape)
- Pas de versioning de prompts

### Inférence LLM
- Pas de service de traduction (le pipeline s'arrête à la transcription)
- Pas de TTS branché (les modèles MMS sont présents mais non utilisés)
- Pas de LiteLLM / Ollama configuré

### Infrastructure
- Pas de Dockerfile / docker-compose
- Pas de FastAPI exposant les services
- Pas de pipeline bout-en-bout (pipeline/__init__.py vide)
- Pas de CI/CD (GitHub Actions)

### Monitoring
- Pas de Prometheus / Grafana
- Pas d'Airflow pour l'évaluation batch
- Pas d'alertes sur dérive de qualité

---

## Phase 1 — Fondations & Expérimentation
> Deadline : fin avril 2026

### Objectif
Valider scientifiquement quelle combinaison **Modèle + Prompt** donne les meilleurs résultats de traduction.
C'est le livrable MLOps le plus important : sans cette phase, tout le reste n'a pas de sens.

### 1.1 — Environnement (UV)
- [ ] Migrer de `venv` vers UV pour la gestion des dépendances
- [ ] Créer `uv.lock` versionné dans git
- [ ] Documenter `uv sync` dans le README

```bash
pip install uv
uv venv
uv pip install -e ".[dev]"
```

### 1.2 — Dataset Golden
- [ ] Constituer 50+ fichiers audio de référence
- [ ] Associer à chaque audio une transcription humaine validée
- [ ] Associer une traduction humaine validée (FR -> EN ou FR -> UK)
- [ ] Stocker dans `data/golden/` avec un manifest JSON

```
data/golden/
├── manifest.json          # { "id": "001", "audio": "...", "ref_text": "...", "ref_translation": "..." }
├── audio/
│   ├── golden_001.mp3
│   └── ...
└── references/
    ├── golden_001.txt     # transcription de référence
    └── ...
```

### 1.3 — Intégration Langfuse
- [ ] Installer Langfuse (Docker local ou cloud free tier)
- [ ] Logger chaque run : modèle + prompt_version + métriques
- [ ] Versionner les prompts dans Langfuse (v1.0, v1.1, v1.2...)

```python
# Ce que chaque run doit logger :
{
  "model":             "llama3:8b",
  "prompt_version":    "v1.2",
  "audio_id":          "golden_001",
  "bleu_score":        0.72,
  "meteor_score":      0.68,
  "latency_stt_ms":    1200,
  "latency_llm_ms":    640,
  "language_prob":     0.97,   # déjà dispo dans WhisperService
  "whisper_model":     "small"
}
```

### 1.4 — Ollama + modèles LLM
- [ ] Installer Ollama en local
- [ ] Télécharger 2-3 modèles : `llama3:8b`, `mistral:7b`, `phi3:mini`
- [ ] Créer un client Python simple `src/flash_nlp/llm/client.py`

### 1.5 — Versions de prompts
Créer 2-5 prompts de traduction dans `prompts/` :

```
prompts/
├── v1.0_baseline.txt       # "Traduis ce texte en anglais : {text}"
├── v1.1_context.txt        # Avec instruction de contexte (trafic routier)
├── v1.2_formal.txt         # Ton formel
├── v1.3_simple.txt         # Simplifié (accessibilité)
└── v1.4_structured.txt     # JSON output structuré
```

### 1.6 — Script d'évaluation
- [ ] Créer `scripts/eval_golden.py`
- [ ] Calculer BLEU + METEOR pour chaque combinaison modèle x prompt
- [ ] Exporter les résultats dans `outputs/experiments/results.csv`
- [ ] Logger automatiquement dans Langfuse

### 1.7 — Livrable Phase 1
> Un tableau comparatif + document de sélection :
> "La combinaison gagnante est X + prompt vY car BLEU=0.74, latence=1.8s"

---

## Phase 2 — Microservices & Registres
> Deadline : fin avril 2026

### Objectif
Conteneuriser chaque capacité d'inférence dans un service HTTP indépendant.

### 2.1 — STT Service (Port 8001)
- [ ] Créer `services/stt/main.py` (FastAPI)
- [ ] Wrapper `WhisperService` avec endpoint `POST /transcribe`
- [ ] Dockerfile `services/stt/Dockerfile`
- [ ] Tests d'intégration `tests/test_stt_service.py`

```python
# POST /transcribe
# Input  : multipart audio file
# Output : { "text": "...", "language": "fr", "language_probability": 0.97, "segments": [...] }
```

### 2.2 — LLM Service (Port 8002)
- [ ] Créer `services/llm/main.py` (FastAPI)
- [ ] Endpoint `POST /translate` via LiteLLM (bascule Ollama / GROQ / Anthropic)
- [ ] Dockerfile `services/llm/Dockerfile`
- [ ] Variable d'env `LLM_PROVIDER=ollama|groq|anthropic`

```python
# POST /translate
# Input  : { "text": "...", "source_lang": "fr", "target_lang": "en", "prompt_version": "v1.2" }
# Output : { "translation": "...", "model": "llama3:8b", "latency_ms": 640 }
```

### 2.3 — TTS Service (Port 8003)
- [ ] Créer `services/tts/main.py` (FastAPI)
- [ ] Brancher les modèles MMS-TTS déjà présents (`models/mms-tts-eng/`)
- [ ] Endpoint `POST /synthesize` -> retourne un fichier WAV
- [ ] Dockerfile `services/tts/Dockerfile`

```python
# POST /synthesize
# Input  : { "text": "...", "language": "en" }
# Output : audio/wav stream
```

### 2.4 — MLflow (Registre de modèles)
- [ ] Lancer MLflow en local via Docker
- [ ] Enregistrer les modèles binaires (whisper, MMS-TTS) comme artifacts
- [ ] Versionner : `whisper-small v1`, `mms-tts-eng v1`
- [ ] Les services chargent leurs modèles depuis MLflow au démarrage

### 2.5 — MinIO (Stockage objet)
- [ ] Lancer MinIO via Docker
- [ ] Créer les buckets : `input-audio`, `output-audio`, `golden-dataset`
- [ ] Adapter `fetcher.py` pour uploader les MP3 dans MinIO

### 2.6 — docker-compose Phase 2

```yaml
services:
  stt:      build: services/stt    ports: ["8001:8001"]
  llm:      build: services/llm    ports: ["8002:8002"]
  tts:      build: services/tts    ports: ["8003:8003"]
  mlflow:   image: mlflow/mlflow   ports: ["5000:5000"]
  minio:    image: minio/minio     ports: ["9000:9000"]
  langfuse: image: langfuse/langfuse
```

---

## Phase 3 — Orchestration & API Gateway
> Deadline : mi-avril 2026

### Objectif
Assembler les services en un pipeline cohérent exposé via une API unique.

### 3.1 — Pipeline Service (Orchestrateur Langchain LCEL)
- [ ] Créer `services/pipeline/main.py`
- [ ] Implémenter la chaîne LCEL : STT -> LLM -> TTS
- [ ] Gestion des erreurs (retry, timeout, circuit breaker)
- [ ] Envoi des traces à Langfuse à chaque étape

```python
# services/pipeline/main.py
from langchain_core.runnables import chain

@chain
async def full_pipeline(audio_bytes: bytes):
    text        = await call_stt(audio_bytes)
    translation = await call_llm(text, prompt_version="v1.2")
    audio_out   = await call_tts(translation)
    return {"text": text, "translation": translation, "audio": audio_out}
```

### 3.2 — API Gateway
- [ ] Configurer Nginx ou Traefik comme reverse proxy
- [ ] Routing : `POST /api/v1/translate` -> pipeline:8000
- [ ] Authentification JWT basique
- [ ] Rate limiting

### 3.3 — Interface Streamlit (optionnelle)
- [ ] Upload fichier audio
- [ ] Affichage transcription + traduction
- [ ] Téléchargement audio traduit
- [ ] Affichage des métriques du run (latence, modèle utilisé)

### 3.4 — CLI bout-en-bout
- [ ] Compléter `src/flash_nlp/pipeline/__init__.py`
- [ ] Script CLI `python -m flash_nlp.pipeline run --audio audio.mp3 --target-lang en`

---

## Phase 4 — Monitoring & Évaluation Batch
> Deadline : mi-août 2026

### Objectif
Surveiller la qualité en production et détecter les dérives automatiquement.

### 4.1 — Prometheus + Grafana
- [ ] Exposer des métriques depuis chaque service (endpoint `/metrics`)
- [ ] Métriques clés à collecter :
  - Latence STT (ms) — p50, p95, p99
  - Latence LLM (ms)
  - Score BLEU moyen (fenêtre glissante 24h)
  - `language_probability` Whisper (si < 0.8 -> alerte audio dégradé)
  - VRAM GPU utilisée
- [ ] Dashboard Grafana "Pipeline Health"
- [ ] Alertes si BLEU < 0.6 ou latence p95 > 5s

### 4.2 — Airflow (Évaluation batch)
- [ ] DAG `eval_golden` : quotidien / hebdomadaire
  1. Extract : récupérer dataset golden depuis MinIO
  2. Run : appeler directement le Pipeline Service
  3. Evaluate : calculer BLEU + METEOR via Ragas / Evidently
  4. Compare : comparer avec baseline enregistrée
  5. Alert : Slack si dérive détectée
- [ ] DAG `fetch_archive` : remplacer le cron actuel par un DAG Airflow

### 4.3 — CI/CD GitHub Actions
- [ ] Workflow `ci.yml` : lint + pytest sur chaque push
- [ ] Workflow `eval.yml` : évaluation sur dataset golden à chaque PR
- [ ] Workflow `build.yml` : docker build + push sur merge main

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install uv && uv sync
      - run: pytest tests/ -v
  eval:
    runs-on: ubuntu-latest
    steps:
      - run: python scripts/eval_golden.py --output outputs/ci_results.csv
```

---

## Recap "2ème moitié" à construire

```
CE QU'ON A (1ère moitié)           CE QUI MANQUE (2ème moitié)
──────────────────────────────      ──────────────────────────────
WhisperService              OK      FastAPI wrapper STT           TODO
audio_utils (ffmpeg)        OK      LiteLLM / Ollama              TODO
event_extractor (NLP)       OK      Langchain LCEL pipeline       TODO
notifier (webhook)          OK      Langfuse tracking             TODO
pytest suite                OK      Dataset golden 50+ audios     TODO
pyproject.toml              OK      Script eval BLEU/METEOR       TODO
MMS-TTS models présents     OK      TTS service branché           TODO
index.csv logs              OK      MLflow registre actif         TODO
fetcher.py (archive)        OK      MinIO stockage                TODO
                                    Docker / docker-compose       TODO
                                    API Gateway                   TODO
                                    Prometheus + Grafana          TODO
                                    Airflow DAGs                  TODO
                                    GitHub Actions CI/CD          TODO
```

---

## Retro-planning

| Période | Phase | Livrables clés |
|---|---|---|
| Maintenant -> 20 avril | Phase 1 | Dataset golden + Langfuse + 12 runs comparatifs + combinaison gagnante |
| 20 -> 30 avril | Phase 2 | 3 services Docker + MLflow + MinIO + docker-compose |
| 30 avril -> mi-mai | Phase 3 | Pipeline LCEL + API Gateway + Streamlit |
| Mai -> Juillet | Phase 3 polish | Tests intégration + CI/CD + CLI |
| Juillet -> mi-août | Phase 4 | Prometheus + Grafana + Airflow DAGs |
| Mi-août -> 28 août | Finalisation | Documentation + slides + démo complète |
| 4 septembre | Soutenance | Démo live : upload audio -> traduction -> métriques |

---

## Objectif soutenance

> "J'ai testé 3 modèles x 4 prompts = 12 combinaisons.
> Le run #7 (Llama3 8B + prompt v1.2) donne BLEU=0.74, latence=1.8s.
> Ce modèle est enregistré dans MLflow v2, déployé via Docker,
> et surveillé par Prometheus. Si le BLEU tombe sous 0.6, une alerte part sur Slack."

---

## Stack technologique cible

| Couche | Technologie |
|---|---|
| STT | faster-whisper (déjà intégré) |
| LLM | Ollama (local) + LiteLLM proxy |
| TTS | MMS-TTS Meta (déjà présent) |
| Orchestration | Langchain LCEL + FastAPI |
| Expérimentation | Langfuse |
| Registre modèles | MLflow |
| Stockage | MinIO (S3-like local) |
| Gateway | Nginx ou Traefik |
| UI | Streamlit |
| Monitoring | Prometheus + Grafana |
| Batch | Apache Airflow |
| CI/CD | GitHub Actions |
| Env | UV + Docker |
