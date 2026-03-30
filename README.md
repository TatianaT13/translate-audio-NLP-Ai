# traduction-audio — LLMOps Audio Translation Pipeline

Système LLMOps de traduction audio temps réel : **Audio FR → Transcription → Traduction EN/UK/ES/DE → Synthèse vocale**.

Architecture microservices avec orchestration Langchain LCEL, tracing Langfuse, évaluation BLEU sur 84 runs.

---

## Architecture

```
Client (Next.js)
       │
       │ POST /process (audio)
       ▼
┌─────────────────────────────────────────┐
│  Pipeline Service  :8000                │
│  Langchain LCEL orchestrateur           │
│  STT ──► LLM ──► TTS                   │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
  STT :8001   LLM :8002   TTS :8003
  Whisper     LiteLLM     Mistral
              + Groq       Voxtral
```

| Service | Port | Technologie |
|---------|------|-------------|
| Pipeline (orchestrateur) | 8000 | FastAPI + Langchain LCEL |
| STT | 8001 | FastAPI + Faster-Whisper |
| LLM | 8002 | FastAPI + LiteLLM + Groq |
| TTS | 8003 | FastAPI + Mistral Voxtral |
| Frontend | 3000 | Next.js |

---

## Démarrage rapide

### Prérequis

- Docker Desktop
- Clé API Groq : [console.groq.com](https://console.groq.com)
- Clé API Mistral : [console.mistral.ai](https://console.mistral.ai)

### Configuration

```bash
cp .env.example .env
# Renseigner dans .env :
# GROQ_API_KEY=gsk_...
# MISTRAL_API_KEY=...
# MISTRAL_VOICE_ID=...  (voice ID créé sur console.mistral.ai)
```

### Lancement avec Docker

```bash
docker compose up --build
```

Les 4 services démarrent :
- Pipeline : http://localhost:8000/docs
- STT : http://localhost:8001/docs
- LLM : http://localhost:8002/docs
- TTS : http://localhost:8003/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Structure

```
.
├── services/
│   ├── pipeline/           # Orchestrateur Langchain LCEL (port 8000)
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── stt/                # Speech-to-Text Whisper (port 8001)
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── llm/                # Traduction LiteLLM/Groq (port 8002)
│   │   ├── main.py
│   │   └── Dockerfile
│   └── tts/                # Synthèse vocale Mistral Voxtral (port 8003)
│       ├── main.py
│       └── Dockerfile
├── src/
│   └── flash_nlp/          # Package Python (Whisper, audio utils)
├── frontend/               # Interface Next.js
├── scripts/
│   ├── run_pipeline.py     # Pipeline CLI (hors Docker)
│   ├── eval_golden.py      # Évaluation BLEU sur dataset golden
│   └── langfuse_import.py  # Import des 84 runs dans Langfuse
├── outputs/
│   └── experiments/
│       ├── results.csv              # 84 runs (12 combos × 7 audios)
│       └── evaluation_report.md    # Rapport BLEU par modèle/prompt
├── data/
│   └── flash_audio_archive/        # Archive MP3 trafic
├── docker-compose.yml
└── pyproject.toml
```

---

## Évaluation (Phase 1)

84 runs évalués : 3 modèles Whisper × 4 prompts × 7 fichiers audio.

| Rang | Whisper | LLM | Prompt | BLEU moyen |
|------|---------|-----|--------|-----------|
| 1 | large-v3 | llama-3.1-8b | v1.1 | 31.55 |
| 2 | large-v3 | llama-3.1-8b | v1.2 | 30.87 |
| 3 | medium | llama-3.1-8b | v1.1 | 29.43 |

Rapport complet : [outputs/experiments/evaluation_report.md](outputs/experiments/evaluation_report.md)

```bash
# Relancer l'évaluation
python scripts/eval_golden.py --whisper-model small --model groq/llama-3.1-8b-instant
```

---

## Pipeline CLI (sans Docker)

```bash
# Installer les dépendances
pip install -e ".[dev]"

# Lancer le pipeline sur un fichier audio
python scripts/run_pipeline.py \
    --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3 \
    --model groq/llama-3.1-8b-instant \
    --target-lang en \
    --prompt-version v1.1
```

---

## Tracing Langfuse

Toutes les exécutions sont tracées dans [Langfuse](https://cloud.langfuse.com) :
- Latences STT / LLM / TTS
- Score BLEU (quand référence disponible)
- Version de prompt utilisée

```bash
# Importer les 84 runs historiques dans Langfuse
python scripts/langfuse_import.py
```

Variables requises dans `.env` :
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

81 tests couvrant : acquisition, transcription, audio utils, NLP, I/O.

CI GitHub Actions : `.github/workflows/ci.yml` (pytest sur chaque push/PR).

---

## Variables d'environnement

| Variable | Description | Requis |
|----------|-------------|--------|
| `GROQ_API_KEY` | Clé API Groq (LLM) | Oui |
| `MISTRAL_API_KEY` | Clé API Mistral (TTS) | Oui |
| `MISTRAL_VOICE_ID` | ID de voix Mistral Voxtral | Oui |
| `WHISPER_MODEL` | Modèle Whisper (`small`, `large-v3`) | Non (défaut: `small`) |
| `LLM_MODEL` | Modèle LiteLLM | Non (défaut: `groq/llama-3.1-8b-instant`) |
| `PROMPT_VERSION` | Version du prompt (`v1.0`–`v1.2`) | Non (défaut: `v1.1`) |
| `LANGFUSE_PUBLIC_KEY` | Clé publique Langfuse | Non |
| `LANGFUSE_SECRET_KEY` | Clé secrète Langfuse | Non |
| `LANGFUSE_HOST` | URL Langfuse | Non |

---

## Roadmap

- [x] Phase 1 - Dataset golden + evaluation BLEU (84 runs)
- [x] Phase 2 - Microservices Docker (STT / LLM / TTS)
- [x] Phase 2 - Langfuse tracing
- [x] Phase 2 - CI GitHub Actions
- [x] Phase 3 - Pipeline Service orchestrateur (Langchain LCEL)
- [x] Phase 3 - Frontend Next.js
- [ ] Phase 3 - API Gateway (auth + rate limiting)
- [ ] Phase 4 - MLflow model registry
- [ ] Phase 4 - Prometheus + Grafana
- [ ] Phase 4 - Airflow batch evaluation

---

## Licence

[LICENSE](LICENSE)
