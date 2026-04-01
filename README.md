# traduction-audio — LLMOps Audio Translation Pipeline

Système LLMOps de traduction audio temps réel : **Audio FR → Transcription → Traduction EN/UK/ES/DE → Synthèse vocale**.

Architecture microservices avec orchestration Langchain LCEL, authentification JWT, tracing Langfuse, évaluation BLEU/METEOR/WER sur 84 runs, monitoring trafic autoroutier temps réel.

---

## Architecture

```
Client (Next.js)
       │
       │ POST /auth/login  │  POST /process (audio)
       ▼
┌──────────────────────────┐
│  Gateway Service  :8004  │   Auth JWT + admin API
│  FastAPI + SQLAlchemy    │
└──────────┬───────────────┘
           │ proxy / forward
           ▼
┌─────────────────────────────────────────┐
│  Pipeline Service  :8000                │
│  Langchain LCEL orchestrateur           │
│  + Langfuse tracing                     │
│  STT ──► LLM ──► TTS                   │
└──────┬──────────┬──────────┬────────────┘
       │          │          │
       ▼          ▼          ▼
  STT :8001   LLM :8002   TTS :8003
  Whisper     LiteLLM     Mistral
  large-v3    + Groq       Voxtral

┌──────────────────────────┐
│  Watcher Service  :8005  │   Trafic Live (admin only)
│  Polling autorouteinfo   │
│  Whisper small + SSE     │
└──────────────────────────┘
```

| Service | Port | Technologie |
|---------|------|-------------|
| Gateway (auth + admin) | 8004 | FastAPI + SQLAlchemy + JWT |
| Pipeline (orchestrateur) | 8000 | FastAPI + Langchain LCEL + Langfuse |
| STT | 8001 | FastAPI + Faster-Whisper large-v3 |
| LLM | 8002 | FastAPI + LiteLLM + Groq |
| TTS | 8003 | FastAPI + Mistral Voxtral |
| Watcher (trafic live) | 8005 | FastAPI + Whisper small + SSE |
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
# MISTRAL_VOICE_ID=...       (voice ID créé sur console.mistral.ai)
# JWT_SECRET=...             (chaîne aléatoire 32+ chars)
# WHISPER_MODEL=large-v3     (défaut : small)
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
```

### Lancement

```bash
# Terminal 1 — backend (6 services Docker)
docker compose up --build

# Terminal 2 — frontend
cd frontend
npm install && npm run dev
# → http://localhost:3000
```

Services disponibles :
- Gateway : http://localhost:8004/docs
- Pipeline : http://localhost:8000/docs
- STT : http://localhost:8001/docs
- LLM : http://localhost:8002/docs
- TTS : http://localhost:8003/docs
- Watcher : http://localhost:8005/docs

### Créer le premier compte admin

```bash
# 1. S'inscrire sur http://localhost:3000/register
# 2. Promouvoir en admin
curl -X POST http://localhost:8004/admin/seed
# 3. Accéder au dashboard : http://localhost:3000/admin
```

---

## Authentification (Gateway JWT)

Le service Gateway (`services/gateway/`) gère toute l'authentification :

- **Register / Login** : création de compte + tokens JWT
- **Access token** (15 min) + **refresh token** rotatif (7 jours, hashé en base)
- **Mot de passe oublié** : flux reset par lien (DEV_MODE retourne l'URL directement)
- **Suppression de compte** et changement de mot de passe
- **Protection des routes** : middleware Next.js, redirection auto vers `/login`

Pages frontend : `/login`, `/register`, `/forgot-password`, `/reset-password`

---

## Dashboard Admin MLOps

Accessible à `/admin` pour les utilisateurs avec le rôle `is_admin`.

| Onglet | Contenu |
|--------|---------|
| Vue générale | Stats utilisateurs, KPIs pipeline (runs, latences, BLEU/METEOR/WER moyens) |
| Traces & Modèles | Latences STT/LLM/TTS, histogrammes BLEU/METEOR/WER/confiance, tableau comparatif modèles |
| Trafic Live | Incidents autoroutiers temps réel par zone (Nord/Sud/Ouest) via SSE — usage interne admin |
| Expériences | Stub MLflow (roadmap Phase 4) |
| Infrastructure | Health checks temps réel sur les 6 microservices |
| Pipelines | DAG cards + stub Airflow (roadmap Phase 4) |
| Utilisateurs | CRUD complet : activer, promouvoir admin, supprimer |

---

## Watcher — Trafic Live

Service dédié (`services/watcher/`) qui tourne en arrière-plan en permanence :

- Poll adaptatif toutes les **~15s** sur 3 flux autorouteinfo.fr (nord / sud / ouest)
- Requêtes conditionnelles ETag/Last-Modified — zéro bande passante si pas de changement
- **STT Whisper small** en mémoire — fichiers audio supprimés immédiatement après transcription
- Extraction d'événements trafic par regex (`event_extractor.py`) : accident, bouchon, animal, fermeture, intempéries, travaux, véhicule en panne
- Filtre automatique : seuls les événements `high` et `medium` sont conservés
- **Ring buffer `deque(maxlen=4)`** par zone — pas de base de données, zéro persistance
- **SSE `/stream`** → dashboard admin mis à jour en temps réel

> Usage strictement interne (admin uniquement). L'app publique ne redistribue pas ce contenu.

---

## Évaluation (Phase 1)

84 runs évalués : 2 modèles Whisper × 2 LLMs × 3 prompts × 7 fichiers audio.

**Métriques calculées** : BLEU (sacrebleu), METEOR (nltk), WER (jiwer — nécessite refs FR)

| Rang | Whisper | LLM | Prompt | BLEU moyen |
|------|---------|-----|--------|-----------|
| 1 | large-v3 | llama-3.1-8b | v1.1 | 31.55 |
| 2 | large-v3 | llama-3.1-8b | v1.2 | 30.87 |
| 3 | medium | llama-3.1-8b | v1.1 | 29.43 |

**Combinaison déployée** : `large-v3 + llama-3.1-8b-instant + prompt v1.1`

Rapport complet : [outputs/experiments/evaluation_report.md](outputs/experiments/evaluation_report.md)

```bash
# Relancer l'évaluation complète
python scripts/eval_golden.py

# Sur un seul audio
python scripts/eval_golden.py --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3

# Ajouter des références WER (transcriptions FR)
# → data/golden/references/flash_<nom>_fr.txt
```

---

## Tracing Langfuse

Toutes les exécutions du pipeline sont tracées dans [Langfuse](https://cloud.langfuse.com) :
- Latences STT / LLM / TTS par run
- Scores BLEU, METEOR, WER (quand références disponibles)
- Version de prompt utilisée
- Visible dans le Dashboard Admin → onglet **Traces & Modèles**

```bash
# Importer les 84 runs historiques dans Langfuse
python scripts/langfuse_import.py
```

---

## Pipeline CLI (sans Docker)

```bash
pip install -e ".[dev]"

python scripts/run_pipeline.py \
    --audio data/flash_audio_archive/2026-01-23/nord/flash_nord_20260123_164916.mp3 \
    --model groq/llama-3.1-8b-instant \
    --target-lang en \
    --prompt-version v1.1 \
    --whisper-model large-v3
```

---

## Structure

```
.
├── services/
│   ├── gateway/            # Auth JWT + admin API (port 8004)
│   │   ├── main.py         # Routes auth + admin + Langfuse + trafic proxy
│   │   ├── auth.py         # JWT, bcrypt, refresh tokens
│   │   ├── models.py       # User SQLAlchemy (is_admin, refresh_token_hash)
│   │   ├── schemas.py      # Pydantic schemas
│   │   ├── database.py     # SQLite/PostgreSQL engine
│   │   └── Dockerfile
│   ├── watcher/            # Trafic Live — polling autorouteinfo (port 8005)
│   │   ├── main.py         # Fetch + STT + event_extractor + SSE + ring buffer
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── pipeline/           # Orchestrateur Langchain LCEL + Langfuse (port 8000)
│   ├── stt/                # Speech-to-Text Whisper large-v3 (port 8001)
│   ├── llm/                # Traduction LiteLLM/Groq (port 8002)
│   └── tts/                # Synthèse vocale Mistral Voxtral (port 8003)
├── src/
│   └── flash_nlp/
│       ├── acquisition/    # Fetcher autorouteinfo (fetch_flashes.py)
│       ├── transcription/  # WhisperService, audio_utils
│       ├── analysis/       # event_extractor, notifier
│       └── io/             # file_utils
├── frontend/               # Interface Next.js
│   ├── app/
│   │   ├── page.tsx        # Page principale + UserMenu
│   │   ├── login/
│   │   ├── register/
│   │   ├── forgot-password/
│   │   ├── reset-password/
│   │   └── admin/          # Dashboard MLOps admin (7 onglets)
│   ├── lib/auth.ts         # Client auth
│   ├── lib/admin.ts        # Client API admin + SSE trafic
│   └── middleware.ts       # Protection des routes
├── scripts/
│   ├── run_pipeline.py     # Pipeline CLI
│   ├── eval_golden.py      # Évaluation BLEU/METEOR/WER sur dataset golden
│   ├── fetch_flashes.py    # Téléchargement manuel des flashs
│   └── langfuse_import.py  # Import des 84 runs dans Langfuse
├── outputs/
│   └── experiments/
│       ├── results.csv              # 84 runs (métriques complètes)
│       └── evaluation_report.md    # Rapport par modèle/prompt
├── data/
│   ├── flash_audio_archive/         # Archive MP3 trafic
│   └── golden/
│       └── references/              # Références traduction EN + transcription FR (WER)
├── docker-compose.yml
└── pyproject.toml
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
| `JWT_SECRET` | Clé secrète JWT (32+ chars) | Oui |
| `WHISPER_MODEL` | Modèle STT service (`small`, `large-v3`) | Non (défaut: `small`) |
| `LLM_MODEL` | Modèle LiteLLM | Non (défaut: `groq/llama-3.1-8b-instant`) |
| `PROMPT_VERSION` | Version du prompt (`v1.0`–`v1.2`) | Non (défaut: `v1.1`) |
| `DATABASE_URL` | URL base de données | Non (défaut: SQLite) |
| `DEV_MODE` | Endpoints de développement | Non (défaut: `false`) |
| `POLL_INTERVAL_S` | Intervalle polling watcher (secondes) | Non (défaut: `15`) |
| `MAX_EVENTS_PER_ZONE` | Ring buffer watcher | Non (défaut: `4`) |
| `LANGFUSE_PUBLIC_KEY` | Clé publique Langfuse | Non |
| `LANGFUSE_SECRET_KEY` | Clé secrète Langfuse | Non |
| `LANGFUSE_HOST` | URL Langfuse | Non (défaut: cloud.langfuse.com) |

---

## Roadmap

- [x] Phase 1 — Dataset golden + évaluation BLEU (84 runs, 7 audios, 12 combinaisons)
- [x] Phase 1+ — Métriques METEOR et WER ajoutées à l'évaluation
- [x] Phase 2 — Microservices Docker (STT / LLM / TTS)
- [x] Phase 2 — Langfuse tracing (import historique + tracing temps réel)
- [x] Phase 2 — CI GitHub Actions
- [x] Phase 3 — Pipeline Service orchestrateur (Langchain LCEL)
- [x] Phase 3 — Frontend Next.js
- [x] Phase 3 — API Gateway (auth JWT + refresh tokens)
- [x] Phase 3 — Dashboard Admin MLOps (7 onglets)
- [x] Phase 3+ — Watcher trafic temps réel (SSE + ring buffer, admin uniquement)
- [ ] Phase 4 — Prometheus + Grafana (monitoring système)
- [ ] Phase 4 — MLflow model registry
- [ ] Phase 4 — Airflow batch evaluation
- [ ] Déploiement — VPS + nginx + SSL (traduction-audio.fr)

---

## Licence

[LICENSE](LICENSE)
