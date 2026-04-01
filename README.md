# traduction-audio — LLMOps Audio Translation Pipeline

Système LLMOps de traduction audio temps réel : **Audio FR → Transcription → Traduction EN/UK/ES/DE → Synthèse vocale**.

Architecture microservices avec orchestration Langchain LCEL, authentification JWT, tracing Langfuse, évaluation BLEU sur 84 runs.

---

## Architecture

```
Client (Next.js)
       │
       │ POST /auth/login  │  POST /process (audio)
       ▼
┌─────────────────────────┐
│  Gateway Service  :8004 │   Auth JWT + rate limiting
│  FastAPI + SQLAlchemy   │
└──────────┬──────────────┘
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
              + Groq       Voxtral
```

| Service | Port | Technologie |
|---------|------|-------------|
| Gateway (auth + admin) | 8004 | FastAPI + SQLAlchemy + JWT |
| Pipeline (orchestrateur) | 8000 | FastAPI + Langchain LCEL + Langfuse |
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
# LANGFUSE_PUBLIC_KEY=pk-lf-...
# LANGFUSE_SECRET_KEY=sk-lf-...
```

### Lancement avec Docker

```bash
docker compose up --build
```

Les 5 services démarrent :
- Gateway : http://localhost:8004/docs
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

## Authentification (Gateway JWT)

Le service Gateway (`services/gateway/`) gère toute l'authentification :

- **Register / Login** : création de compte + session cookie
- **JWT access token** (15 min) + **refresh token** rotatif (7 jours, hashé en base)
- **Mot de passe oublié** : flux reset par lien (DEV_MODE retourne l'URL directement)
- **Suppression de compte** et changement de mot de passe
- **Protection des routes** : middleware Next.js (cookie-based), redirection auto vers `/login`

Pages frontend : `/login`, `/register`, `/forgot-password`, `/reset-password`

---

## Dashboard Admin MLOps

Accessible à `/admin` pour les utilisateurs avec le rôle `is_admin`.

Le premier utilisateur peut être promu admin via l'endpoint DEV_MODE :
```bash
curl -X POST http://localhost:8004/dev/promote-first-user
```

### Onglets du dashboard

| Onglet | Contenu |
|--------|---------|
| Vue générale | Stats utilisateurs, KPIs pipeline (runs, latence moyenne) |
| Traces & Modèles | Latences STT/LLM/TTS, histogrammes BLEU/confiance, comparaison modèles |
| Expériences | Stub MLflow (roadmap Phase 4) |
| Infrastructure | Health checks temps réel sur chaque microservice + stub Grafana |
| Pipelines | DAG cards + stub Airflow (roadmap Phase 4) |
| Utilisateurs | CRUD complet : activer, promouvoir admin, supprimer |

Les métriques Langfuse (latences, BLEU, comparaison modèles) sont récupérées depuis l'API Langfuse cloud et exposées par le Gateway sur `/admin/langfuse-metrics`.

---

## Structure

```
.
├── services/
│   ├── gateway/            # Auth JWT + admin API (port 8004)
│   │   ├── main.py         # Routes auth + admin + Langfuse metrics
│   │   ├── auth.py         # JWT, bcrypt, refresh tokens
│   │   ├── models.py       # User SQLAlchemy (is_admin, refresh_token_hash)
│   │   ├── schemas.py      # Pydantic schemas
│   │   ├── database.py     # SQLite/PostgreSQL engine
│   │   └── Dockerfile
│   ├── pipeline/           # Orchestrateur Langchain LCEL + Langfuse (port 8000)
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
│   ├── app/
│   │   ├── page.tsx        # Page principale + UserMenu
│   │   ├── login/          # Connexion
│   │   ├── register/       # Inscription
│   │   ├── forgot-password/
│   │   ├── reset-password/
│   │   └── admin/          # Dashboard MLOps admin (6 onglets)
│   ├── lib/auth.ts         # Client auth (login, register, refresh...)
│   ├── lib/admin.ts        # Client API admin + health checks
│   └── middleware.ts       # Protection des routes
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

Toutes les exécutions du pipeline sont tracées dans [Langfuse](https://cloud.langfuse.com) :
- Latences STT / LLM / TTS par run
- Score BLEU (quand référence disponible)
- Version de prompt utilisée
- Visible dans le Dashboard Admin → onglet **Traces & Modèles**

```bash
# Importer les 84 runs historiques dans Langfuse
python scripts/langfuse_import.py
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
| `JWT_SECRET_KEY` | Clé secrète pour signer les JWT | Oui (Gateway) |
| `DATABASE_URL` | URL base de données (défaut: SQLite) | Non |
| `DEV_MODE` | Active les endpoints de dev (`true`/`false`) | Non |
| `LANGFUSE_PUBLIC_KEY` | Clé publique Langfuse | Non |
| `LANGFUSE_SECRET_KEY` | Clé secrète Langfuse | Non |
| `LANGFUSE_HOST` | URL Langfuse | Non |

---

## Roadmap

- [x] Phase 1 — Dataset golden + évaluation BLEU (84 runs)
- [x] Phase 2 — Microservices Docker (STT / LLM / TTS)
- [x] Phase 2 — Langfuse tracing (import historique + tracing pipeline temps réel)
- [x] Phase 2 — CI GitHub Actions
- [x] Phase 3 — Pipeline Service orchestrateur (Langchain LCEL)
- [x] Phase 3 — Frontend Next.js
- [x] Phase 3 — API Gateway (auth JWT + refresh tokens + rate limiting)
- [x] Phase 3 — Dashboard Admin MLOps (traces, modèles, infra, utilisateurs)
- [ ] Phase 4 — MLflow model registry
- [ ] Phase 4 — Prometheus + Grafana
- [ ] Phase 4 — Airflow batch evaluation

---

## Licence

[LICENSE](LICENSE)
