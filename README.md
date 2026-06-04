# traduction-audio — LLMOps Audio Translation Pipeline

Système LLMOps de traduction audio temps réel : **Audio FR → Transcription → Traduction EN/UK/ES/DE → Synthèse vocale**.

Architecture microservices avec orchestration Langchain LCEL, authentification JWT, tracing Langfuse end-to-end, MLflow Model Registry + 12 configurations comparées, **Airflow batch (2 DAGs : nightly eval + weekly drift)**, monitoring Prometheus+Grafana, garde-fou prompt injection 3 couches, monitoring trafic autoroutier temps réel.

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
| Frontend | 3000 | Next.js 15 (standalone, conteneurisé, non-root) |
| Pipeline (orchestrateur) | 8000 | FastAPI + Langchain LCEL + Langfuse |
| STT | 8001 | FastAPI + Faster-Whisper large-v3 |
| LLM | 8002 | FastAPI + LiteLLM + Groq |
| TTS | 8003 | FastAPI + Mistral Voxtral |
| Gateway (auth + admin) | 8004 | FastAPI + SQLAlchemy + JWT (15min) + refresh (7j) |
| Watcher (trafic live SSE) | 8005 | FastAPI + Whisper + extraction events |
| **Prometheus** | 9090 | Scrape `/metrics` toutes les 15s, rétention 30j |
| **Grafana** | 3001 | Dashboard "LLMOps Overview" préconfiguré |
| **MLflow** | 5050 | Model Registry + Experiment Tracking |
| **Airflow** | 8080 | Orchestration batch (DAGs nightly eval + weekly drift) |
| **Postgres (Airflow)** | — | Metadata DB d'Airflow |

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

**Une seule commande** lance tout (frontend + 6 services backend + Prometheus + Grafana + MLflow) :

```bash
docker compose up --build
```

**Apps & APIs** :
- **Frontend** : http://localhost:3000
- Dashboard admin MLOps : http://localhost:3000/admin
- Gateway API : http://localhost:8004/docs
- Pipeline API : http://localhost:8000/docs
- STT / LLM / TTS / Watcher : http://localhost:8001-8005/docs

**Observabilité & Registres** :
- **Grafana** (monitoring système) : http://localhost:3001
- **Prometheus** (métriques brutes) : http://localhost:9090
- **MLflow** (model registry + 12 expériences agrégées) : http://localhost:5050
- **Airflow** (2 DAGs : nightly eval + weekly drift) : http://localhost:8080 (admin / admin)

> Le frontend attend que le gateway et le pipeline soient `healthy` avant de démarrer (`depends_on: condition: service_healthy`).

### Créer le premier compte admin

```bash
# 1. S'inscrire sur http://localhost:3000/register
# 2. Promouvoir en admin
curl -X POST http://localhost:8004/admin/seed
# 3. Accéder au dashboard : http://localhost:3000/admin
```

---

## Authentification (Gateway JWT)

Le service Gateway (`backend/services/gateway/`) gère toute l'authentification :

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
| Vue générale | Stats utilisateurs, KPIs pipeline (runs, latences, BLEU/METEOR/WER, **coût total $, coût/run, tokens**) |
| Traces & Modèles | Latences STT/LLM/TTS, histogrammes BLEU/METEOR/WER/confiance, tableau comparatif modèles avec colonne **Coût** triable |
| Trafic Live | Incidents autoroutiers temps réel par zone (Nord/Sud/Ouest) via SSE — toggle "Tous / Urgences uniquement", usage interne admin |
| **Expériences** | **MLflow natif via API REST** — 12 configurations + champion + 3 modèles registry (cartes intégrées au design) |
| **Infrastructure** | Health checks temps réel des 6 microservices + **Grafana embedded** (req/s, p95, erreurs) |
| **Pipelines** | **Airflow natif via API REST** — liste des DAGs réels avec état, schedule, tags, dernier run |
| Utilisateurs | CRUD complet : activer, promouvoir admin, supprimer |

---

## Watcher — Trafic Live

Service dédié (`backend/services/watcher/`) qui tourne en arrière-plan en permanence :

- Poll adaptatif toutes les **~15s** sur 3 flux autorouteinfo.fr (nord / sud / ouest)
- Requêtes conditionnelles ETag/Last-Modified — zéro bande passante si pas de changement
- **STT Whisper small** en mémoire (`mem_limit: 2g`) — fichiers audio supprimés immédiatement après transcription
- Extraction d'événements trafic par regex (`event_extractor.py`) : accident, bouchon, animal, fermeture, intempéries, travaux, véhicule en panne
- **Tous les niveaux de sévérité conservés** (`high` / `medium` / `low`) — le filtre est côté UI
- **Ring buffer `deque(maxlen=10)`** par zone — persisté sur disque (`/app/state`)
- **SSE `/stream`** → dashboard admin mis à jour en temps réel
- **Toggle UI** : "Tous les flashs" vs "Urgences uniquement" (filtrage `high`)

> Usage strictement interne (admin uniquement). L'app publique ne redistribue pas ce contenu.

---

## Évaluation (Phase 1)

**12 configurations** testées : 2 modèles Whisper × 2 LLMs × 3 prompts, évaluées sur 7 fichiers audio golden = 36 lignes de résultats agrégées.

**Métriques calculées** : BLEU (sacrebleu), METEOR (nltk), WER (jiwer — nécessite refs FR)

| Rang | Whisper | LLM | Prompt | BLEU moy. | METEOR moy. |
|------|---------|-----|--------|-----------|-------------|
| 🏆 | large-v3 | llama-3.3-70b | v1.1 | **49.64** | **0.713** |
| 2 | large-v3 | llama-3.3-70b | v1.0 | 47.01 | 0.697 |
| 3 | large-v3 | llama-3.1-8b | v1.0 | 46.25 | 0.638 |
| 4 | large-v3 | llama-3.3-70b | v1.2 | 41.67 | 0.615 |
| 5 | large-v3 | llama-3.1-8b | v1.1 | 41.15 | 0.636 |

**Combinaison déployée** : `large-v3 + llama-3.1-8b-instant + prompt v1.1` (compromis qualité / vitesse / coût)
**Champion qualité** (tagué dans MLflow) : `large-v3 + llama-3.3-70b-versatile + v1.1`

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

## Observabilité & Registres

### Langfuse (tracing métier LLM)
- Chaque traduction crée 1 `trace` + 1 `generation` (avec model + usage tokens + cost)
- Scores : latences (STT/LLM/TTS), confiance langue, **coût $**, total tokens, BLEU, METEOR, WER
- Dashboards natifs Langfuse remplis automatiquement (Cost, Tokens)
```bash
python scripts/langfuse_import.py    # Importer les 84 runs historiques
```

### MLflow (model registry + experiment tracking)
- **12 runs** dans l'expérience `translate-audio-llmops` — 1 par configuration unique (`whisper × llm × prompt`)
- Métriques **agrégées** sur les 7 audios golden (`bleu_mean`, `meteor_mean`, `wer_mean`, latences)
- **Champion automatique** : tag `champion=true` + `stage=production` sur la meilleure config (BLEU max)
- Texte du prompt attaché en tag `prompt_text` (audit + reproductibilité)
- **3 modèles registry** : `whisper-stt` · `llama-translation` · `voxtral-tts` (provider, version, type)
- **Affichage natif** dans l'onglet Expériences du dashboard (API REST MLflow, pas d'iframe)
- UI complète : http://localhost:5050
```bash
python scripts/mlflow_register.py   # (Re-)importer les configs + register models
```

### Prometheus + Grafana (monitoring système + business)
- Prometheus scrape `/metrics` toutes les 15s sur les 6 services (instrumenté via `prometheus-fastapi-instrumentator`)
- Grafana : dashboard "LLMOps Overview" préconfiguré (2 rows)
  - **Microservices** : req/s · latence p95 · taux erreur 5xx · services up · req/min · % erreurs
  - **Watcher Live** : polls/min par zone · events extraits/h par sévérité · coût LLM total · tokens total
- Métriques business custom watcher : `watcher_polls_total`, `watcher_events_extracted_total`,
  `watcher_translation_cost_usd_total`, `watcher_translation_tokens_total`
- Provisioning : `monitoring/grafana/provisioning/` (datasource + dashboard JSON)

### Airflow (orchestration batch)
- Stack : `airflow-postgres` + `airflow-init` + `airflow-webserver` + `airflow-scheduler` (LocalExecutor)
- UI : http://localhost:8080 (admin / admin)
- **2 DAGs** dans [airflow/dags/](airflow/dags/) :
  - `nightly_golden_eval` — `0 2 * * *` (2h tous les jours) : ping pipeline → relance les 7 audios golden → agrège succès/échecs/latences/coût → alerte si trop d'échecs
  - `weekly_drift_check` — `0 3 * * 0` (dimanche 3h) : interroge Langfuse semaine N vs N-1 → alerte si dégradation > 10% sur latence/coût/BLEU
- Endpoint gateway `/admin/airflow/summary` → onglet **Pipelines** du dashboard affiche les DAGs natifs (état réel)

### Sécurité — anti-prompt-injection (3 couches)
- **Pre-check** ([prompt_guard.py](backend/services/pipeline/src/pipeline/prompt_guard.py)) : regex sur la transcription (FR + EN) → 422 si tentative
- **Sandbox prompt** : texte utilisateur encadré dans `<user_text>…</user_text>` avec règles strictes
- **Post-check** : détection prompt leak (`"I am an AI"`, `"system prompt"`) + anti-hallucination (ratio output/input)

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
├── frontend/                       # Next.js (conteneurisé, port 3000)
│   ├── Dockerfile                  # Multi-stage : deps → build → runtime alpine
│   ├── .dockerignore
│   ├── app/
│   │   ├── page.tsx                # Page principale + UserMenu
│   │   ├── login/  register/  forgot-password/  reset-password/
│   │   └── admin/                  # Dashboard MLOps admin
│   ├── lib/
│   │   ├── auth.ts                 # Client auth
│   │   ├── api.ts                  # Client pipeline
│   │   └── admin.ts                # Client admin API + SSE trafic
│   └── middleware.ts               # Protection des routes
│
├── backend/
│   └── services/                   # 6 microservices Python (FastAPI)
│       ├── gateway/                # Auth JWT + admin API (port 8004)
│       │   ├── Dockerfile          # Multi-stage uv + non-root
│       │   ├── .dockerignore
│       │   ├── pyproject.toml      # Deps isolées par service
│       │   └── src/gateway/
│       │       ├── main.py         # Routes auth + admin + Langfuse + trafic proxy
│       │       ├── auth.py         # JWT + bcrypt + refresh tokens
│       │       ├── models.py       # SQLAlchemy User
│       │       ├── schemas.py      # Pydantic
│       │       └── database.py     # SQLite/PostgreSQL
│       ├── pipeline/               # Orchestrateur Langchain LCEL (port 8000)
│       ├── stt/                    # Faster-Whisper (port 8001)
│       ├── llm/                    # LiteLLM → Groq/OpenAI/Anthropic (port 8002)
│       ├── tts/                    # Mistral Voxtral + MMS-TTS (port 8003)
│       └── watcher/                # Trafic Live SSE (port 8005)
│
├── src/flash_nlp/                  # Lib partagée pour les scripts CLI
│   ├── acquisition/  transcription/  analysis/  io/
│
├── scripts/                        # Scripts CLI (eval, import, batch)
├── outputs/experiments/            # results.csv (84 runs) + rapport
├── data/                           # Datasets golden, archives audio
│
├── docker-compose.yml              # 1 commande lance tout (front + back)
└── pyproject.toml                  # Lib racine flash_nlp (scripts CLI)
```

### Conventions backend

Chaque service Python suit la même structure pro :

- `pyproject.toml` : dépendances **isolées** par service (pas de monorepo `requirements.txt` partagé)
- **Dockerfile multi-stage** :
  - **Stage 1 (builder)** : `uv` (10× plus rapide que pip) installe les deps dans `/opt/venv`
  - **Stage 2 (runtime)** : image minimale, **utilisateur non-root** (`USER app`, UID 1000), copie uniquement le venv + le code source
- `HEALTHCHECK` intégré
- `.dockerignore` strict (pas de `node_modules`, `.venv`, `__pycache__`, `.git`)

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
| `WHISPER_MODEL` | Modèle Whisper du STT service (`small`, `medium`, `large-v3`) | Non (défaut: `small`) |
| `WATCHER_WHISPER_MODEL` | Modèle Whisper du watcher (séparé du STT) | Non (défaut: `small` — `large-v3` cause OOM) |
| `LLM_MODEL` | Modèle LiteLLM | Non (défaut: `groq/llama-3.1-8b-instant`) |
| `PROMPT_VERSION` | Version du prompt (`v1.0`–`v1.2`) | Non (défaut: `v1.1`) |
| `DATABASE_URL` | URL base de données | Non (défaut: SQLite) |
| `DEV_MODE` | Endpoints de développement (ex: `/admin/seed`, mots de passe reset retournés en clair) | Non (défaut: `false`) |
| `POLL_INTERVAL_S` | Intervalle polling watcher (secondes) | Non (défaut: `15`) |
| `MAX_EVENTS_PER_ZONE` | Ring buffer watcher | Non (défaut: `10`) |
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
- [x] Phase 3+ — Garde-fou prompt injection 3 couches (OWASP LLM01)
- [x] Phase 3+ — Suivi des coûts LLM (Langfuse generations + dashboard $/run/total)
- [x] Phase 4 — **Prometheus + Grafana** (monitoring système + business watcher)
- [x] Phase 4 — **MLflow** model registry (3 modèles) + experiment tracking (12 configurations + champion)
- [x] Phase 4 — **Airflow** batch evaluation (2 DAGs : nightly_golden_eval + weekly_drift_check)
- [x] Phase 4 — **Watcher tracing E2E** (Langfuse traces + Prometheus custom metrics)
- [x] Architecture pro — backend/services/<svc>/ avec multi-stage uv + non-root + healthchecks
- [x] Frontend conteneurisé Next.js standalone (1 commande lance tout)
- [ ] Phase 4 — Evidently drift detection (alternative à `weekly_drift_check`)
- [ ] Phase 2 — MinIO (storage S3-like pour audio)
- [ ] Phase 3 — Rate limiting Gateway
- [ ] Déploiement — VPS + nginx + SSL (traduction-audio.fr)

---

## Licence

[LICENSE](LICENSE)
