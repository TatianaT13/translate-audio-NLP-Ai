# Architecture — traudio

Plateforme MLOps de traduction audio en temps réel. **14 microservices** orchestrés via Docker Compose, déployée sur `traduction-audio.fr` (Hetzner VPS).

---

## 1. Vue d'ensemble

```
                        ┌──────────────────┐
                        │   Utilisateur    │
                        │  (browser web)   │
                        └────────┬─────────┘
                                 │ HTTPS + basic-auth
                                 ▼
                        ┌──────────────────┐
                        │      nginx       │  reverse proxy
                        │  (edge, host)    │  Let's Encrypt SSL
                        └────────┬─────────┘
                                 │
                                 ▼  (host 127.0.0.1:3000)
                     ┌────────────────────────┐
                     │  Frontend  (Next.js)   │  standalone, non-root
                     │  rewrites → services   │
                     └───────┬────────────────┘
                             │
              ┌──────────────┼──────────────────────┐
              │              │                      │
              ▼              ▼                      ▼
      ┌─────────────┐  ┌─────────────┐      ┌─────────────┐
      │  Gateway    │  │  Pipeline   │      │  Watcher    │
      │  :8004      │  │  :8000      │      │  :8005      │
      │  Auth JWT   │  │  LangGraph  │      │  Streaming  │
      │  Admin API  │  │  LCEL       │      │  temps réel │
      └──────┬──────┘  └──┬──┬──┬────┘      └──────┬──────┘
             │            │  │  │                  │
             │            ▼  ▼  ▼                  │
             │      ┌─────────────────────┐        │
             │      │  STT     LLM    TTS │        │
             │      │  :8001  :8002  :8003│        │
             │      │  Whisper Groq   Vox │        │
             │      │  large-v3 OpenAI MMS│        │
             │      └─────────────────────┘        │
             │                                     │
             ▼                                     ▼
      ┌──────────────────────────────────────────────────┐
      │  Observability                                   │
      │  MLflow · Prometheus · Grafana · Langfuse        │
      │  :5050    :9090        :3001      cloud          │
      └──────────────────────────────────────────────────┘

      ┌──────────────────────────────────────────────────┐
      │  Batch                                           │
      │  Airflow (scheduler + webserver + PostgreSQL)    │
      │  :8080                                           │
      └──────────────────────────────────────────────────┘
```

---

## 2. Services

### Couche présentation

| Service | Port | Stack | Rôle |
|---------|------|-------|------|
| **Frontend** | 3000 | Next.js 15 standalone | UI (traduction, meeting, admin, support de soutenance) |

Rewrites Next.js internes (`next.config.ts`) pour proxy `/api`, `/pipeline`, `/stt`, `/llm` → services Docker.

### Couche métier

| Service | Port | Stack | Rôle |
|---------|------|-------|------|
| **Gateway** | 8004 | FastAPI + SQLAlchemy | Auth JWT (15 min + 7j refresh rotatif), admin API, agrégation MLflow/Airflow/Langfuse |
| **Pipeline** | 8000 | FastAPI + LangChain LCEL | Orchestrateur STT → LLM → TTS, prompt guards, tracing Langfuse+MLflow |
| **Watcher** | 8005 | FastAPI + Faster-Whisper | Polling autoroutes-info.fr, transcription live, extraction d'événements trafic |

### Couche modèles

| Service | Port | Stack | Rôle |
|---------|------|-------|------|
| **STT** | 8001 | Faster-Whisper large-v3 | Speech-to-Text multilingue |
| **LLM** | 8002 | LiteLLM (Groq/OpenAI/Anthropic) | Traduction FR → EN/UK/ES/DE |
| **TTS** | 8003 | Mistral Voxtral + MMS-TTS (Meta) | Synthèse vocale — routing par langue |

### Couche observability & data

| Service | Port | Stack | Rôle |
|---------|------|-------|------|
| **MLflow** | 5050 | MLflow 2.18+ | Tracking runs, Registry, `mlflow.evaluate()`, tracing |
| **Prometheus** | 9090 | prometheus + fastapi-instrumentator | Métriques HTTP + métier |
| **Grafana** | 3001 | Grafana 11 | 12 panels dashboard "LLMOps Overview" |
| **Langfuse** | cloud | Langfuse | Traces LLM détaillées (spans hiérarchiques) |
| **Airflow** | 8080 | Airflow 2.10 + PostgreSQL | Orchestration batch (nightly_golden_eval) |

---

## 3. Flux de données

### 3.1 Authentification

```
Browser ──POST /auth/register──► Gateway ──► SQLite (users, bcrypt hash)
Browser ──POST /auth/login──► Gateway ──► JWT HS256 (15 min) + refresh (7j SHA256 stocké)
Browser ──POST /auth/refresh──► Gateway ──► JWT rotate (ancien refresh révoqué)
```

**Sécurité** : JWT signé HS256, refresh hashé en DB (raw jamais stocké), rotation obligatoire, cookie httpOnly côté client.

### 3.2 Traduction audio (flow principal)

```
Browser ──POST /pipeline/process (audio.mp3)──► Pipeline
                                                     │
                                                     ▼
                                            _stt_step (LangChain)
                                                     │ HTTP POST
                                                     ▼
                                              STT :8001 (Whisper)
                                                     │ text FR + language_prob
                                                     ▼
                                            prompt_guard.check_input()
                                                     │ (bloque injections)
                                                     ▼
                                            _llm_step (LangChain)
                                                     │ HTTP POST
                                                     ▼
                                              LLM :8002 (LiteLLM)
                                                     │ text EN + tokens + cost
                                                     ▼
                                            prompt_guard.check_output()
                                                     │ (anti-hallucination)
                                                     ▼
                                            _tts_step (LangChain)
                                                     │ HTTP POST
                                                     ▼
                                              TTS :8003 (Voxtral / MMS)
                                                     │ audio_b64 + content_type
                                                     ▼
Browser ◄──JSON { source_text, translation, audio_b64, latencies, cost }
```

**Prompt guards** (défense en profondeur) :
1. **Pre-check** — regex anti-injection sur le texte transcrit (FR + EN + jailbreak)
2. **Sandbox** — `<user_text>...</user_text>` échappe les balises injectées
3. **Post-check** — ratio longueur (anti-hallucination) + détection prompt leak

### 3.3 Watcher — trafic live

```
Toutes les 5 min (POLL_INTERVAL_S=300) :
   Pour chaque zone (nord, sud, est, ouest) :
       poll autoroutes-info.fr (ETag conditional)
       si nouveau flash :
           STT → text FR
           LLM → 4 langues (EN, UK, ES, DE)
           extract_events (regex) → sévérité + type + route
           store dans /app/state/ (ring buffer 10 events/zone)
           SSE broadcast → dashboard admin
```

### 3.4 Batch — évaluation golden

```
Airflow DAG "nightly_golden_eval" (cron 02:00 UTC) :
   translate_one (audio golden × config) → results.csv
   aggregate → moyennes par (whisper, llm, prompt)
   check_drift → alerte si BLEU chute > X%
```

---

## 4. Sécurité — couches transversales

| Couche | Mécanisme |
|--------|-----------|
| **Réseau** | Services bindés sur `127.0.0.1` uniquement. Seul nginx expose le 443 public |
| **HTTPS** | Let's Encrypt auto-renouvelé (certbot) |
| **Auth edge** | basic-auth nginx (fallback) |
| **Auth applicative** | JWT 15 min + refresh rotatif 7j (bcrypt password hash) |
| **Prompt injection** | 3 couches : pre-check regex, sandbox `<user_text>`, post-check output |
| **Isolation** | Container non-root (uid 1000), reseau Docker interne isolé |
| **Secrets** | `.env` gitignored, clés API par provider, JWT_SECRET 32 chars |

---

## 5. Observability — trois couches

### 5.1 Infrastructure (Prometheus + Grafana)
- 6 services FastAPI exposent `/metrics` via `prometheus-fastapi-instrumentator`
- Métriques : `http_requests_total`, `http_request_duration_seconds` (histogramme p95)
- Dashboard Grafana "LLMOps Overview" : 12 panels (req/s, latence p95, taux d'erreur)

### 5.2 Métier (Prometheus custom counters)
- `watcher_polls_total{zone, status}` — activité polling
- `watcher_events_extracted_total{severity, type}` — événements trafic captés
- `watcher_translation_cost_usd_total{lang}` — coût cumulé par langue
- `watcher_translation_tokens_total{lang}` — tokens consommés

### 5.3 LLM applicative (MLflow + Langfuse)
- **MLflow** — expérimentation (runs, params, metrics, `evaluate()`, tracing)
- **Langfuse** — traces LLM détaillées (spans hiérarchiques, coût par trace)

---

## 6. Choix techniques — décisions clés

| Décision | Choix | Alternative | Rationale |
|----------|-------|-------------|-----------|
| Orchestration LLM | LangChain LCEL | LangGraph, custom | Chaînage typé `|`, standard industrie |
| Abstraction LLM | LiteLLM | SDK direct | Provider-agnostic (Groq/OpenAI/Anthropic sans code change) |
| Provider LLM prod | OpenAI GPT-4o-mini | Groq (rate limit) | Pay-as-you-go, pas de plafond dur |
| Container | Docker Compose | Kubernetes | Simplicité, K8s overkill à cette échelle |
| Reverse proxy | Nginx | Traefik/Caddy | Mature, config lisible, Let's Encrypt intégré |
| Auth | JWT custom | Auth0/Keycloak | Contrôle total, zéro dépendance externe |
| Batch | Airflow | Prefect/Dagster | Standard entreprise, compétence transférable |
| Tracking ML | MLflow | W&B, ClearML | Open-source, self-hosted, gratuit |
| TTS ukrainien | MMS-TTS local (Meta) | Voxtral | Voxtral n'est pas entraîné en UK → qualité dégradée |

---

## 7. Structure du repo

```
translate-audio-NLP-Ai/
├── backend/services/
│   ├── gateway/          # Auth JWT + admin API
│   ├── pipeline/         # Orchestrateur LCEL
│   ├── stt/              # Faster-Whisper
│   ├── llm/              # LiteLLM
│   ├── tts/              # Voxtral + MMS
│   └── watcher/          # Streaming trafic
├── frontend/             # Next.js 15 standalone
│   ├── app/              # Pages (page.tsx, meeting, admin, login...)
│   ├── lib/              # Client API + auth
│   └── public/           # Assets + soutenance.html
├── scripts/
│   ├── eval_golden.py         # Benchmark 12 configs
│   ├── mlflow_register.py     # Log runs + evaluate() + registry
│   └── import_metrics_to_langfuse.py
├── monitoring/
│   ├── grafana/dashboards/    # microservices.json (12 panels)
│   └── prometheus.yml
├── airflow/dags/              # nightly_golden_eval.py
├── tests/                     # unit + integration (103 tests, CI GitHub)
├── data/golden/               # Audios de référence + traductions
├── docker-compose.yml
└── .env                       # Secrets (gitignored)
```

---

## 8. Roadmap Phase 2

1. **Refactor sécurité** — tout par `/pipeline` (ne plus exposer `/stt` et `/llm` publiquement)
2. **Traduction bidirectionnelle** — français en cible + auto-détection langue source via Whisper
3. **Tests d'intégration en CI** — job Docker Compose dans GitHub Actions
4. **CD automatisé** — push main → deploy hermes via SSH + rebuild
5. **Playwright E2E** — tests bout-en-bout via UI
6. **Fine-tuning prompts** — dataset golden élargi, iteration continue
