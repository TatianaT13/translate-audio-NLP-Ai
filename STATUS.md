# État du projet — actualisé 1er septembre 2026

> Projet LLMOps · Traduction audio temps réel (FR → EN/UK/ES/DE/IT)
> Soutenance : **3 septembre 2026**
>
> Note : ce fichier consigne les jalons projet. Pour les chiffres à jour de la
> campagne d'évaluation (BLEU, METEOR, protocole), voir le README et le rapport
> technique.

---

## ✅ Ce qui est en place

### Phase 1 — Fondations & Prompt Engineering
- [x] Dataset golden (7 audios de référence)
- [x] **12 configurations testées** : 2 Whisper × 2 LLMs × 3 prompts, chacune évaluée sur les 7 audios golden
- [x] **84 évaluations individuelles** (importées dans Langfuse) → **12 runs agrégés dans MLflow** (1 par configuration, métriques moyennées)
- [x] Métriques **BLEU (sacrebleu) / METEOR (nltk) / WER (jiwer) / latences / coût**
- [x] **Champion expérimental** : `whisper large-v3 + llama-3.3-70b + prompt v1.1` (BLEU 49.64 · METEOR 0.713)
- [x] Configuration historiquement retenue en prod : `llama 8B + v1.1` (compromis coût/qualité)
- [x] Prod actuelle (post-dépréciation Groq) : `openai/gpt-4o-mini` via LiteLLM — nouvelle campagne comparative prévue

### Phase 2 — Microservices & Registres
- [x] **STT Service** (port 8001) — Faster-Whisper
- [x] **LLM Service** (port 8002) — LiteLLM + Groq
- [x] **TTS Service** (port 8003) — Mistral Voxtral
- [x] **Langfuse** — prompts + traces + scores + cost
- [x] **MLflow** (port 5050) — Model Registry (3 modèles) + 12 runs agrégés (1 par configuration)
- [ ] MinIO (storage S3-like) — pas encore

### Phase 3 — Orchestration & Gateway
- [x] **Pipeline Service** (port 8000) — FastAPI + Langchain LCEL (STT → LLM → TTS)
- [x] **Gateway Service** (port 8004) — FastAPI custom (méthodo OK : Kong/Nginx/Traefik **ou FastAPI**)
- [x] **Auth JWT** complète : register, login, logout, refresh tokens rotatifs (7j), bcrypt
- [x] Mot de passe oublié + reset + suppression compte
- [ ] Rate limiting (à faire — 2h)

### Phase 4 — Monitoring & Évaluation Batch
- [x] **Prometheus** (port 9090) — scrape `/metrics` toutes les 15s sur les 6 services
- [x] **Grafana** (port 3001) — dashboard "LLMOps Overview" préconfiguré
  - Requêtes/s par service · Latence p95 · Taux erreur 5xx · Services up
- [x] **Langfuse** — monitoring métier (coût $, tokens, version prompt, BLEU)
- [x] **Dashboard Admin MLOps** custom 7 onglets
- [ ] Airflow batch evaluation (à décider — 1-2 jours)
- [ ] Evidently drift detection (à faire — 4h)
- [x] **GitHub Actions CI** sur chaque push

### Frontend
- [x] **Next.js conteneurisé** (multi-stage standalone, alpine, non-root) — méthodo demandait Streamlit (mieux !)
- [x] **Une seule commande** lance tout : `docker compose up --build`
- [x] Page principale : upload, enregistrement micro, démo, ticker proverbes
- [x] Dashboard admin avec 7 onglets (Vue / Traces / Trafic / Expériences / Infra / Pipelines / Utilisateurs)
- [x] Pages auth (login, register, forgot-password, reset-password)

### Sécurité (tips mentor explicites)
- [x] **Suivi des coûts LLM** dans Langfuse + dashboard ($ par run, total, tokens)
- [x] **Validation entrée/sortie** : taille audio (25 Mo), longueur transcription, ratio output/input
- [x] **Garde-fou prompt injection via audio** — 3 couches OWASP LLM01
  - Pre-check regex (FR + EN : "ignore les instructions", "you are now"…)
  - Sandbox prompt LLM (`<user_text>…</user_text>` avec règles strictes)
  - Post-check sortie (détection prompt leak, anti-hallucination via ratio longueur)
- [x] Conteneurs **non-root** (UID 1000) avec `pyproject.toml` isolés par service
- [x] **Multi-stage builds** avec `uv` (10× plus rapide que pip)
- [x] Healthchecks intégrés sur chaque conteneur

### Watcher Trafic Live (bonus hors-méthodo)
- [x] Service polling autorouteinfo.fr toutes les 15s sur 3 zones (nord/sud/ouest)
- [x] STT Whisper + extraction d'événements + traduction batch EN/UK/ES
- [x] **Streaming SSE** vers dashboard admin
- [x] Fusion automatique des events sur même portion (1 carte, plusieurs badges)
- [x] Filtre "Tous / Urgences uniquement"

---

## 🗺️ Architecture

```
                            ┌─────────────────────────┐
                            │      CLIENT (Browser)   │
                            │  http://localhost:3000  │
                            └────────────┬────────────┘
                                         │ HTTPS / REST
                                         ▼
                            ┌─────────────────────────┐
                            │   FRONTEND (Next.js)    │
                            │   Conteneurisé · :3000  │
                            └────────────┬────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────────┐
              │                          │                              │
              ▼                          ▼                              ▼
   ┌──────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
   │  GATEWAY  :8004  │      │   PIPELINE   :8000   │      │  WATCHER  :8005  │
   │  FastAPI         │      │   FastAPI            │      │  Polling trafic  │
   │  • JWT auth      │      │   + Langchain LCEL   │      │  + STT + SSE     │
   │  • Admin API     │      │   + Langfuse trace   │      │                  │
   │  • Proxy watcher │      └──┬────────┬────────┬─┘      └────────┬─────────┘
   └──────────────────┘         │        │        │                 │
                                ▼        ▼        ▼                 │
                       ┌────────┐ ┌────────┐ ┌────────┐             │
                       │  STT   │ │  LLM   │ │  TTS   │◄────────────┘
                       │ :8001  │ │ :8002  │ │ :8003  │
                       │ Whisper│ │LiteLLM │ │Mistral │
                       │        │ │ +Groq  │ │Voxtral │
                       └────────┘ └────────┘ └────────┘

  ──────────────────────── REGISTRES & MONITORING ────────────────────────

  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │   LANGFUSE   │  │    MLFLOW    │  │  PROMETHEUS  │  │   GRAFANA    │
  │   (cloud)    │  │   :5050      │  │    :9090     │  │    :3001     │
  │              │  │              │  │              │  │              │
  │ • Prompts    │  │ • 12 runs    │  │ • Scrape     │  │ • Dashboard  │
  │ • Traces     │  │ • 3 modèles  │  │   /metrics   │  │   LLMOps     │
  │ • Coût $     │  │   registry   │  │   /15s       │  │   live       │
  │ • Tokens     │  │              │  │ • 30j data   │  │ • Iframe     │
  └──────────────┘  └──────────────┘  └──────────────┘  │   admin      │
                                                        └──────────────┘
```

---

## 📊 Stack technique

| Couche | Technologies |
|--------|--------------|
| **Frontend** | Next.js 15 · React 19 · TypeScript · Tailwind |
| **Backend services** | FastAPI · Python 3.11 · uv · multi-stage Docker · non-root |
| **Orchestration LLM** | Langchain LCEL · LiteLLM (proxy multi-provider) |
| **Modèles** | Faster-Whisper (large-v3) · Llama 3.1 8B (Groq) · Mistral Voxtral |
| **Auth & sécurité** | JWT (15min) + refresh rotatif (7j) · bcrypt · Anti-prompt-injection 3 couches |
| **Tracking** | Langfuse (cloud) · MLflow (self-hosted) |
| **Monitoring** | Prometheus + Grafana (self-hosted, dashboards préconfigurés) |
| **DB** | SQLite (auth) — PostgreSQL ready (prod) |
| **CI** | GitHub Actions (pytest sur chaque push) |
| **Orchestration containers** | Docker Compose (10 services en 1 commande) |

---

## 🔗 URLs locales

| Service | URL | Quoi |
|---------|-----|------|
| App principale | http://localhost:3000 | Frontend public + admin |
| Dashboard admin | http://localhost:3000/admin | 7 onglets MLOps |
| Gateway API | http://localhost:8004/docs | Swagger auth + admin |
| Pipeline API | http://localhost:8000/docs | Swagger pipeline |
| **Grafana** | http://localhost:3001 | Monitoring système |
| **MLflow** | http://localhost:5050 | Model Registry |
| **Prometheus** | http://localhost:9090 | Métriques brutes |

---

## ⏳ Reste à faire (3 mois)

| Item | Effort | Priorité |
|------|--------|----------|
| **Airflow** (1 DAG nightly_eval) ou cron alternatif | 1-2 jours | Haute |
| **MinIO** stockage S3-like | 3h | Moyenne |
| **Rate limiting** Gateway | 2h | Moyenne |
| **Evidently** drift detection | 4h | Moyenne |
| **Slides soutenance** + démo scénarisée | — | Haute |
| Tests à jour après refactor | 1h | Moyenne |

---

## 🚀 Commande unique

```bash
docker compose up --build
```

→ Lance les **10 services** (frontend + 6 microservices backend + Prometheus + Grafana + MLflow).

Pour créer le premier admin (DEV_MODE) :
```bash
# 1. Register sur http://localhost:3000/register
# 2. Promouvoir
curl -X POST http://localhost:8004/admin/seed
# 3. Login → menu utilisateur → "Dashboard admin"
```
