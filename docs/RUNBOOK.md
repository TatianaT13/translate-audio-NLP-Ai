# Runbook — traudio

Guide opérationnel : déployer, redémarrer, débugger, gérer les incidents fréquents.

---

## 1. Déploiement

### 1.1 Local (Mac/Linux — développement)

```bash
git clone git@github.com:TatianaT13/translate-audio-NLP-Ai.git
cd translate-audio-NLP-Ai
cp .env.example .env       # (créer le fichier si absent)
# Renseigner : OPENAI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, JWT_SECRET, LANGFUSE_*
docker compose up -d --build
```

**Prérequis Docker Desktop** : allouer ≥ **12 Go de RAM** (Settings → Resources → Memory) — Whisper `large-v3` consomme ~3 Go + les 13 autres services.

Accès :
- Frontend : http://localhost:3000
- Grafana : http://localhost:3001 (admin/admin)
- MLflow : http://localhost:5050
- Airflow : http://localhost:8080 (admin/admin)

### 1.2 Production (hermes VPS Hetzner)

**Utiliser le script de déploiement** qui garantit toujours la dernière version :

```bash
ssh tanya@144.76.165.54
cd ~/traudio
./scripts/deploy.sh              # rebuild tous les services
# OU pour un seul service :
./scripts/deploy.sh frontend     # rebuild uniquement le frontend
```

Le script fait, dans l'ordre :
1. `git pull --ff-only` de la dernière version
2. `docker compose build --no-cache` (pas de layer périmé)
3. `docker compose up -d --force-recreate` (recharge `.env` + volumes)
4. Attente healthchecks (timeout 90s)
5. Affichage du commit déployé + suggestions post-deploy

⚠️ **Ne PAS utiliser** `docker compose up -d --build` seul — Docker cache
agressivement le layer `npm run build` de Next.js et peut servir un vieux
bundle. Le script utilise `--no-cache` pour éviter ce piège.

Le déploiement est **manuel** — la CI GitHub Actions lance uniquement les tests
(pas de CD auto). Voir Roadmap Phase 2.

---

## 2. Redémarrer un service

**Redémarrer** (sans rebuild) — préserve l'image, recharge les env vars uniquement si `--force-recreate` :
```bash
docker compose restart <service>
# ou pour recharger les env vars :
docker compose up -d --force-recreate <service>
```

**Rebuild** (nouveau code) — obligatoire après modif du code source :
```bash
docker compose up -d --build <service>
```

**Rebuild sans cache** (obligatoire quand Docker cache trop agressivement) :
```bash
docker compose build --no-cache <service>
docker compose up -d --force-recreate <service>
```

⚠️ **Piège fréquent** : `docker compose up -d --build frontend` peut CACHER le layer `npm run build` même si le code TSX a changé. Toujours vérifier :
```bash
docker compose exec frontend grep -c "<CHAINE_ATTENDUE>" server.js
# Si 0 → forcer --no-cache
```

---

## 3. Débugger

### 3.1 Logs
```bash
docker compose logs --tail 50 <service>          # dernières 50 lignes
docker compose logs -f <service>                 # tail -f live
docker compose logs -f | grep -v "GET /metrics"  # filtrer le bruit
```

### 3.2 Health checks
```bash
# Tous les services
docker compose ps

# Détail d'un service (état + OOM + restart count)
docker inspect <container_name> --format='status={{.State.Status}} health={{.State.Health.Status}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}'

# Endpoint /health
curl -s http://127.0.0.1:8002/health | python3 -m json.tool
```

### 3.3 Env vars du container
```bash
docker compose exec <service> env | grep -E "LLM_MODEL|OPENAI|GROQ"
```

### 3.4 Shell dans un container
```bash
docker compose exec <service> sh
```

---

## 4. Incidents fréquents

### 4.1 STT crashe en OOM (exit code 137)

**Symptôme** : `stt-1 exited with code 137 (restarting)` dans les logs, "Pipeline error 500" côté user.

**Cause** : Docker Desktop n'a pas assez de RAM allouée pour Whisper large-v3 (3 Go).

**Fix** :
1. Docker Desktop → Settings → Resources → Memory → passer à **12-14 Go**
2. Apply & restart
3. `docker compose up -d --force-recreate stt`

**Contournement rapide** : passer sur `WHISPER_MODEL=medium` (500 Mo) dans `.env` en local uniquement.

### 4.2 LLM rate limit — Groq quota atteint

**Symptôme** :
```
RateLimitError: Rate limit reached for model `openai/gpt-oss-20b`
Limit 200000, Used 199590
```

**Fix immédiat** (30 sec) — bascule sur un autre modèle Groq (quota séparé) :
```bash
sed -i 's|^LLM_MODEL=.*|LLM_MODEL=groq/openai/gpt-oss-120b|' .env
docker compose up -d --force-recreate llm pipeline
```

**Fix permanent** — basculer sur OpenAI (pay-as-you-go) :
```bash
echo "OPENAI_API_KEY=sk-proj-..." >> .env
sed -i 's|^LLM_MODEL=.*|LLM_MODEL=openai/gpt-4o-mini|' .env
docker compose up -d --force-recreate llm pipeline
```

### 4.3 Modèle Groq déprécié

**Symptôme** :
```
GroqException - The model `llama-3.1-8b-instant` does not exist or you do not have access to it.
```

**Cause** : Groq change son catalogue périodiquement. Le modèle a été supprimé.

**Fix** — récupérer la liste actuelle des modèles Groq :
```bash
source .env
curl -s https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]"
```

Choisir un modèle et mettre à jour `.env` + rebuild.

### 4.4 Frontend force un modèle obsolète

**Symptôme** : les changements du `.env` serveur n'ont aucun effet — le pipeline reçoit toujours l'ancien modèle en form data.

**Cause** : `frontend/app/page.tsx` a un default via `NEXT_PUBLIC_DEFAULT_LLM_MODEL` **bakerisé au build**. Le user a une ancienne valeur en `localStorage`.

**Fix** :
1. Vérifier le `.env` :
   ```bash
   grep NEXT_PUBLIC_DEFAULT_LLM_MODEL .env
   # Si absent : echo "NEXT_PUBLIC_DEFAULT_LLM_MODEL=openai/gpt-4o-mini" >> .env
   ```
2. Rebuild frontend **sans cache** :
   ```bash
   docker compose build --no-cache frontend
   docker compose up -d --force-recreate frontend
   ```
3. Côté user : F12 → Console → `localStorage.clear(); location.reload();`

### 4.5 Watcher pollue le quota LLM

**Symptôme** : quota Groq/OpenAI épuisé sans avoir fait beaucoup de tests.

**Cause** : le watcher tourne 24/7 et traduit chaque flash trafic dans 4 langues.

**Fix** — augmenter la fréquence de polling :
```bash
echo "POLL_INTERVAL_S=300" >> .env      # 5 min au lieu de 15s (÷20 consommation)
docker compose up -d --force-recreate watcher
```

Ou arrêter le watcher entre les démos :
```bash
docker compose stop watcher
docker compose start watcher            # avant la soutenance
```

### 4.6 Nginx sur hermes ne route pas correctement

**Symptôme** : `502 Bad Gateway` sur https://traduction-audio.fr

**Fix** :
```bash
# Vérifier que le frontend écoute bien
curl -sI http://127.0.0.1:3000

# Recharger nginx (nécessite sudo — Arnaud gère)
sudo nginx -t && sudo systemctl reload nginx

# Logs nginx
sudo tail -50 /var/log/nginx/traudio-error.log
```

### 4.7 Users pytest_* qui polluent la DB admin

**Symptôme** : le dashboard admin affiche des comptes `pytest_XXX@example.com`

**Cause** : anciens tests d'intégration qui n'avaient pas de cleanup.

**Fix immédiat** :
```bash
docker compose exec gateway python3 -c "
import sqlite3
c = sqlite3.connect('/app/data/auth.db')
n = c.execute('DELETE FROM users WHERE email LIKE \"pytest_%\"').rowcount
c.commit(); c.close()
print(f'Supprimés: {n}')
"
```

Le fix de fond est en place (fixture teardown + cleanup session-scoped dans `tests/integration/conftest.py`).

---

## 5. Rotation des clés API

**À faire si une clé est exposée** (screenshot, chat, log) :

| Clé | Où rotate |
|-----|-----------|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys → Revoke + Create new |
| `GROQ_API_KEY` | https://console.groq.com/keys → Delete + New |
| `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys/ → Revoke + Create |
| `LANGFUSE_SECRET_KEY` | https://cloud.langfuse.com → Settings → API Keys |
| `JWT_SECRET` | `openssl rand -hex 32` → remplace dans `.env` |

Après rotation :
```bash
# Mettre à jour .env local + hermes
docker compose up -d --force-recreate gateway pipeline llm tts watcher
```

⚠️ Si `JWT_SECRET` change → tous les tokens JWT existants deviennent invalides → tous les users doivent se relogger. C'est OK, c'est le point.

---

## 6. Backup / Restore

### Backup DB gateway (users)
```bash
docker compose exec gateway python3 -c "
import sqlite3, json
c = sqlite3.connect('/app/data/auth.db')
users = [dict(zip([col[0] for col in c.execute('SELECT * FROM users').description],
                  row)) for row in c.execute('SELECT * FROM users')]
print(json.dumps(users, default=str, indent=2))
" > backup_users_$(date +%Y%m%d).json
```

### Backup MLflow (runs)
```bash
docker compose cp mlflow:/mlflow ~/backup_mlflow_$(date +%Y%m%d)
```

### Backup watcher state
```bash
docker compose cp watcher:/app/state ~/backup_watcher_$(date +%Y%m%d)
```

---

## 7. Monitoring — dashboards utiles

| Dashboard | URL | Ce qu'on regarde |
|-----------|-----|-------------------|
| Grafana LLMOps Overview | http://localhost:3001/d/llmops-overview | Latence p95, req/s, taux d'erreur 5xx |
| MLflow experiments | http://localhost:5050 | Comparaison runs, champion, evaluate() |
| Prometheus targets | http://localhost:9090/targets | Health des 6 services scrapés |
| Langfuse traces | https://cloud.langfuse.com | Spans détaillés + coût par trace |
| Admin dashboard | http://localhost:3000/admin | Users, health, expériences agrégées |

---

## 8. Testing

```bash
# Unitaires (rapide, aucune dépendance)
pytest tests/unit/

# Intégration (nécessite docker compose up)
pytest tests/integration/

# Couverture
pytest --cov=backend --cov-report=term-missing
```

CI GitHub Actions lance `tests/unit/` à chaque push (103 tests, ~5s).

---

## 9. Contacts & escalation

| Rôle | Contact |
|------|---------|
| Owner du code | Maintainer principal du repo |
| Admin serveur (VPS Hetzner) | Interne |
| Providers LLM | Groq / OpenAI / Anthropic |
| DNS + Domaine | OVHcloud |

Si tout est cassé et rien ne redémarre : `docker compose down -v && docker compose up -d --build` (perte des données MLflow/Airflow, users préservés si volume `gateway_data` non détruit).
