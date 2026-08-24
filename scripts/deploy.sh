#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# scripts/deploy.sh — Déploiement traudio en 1 commande
#
# Garantit :
#   1. git pull de la dernière version
#   2. rebuild SANS cache (pas de layer périmé)
#   3. force-recreate des containers (recharge .env + volumes)
#   4. attente healthchecks
#   5. affichage du commit déployé
#
# Usage :
#   ./scripts/deploy.sh                    # rebuild tous les services app
#   ./scripts/deploy.sh frontend           # rebuild uniquement le frontend
#   ./scripts/deploy.sh frontend pipeline  # rebuild plusieurs services
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

cd "$(dirname "$0")/.."

# Services par défaut : tous ceux qu'on construit nous-mêmes
readonly DEFAULT_SERVICES="frontend pipeline llm stt tts gateway watcher"
readonly TARGETS="${*:-$DEFAULT_SERVICES}"

# Couleurs pour lisibilité
info()  { printf "\033[1;36m▸ %s\033[0m\n" "$*"; }
ok()    { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m⚠ %s\033[0m\n" "$*"; }
err()   { printf "\033[1;31m✗ %s\033[0m\n" "$*" >&2; }

START_TIME=$(date +%s)

# 1. Pull dernière version
info "Étape 1/5 — git pull"
git fetch origin main
BEFORE_COMMIT=$(git rev-parse --short HEAD)
git pull --ff-only origin main
AFTER_COMMIT=$(git rev-parse --short HEAD)
if [ "$BEFORE_COMMIT" = "$AFTER_COMMIT" ]; then
    ok "Déjà à jour ($AFTER_COMMIT)"
else
    ok "Mis à jour : $BEFORE_COMMIT → $AFTER_COMMIT"
fi
echo ""

# 2. Rebuild sans cache — garantit que le code sur disque est bien dans l'image
info "Étape 2/5 — docker compose build --no-cache $TARGETS"
docker compose build --no-cache $TARGETS
ok "Images reconstruites"
echo ""

# 3. Force recreate — garantit que les env vars du .env sont rechargées
info "Étape 3/5 — docker compose up -d --force-recreate"
docker compose up -d --force-recreate
ok "Containers recréés"
echo ""

# 4. Wait for health
info "Étape 4/5 — attente healthchecks (max 90s)"
for i in {1..30}; do
    UNHEALTHY=$(docker compose ps --format "{{.Status}}" | grep -cE "starting|unhealthy" || true)
    if [ "$UNHEALTHY" = "0" ]; then
        ok "Tous les services healthy en ${i}× 3s"
        break
    fi
    if [ "$i" = "30" ]; then
        warn "Timeout attente healthchecks — vérifier docker compose ps"
    fi
    sleep 3
done
echo ""

# 5. Récap
info "Étape 5/5 — état final"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
echo ""

ELAPSED=$(( $(date +%s) - START_TIME ))
ok "Déploiement terminé en ${ELAPSED}s — version $AFTER_COMMIT"
echo ""

# Suggestion post-deploy
cat <<EOF
─── Étapes recommandées côté client ─────────────────
  1. Vider le localStorage browser (F12 → Console) :
       localStorage.clear(); location.reload();
  2. Vérifier /health :
       curl -s http://127.0.0.1:8002/health | python3 -m json.tool
─────────────────────────────────────────────────────
EOF
