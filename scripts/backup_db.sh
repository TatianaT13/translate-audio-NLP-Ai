#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# scripts/backup_db.sh — Sauvegarde de la base des comptes et des crédits
#
# Tout ce qui ne se reconstruit pas (comptes, abonnements, soldes, historique
# des crédits) tient dans un seul fichier SQLite du gateway. Ce script en fait
# une copie cohérente à chaud (API backup de SQLite, pas un simple cp), la
# vérifie, la compresse et supprime les copies trop anciennes.
#
# Usage :
#   ./scripts/backup_db.sh                 # vers ~/backups/traudio
#   ./scripts/backup_db.sh /chemin/dossier
#
# Cron quotidien (crontab -e) :
#   0 3 * * * cd ~/traudio && ./scripts/backup_db.sh >> ~/backups/traudio/backup.log 2>&1
#
# Une sauvegarde restée sur le même serveur ne protège pas d'une panne du
# serveur : recopier le dossier ailleurs (rsync, rclone...).
# Restauration : voir docs/RUNBOOK.md § 6.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

cd "$(dirname "$0")/.."

readonly DEST="${1:-$HOME/backups/traudio}"
readonly KEEP_DAYS="${BACKUP_KEEP_DAYS:-30}"
readonly OUT="$DEST/auth_$(date +%Y%m%d_%H%M%S).db"

mkdir -p "$DEST"
chmod 700 "$DEST"   # contient des emails et des hashs de mots de passe

# Copie + contrôle d'intégrité dans le conteneur (seul endroit où le fichier est lisible)
docker compose exec -T gateway python3 - <<'PY'
import sqlite3, sys

src = sqlite3.connect("/app/data/auth.db")
dst = sqlite3.connect("/tmp/auth_backup.db")
src.backup(dst)
src.close()
check = dst.execute("PRAGMA integrity_check").fetchone()[0]
users = dst.execute("SELECT COUNT(*) FROM users").fetchone()[0]
dst.close()
if check != "ok":
    sys.exit(f"sauvegarde corrompue : {check}")
print(f"copie verifiee : {users} compte(s)")
PY

docker compose cp gateway:/tmp/auth_backup.db "$OUT"
docker compose exec -T gateway rm -f /tmp/auth_backup.db

gzip "$OUT"
chmod 600 "$OUT.gz"

find "$DEST" -name 'auth_*.db.gz' -mtime +"$KEEP_DAYS" -delete

echo "$(date '+%F %T') sauvegarde : $OUT.gz ($(du -h "$OUT.gz" | cut -f1)), $(find "$DEST" -name 'auth_*.db.gz' | wc -l | tr -d ' ') copie(s) conservée(s)"
