"""
Supprime TOUS les scores Langfuse via l'API publique.
À lancer avant import_metrics_to_langfuse.py pour repartir propre.

Usage :
    .venv.nosync/bin/python scripts/clear_langfuse_scores.py
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx

HOST   = os.getenv("LANGFUSE_HOST",       "https://cloud.langfuse.com")
PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")

if not PUBLIC or not SECRET:
    print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY manquants.")
    sys.exit(1)

auth = (PUBLIC, SECRET)


def fetch_all_scores() -> list[dict]:
    scores: list[dict] = []
    page = 1
    with httpx.Client(timeout=30.0) as client:
        while True:
            r = client.get(f"{HOST}/api/public/scores",
                           auth=auth, params={"limit": 100, "page": page})
            r.raise_for_status()
            batch = r.json().get("data", [])
            scores.extend(batch)
            print(f"  page {page} — {len(batch)} scores (total {len(scores)})")
            if len(batch) < 100:
                break
            page += 1
    return scores


def delete_score(client: httpx.Client, sid: str) -> bool:
    try:
        r = client.delete(f"{HOST}/api/public/scores/{sid}", auth=auth)
        return r.is_success
    except Exception:
        return False


def main() -> None:
    print(f"Récupération des scores depuis {HOST}…")
    scores = fetch_all_scores()
    print(f"\n{len(scores)} scores trouvés. Suppression…")

    if not scores:
        return

    deleted = 0
    with httpx.Client(timeout=30.0) as client:
        for i, s in enumerate(scores, 1):
            if delete_score(client, s["id"]):
                deleted += 1
            if i % 50 == 0 or i == len(scores):
                print(f"  [{i}/{len(scores)}] supprimé={deleted}")

    print(f"\nTerminé : {deleted}/{len(scores)} scores supprimés.")


if __name__ == "__main__":
    main()
