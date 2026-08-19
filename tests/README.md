# Tests

Structure organisée par **niveau de test** — de la logique pure aux flows bout-en-bout.

```
tests/
├── conftest.py               # fixtures globales (config, paths)
│
├── unit/                     # logique pure, sans réseau, rapide (<1s / test)
│   ├── pipeline/
│   │   └── test_prompt_guard.py
│   ├── gateway/              # (à venir) — auth utils, JWT encoding…
│   ├── llm/                  # (à venir) — cleanup meta, pricing…
│   └── legacy/               # anciens tests flash_nlp (avant microservices)
│
├── integration/              # appels HTTP réels aux services (Docker up requis)
│   ├── conftest.py           # fixtures HTTP (URLs, clients, tokens)
│   ├── gateway/
│   │   └── test_auth_flow.py
│   ├── pipeline/
│   │   └── test_pipeline_e2e.py
│   ├── stt/                  # (à venir)
│   ├── llm/                  # (à venir)
│   ├── tts/                  # (à venir)
│   └── watcher/              # (à venir)
│
└── e2e/                      # Phase 2 — Playwright via UI Next.js
    └── (à venir)
```

## Lancer les tests

**Tous les tests unitaires** (rapide, aucune dépendance) :
```bash
pytest tests/unit/ -v
```

**Tests d'intégration** (nécessite `docker compose up -d`) :
```bash
pytest tests/integration/ -v
```

**Un seul dossier / fichier** :
```bash
pytest tests/integration/gateway/ -v
pytest tests/unit/pipeline/test_prompt_guard.py -v
```

**Filtrer par marker** :
```bash
pytest -m "not integration"   # skip les tests HTTP
pytest -m integration          # que les tests HTTP
```

**Couverture** :
```bash
pytest --cov=backend --cov-report=term-missing tests/unit/
```

## Conventions

- **Nommer les fichiers** : `test_<module>.py` (miroir de la structure `backend/services/`)
- **Nommer les fonctions** : `test_<comportement>_<contexte>()` — ex: `test_check_input_blocks_injection_fr()`
- **Un test = une assertion logique** — préférer plusieurs tests courts à un long test avec 10 assertions
- **Fixtures** dans le `conftest.py` du dossier concerné (portée locale) ou racine (portée globale)
- **Skip élégant** si un service n'est pas up : les tests d'intégration checkent la santé avant de courir

## Ajouter un test pour un nouveau service

1. Créer le dossier `tests/integration/<service>/`
2. Ajouter un `__init__.py` vide
3. Écrire `test_<endpoint>.py` avec fixtures depuis `tests/integration/conftest.py`
