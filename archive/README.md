# Archive

Dossiers et fichiers conservés pour référence historique mais **non utilisés** par l'architecture actuelle (microservices conteneurisés).

## Contenu

| Élément | Origine | Pourquoi archivé |
|---------|---------|------------------|
| `app/` | Ancienne app Streamlit standalone | Remplacée par le frontend Next.js + microservices backend |
| `models/` | Modèles MMS-TTS téléchargés localement (`mms-tts-eng`, `mms-tts-ukr`) | Le service TTS utilise maintenant le cache HuggingFace dans le conteneur (volume `tts_models`) |
| `notebooks/` | Dossier Jupyter (vide, `.gitkeep`) | Aucun notebook utilisé en prod |
| `.live_chunk.wav` | Échantillon audio de test live | Test ponctuel |
| `watcher_requirements.txt.legacy` | Ancien `requirements.txt` du watcher | Remplacé par `pyproject.toml` |

## Restauration

Pour réutiliser un de ces éléments, le déplacer hors de `archive/` :

```bash
mv archive/<nom> ./
```
