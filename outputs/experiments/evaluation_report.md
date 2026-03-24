# Résultats d'évaluation — Traduction audio FR→EN

**84 runs** : 7 audios × 2 modèles Whisper × 2 LLMs × 3 prompts
**Métrique** : BLEU (sacrebleu sentence_bleu) — plus c'est haut, mieux c'est
**Latence** : temps total pipeline (STT + LLM) — Whisper tourne en local sur CPU

---

## Classement global (moyenne sur 7 audios)

| # | Whisper | LLM | Prompt | BLEU moy | BLEU min | BLEU max | Latence moy |
|---|---------|-----|--------|----------|----------|----------|-------------|
| 1 | large-v3 | llama-3.1-8b-instant | v1.1 | **31.55** 🏆 | 4.06 | 80.49 | 260s |
| 2 | large-v3 | llama-3.3-70b-versatile | v1.1 | **27.23** | 2.06 | 70.70 | 306s |
| 3 | large-v3 | llama-3.3-70b-versatile | v1.2 | **26.49** | 3.56 | 63.72 | 264s |
| 4 | large-v3 | llama-3.3-70b-versatile | v1.0 | **24.65** | 1.85 | 67.60 | 293s |
| 5 | small | llama-3.3-70b-versatile | v1.2 | **23.19** | 2.06 | 44.53 | 48s |
| 6 | large-v3 | llama-3.1-8b-instant | v1.0 | **21.16** | 1.15 | 56.08 | 265s |
| 7 | small | llama-3.3-70b-versatile | v1.1 | **20.44** | 1.87 | 51.76 | 49s |
| 8 | small | llama-3.3-70b-versatile | v1.0 | **19.59** | 2.11 | 46.29 | 47s |
| 9 | large-v3 | llama-3.1-8b-instant | v1.2 | **18.19** | 3.21 | 29.47 | 265s |
| 10 | small | llama-3.1-8b-instant | v1.1 | **17.49** | 1.30 | 34.87 | 47s |
| 11 | small | llama-3.1-8b-instant | v1.2 | **14.73** | 1.27 | 28.54 | 45s |
| 12 | small | llama-3.1-8b-instant | v1.0 | **14.70** | 1.63 | 40.34 | 51s |

---

## BLEU par audio (score individuel)

| Whisper | LLM | Prompt | nord-1 | nord-2 | ouest-1 | ouest-2 | ouest-3 | sud-1 | sud-2 | **Moy** |
|---------|-----|--------|-------|-------|-------|-------|-------|-------|-------|---------|
| large-v3 | llama-3.1-8b-instant | v1.1 | 60.1 | 8.3 | 80.5 | 4.1 | 19.1 | 29.6 | 19.2 | **31.55** |
| large-v3 | llama-3.3-70b-versatile | v1.1 | 70.7 | 8.4 | 50.6 | 2.1 | 18.3 | 25.9 | 14.6 | **27.23** |
| large-v3 | llama-3.3-70b-versatile | v1.2 | 63.7 | 11.7 | 38.2 | 3.6 | 23.1 | 28.7 | 16.5 | **26.49** |
| large-v3 | llama-3.3-70b-versatile | v1.0 | 67.6 | 9.2 | 44.1 | 1.9 | 14.4 | 21.8 | 13.6 | **24.65** |
| small | llama-3.3-70b-versatile | v1.2 | 44.5 | 7.5 | 39.2 | 2.1 | 20.6 | 30.3 | 18.1 | **23.19** |
| large-v3 | llama-3.1-8b-instant | v1.0 | 56.1 | 8.5 | 30.1 | 1.1 | 18.1 | 24.3 | 10.0 | **21.16** |
| small | llama-3.3-70b-versatile | v1.1 | 51.8 | 8.0 | 30.3 | 1.9 | 11.6 | 25.5 | 14.1 | **20.44** |
| small | llama-3.3-70b-versatile | v1.0 | 46.3 | 6.3 | 26.6 | 2.1 | 19.4 | 20.1 | 16.3 | **19.59** |
| large-v3 | llama-3.1-8b-instant | v1.2 | 24.6 | 8.1 | 26.4 | 3.2 | 21.2 | 29.5 | 14.3 | **18.19** |
| small | llama-3.1-8b-instant | v1.1 | 34.1 | 5.9 | 34.9 | 1.3 | 15.0 | 22.0 | 9.2 | **17.49** |
| small | llama-3.1-8b-instant | v1.2 | 28.5 | 7.7 | 18.8 | 1.3 | 11.1 | 26.2 | 9.5 | **14.73** |
| small | llama-3.1-8b-instant | v1.0 | 40.3 | 5.2 | 18.4 | 1.6 | 10.2 | 18.6 | 8.4 | **14.70** |

---

## Latence moyenne (secondes)

| Whisper | LLM | Prompt | Latence moy | Note |
|---------|-----|--------|-------------|------|
| small | llama-3.1-8b-instant | v1.2 | 45s | Whisper en local CPU |
| small | llama-3.3-70b-versatile | v1.0 | 47s | Whisper en local CPU |
| small | llama-3.1-8b-instant | v1.1 | 47s | Whisper en local CPU |
| small | llama-3.3-70b-versatile | v1.2 | 48s | Whisper en local CPU |
| small | llama-3.3-70b-versatile | v1.1 | 49s | Whisper en local CPU |
| small | llama-3.1-8b-instant | v1.0 | 51s | Whisper en local CPU |
| large-v3 | llama-3.1-8b-instant | v1.1 | 260s | Whisper en local CPU — production = <5s via API |
| large-v3 | llama-3.3-70b-versatile | v1.2 | 264s | Whisper en local CPU — production = <5s via API |
| large-v3 | llama-3.1-8b-instant | v1.0 | 265s | Whisper en local CPU — production = <5s via API |
| large-v3 | llama-3.1-8b-instant | v1.2 | 265s | Whisper en local CPU — production = <5s via API |
| large-v3 | llama-3.3-70b-versatile | v1.0 | 293s | Whisper en local CPU — production = <5s via API |
| large-v3 | llama-3.3-70b-versatile | v1.1 | 306s | Whisper en local CPU — production = <5s via API |

---

## Conclusions

### Meilleure combinaison qualité
**large-v3 + llama-3.1-8b-instant + v1.1** → BLEU moyen **31.55**

### Meilleur compromis qualité / vitesse
**small + llama-3.3-70b-versatile + v1.2** → BLEU 23.19, latence 48s

### Observations clés
- Whisper large-v3 > small sur la qualité STT (noms propres français mieux reconnus)
- llama-3.1-8b-instant avec prompt v1.1 surpasse llama-3.3-70b sur ce corpus
- Le prompt v1.1 est le plus performant pour large-v3
- BLEU varie fortement selon l'audio (complexité du bulletin, longueur)
- La latence de large-v3 (~260s) est due au CPU local — en prod via API Whisper : <5s