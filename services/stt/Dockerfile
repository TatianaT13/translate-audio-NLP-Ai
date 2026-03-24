FROM python:3.11-slim

# Dépendances système pour faster-whisper + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le code source du package
COPY src/ src/
COPY scripts/run_pipeline.py scripts/run_pipeline.py
COPY services/stt/main.py services/stt/main.py

# Installer uniquement les dépendances nécessaires au STT
RUN pip install --no-cache-dir \
    faster-whisper>=1.0.3 \
    numpy>=1.26 \
    scipy>=1.11 \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    python-multipart>=0.0.9 \
    python-dotenv>=1.0

# Installer le package local
RUN pip install --no-cache-dir -e src/

ENV WHISPER_MODEL=small
ENV PORT=8001

EXPOSE 8001

CMD ["uvicorn", "services.stt.main:app", "--host", "0.0.0.0", "--port", "8001"]
