FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Package flash_nlp
COPY pyproject.toml ./
COPY src/ src/

# Service
COPY services/watcher/main.py services/watcher/main.py

# Dépendances
RUN pip install --no-cache-dir \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    httpx>=0.27 \
    faster-whisper>=1.0.3 \
    numpy>=1.26 \
    python-dotenv>=1.0 \
    requests>=2.31

# Installer flash_nlp comme package
RUN pip install --no-cache-dir -e .

EXPOSE 8005
CMD ["uvicorn", "services.watcher.main:app", "--host", "0.0.0.0", "--port", "8005"]
