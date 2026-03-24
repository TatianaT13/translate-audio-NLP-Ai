FROM python:3.11-slim

WORKDIR /app

COPY src/ src/
COPY services/tts/main.py services/tts/main.py
# Les modèles MMS-TTS sont montés via volume dans docker-compose
# (trop lourds pour être copiés dans l'image)

RUN pip install --no-cache-dir \
    transformers>=4.40 \
    torch>=2.2 \
    numpy>=1.26 \
    scipy>=1.11 \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    python-dotenv>=1.0

RUN pip install --no-cache-dir -e src/

ENV PORT=8003

EXPOSE 8003

CMD ["uvicorn", "services.tts.main:app", "--host", "0.0.0.0", "--port", "8003"]
