FROM python:3.11-slim

WORKDIR /app

COPY services/pipeline/main.py services/pipeline/main.py

RUN pip install --no-cache-dir \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    httpx>=0.27 \
    langchain-core>=0.2 \
    python-multipart>=0.0.9 \
    python-dotenv>=1.0 \
    langfuse>=2.0

ENV STT_URL=http://stt:8001
ENV LLM_URL=http://llm:8002
ENV TTS_URL=http://tts:8003
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "services.pipeline.main:app", "--host", "0.0.0.0", "--port", "8000"]
