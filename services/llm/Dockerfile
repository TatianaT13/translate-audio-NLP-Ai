FROM python:3.11-slim

WORKDIR /app

COPY src/ src/
COPY scripts/run_pipeline.py scripts/run_pipeline.py
COPY services/llm/main.py services/llm/main.py

RUN pip install --no-cache-dir \
    litellm>=1.40 \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    python-dotenv>=1.0

RUN pip install --no-cache-dir -e src/

ENV LLM_MODEL=groq/llama-3.1-8b-instant
ENV PROMPT_VERSION=v1.1
ENV PORT=8002

EXPOSE 8002

CMD ["uvicorn", "services.llm.main:app", "--host", "0.0.0.0", "--port", "8002"]
