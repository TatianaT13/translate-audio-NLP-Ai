# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml ./
# Torch CPU-only first (évite ~2 Go de libs CUDA inutiles), puis le reste
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        torch && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml

FROM python:3.11-slim AS runtime
RUN groupadd -r app && useradd -r -g app -u 1000 -m app && \
    mkdir -p /home/app/.cache/huggingface && \
    chown -R app:app /home/app/.cache
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app src/ /app/src/
USER app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/health').status==200 else 1)" || exit 1
CMD ["uvicorn", "tts.main:app", "--host", "0.0.0.0", "--port", "8003"]
