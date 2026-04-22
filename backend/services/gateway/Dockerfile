# syntax=docker/dockerfile:1.7

# ─── Stage 1 : Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv (much faster than pip)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install deps in /opt/venv (cached layer — only re-runs if pyproject.toml changes)
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r pyproject.toml

# ─── Stage 2 : Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user (UID 1000)
RUN groupadd -r app && useradd -r -g app -u 1000 -m app && \
    mkdir -p /app/data /app/experiments && \
    chown -R app:app /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/ /app/src/

USER app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8004/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8004"]
