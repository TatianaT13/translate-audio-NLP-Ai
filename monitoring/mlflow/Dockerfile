# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow==2.17.2"

# Non-root user
RUN groupadd -r app && useradd -r -g app -u 1000 -m app && \
    mkdir -p /mlflow/artifacts && chown -R app:app /mlflow

USER app
WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
