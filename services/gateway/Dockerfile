FROM python:3.11-slim

WORKDIR /app/services/gateway

COPY services/gateway/ .

RUN pip install --no-cache-dir \
    fastapi>=0.111 \
    "uvicorn[standard]>=0.29" \
    "python-jose[cryptography]>=3.3" \
    "passlib[bcrypt]>=1.7" \
    sqlalchemy>=2.0 \
    "pydantic[email]>=2.0" \
    python-dotenv>=1.0 \
    httpx>=0.27

ENV JWT_SECRET=""
ENV DEV_MODE="false"
ENV DATABASE_URL="sqlite:////app/data/auth.db"
ENV FRONTEND_URLS="http://localhost:3000"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
