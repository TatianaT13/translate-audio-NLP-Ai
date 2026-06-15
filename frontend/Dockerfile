# syntax=docker/dockerfile:1.7

# ─── Stage 1 : Dependencies ────────────────────────────────────────────────────
FROM node:22-alpine AS deps

RUN apk add --no-cache libc6-compat

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

# ─── Stage 2 : Builder ─────────────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js fige les NEXT_PUBLIC_* AU BUILD (pas au runtime). On les reçoit via
# des build args passés par docker-compose → ils définissent ENV → npm build.
# Défauts = dev local (URLs localhost). En prod : surcharge par .env du serveur.
ARG NEXT_PUBLIC_GATEWAY_URL=http://localhost:8004
ARG NEXT_PUBLIC_PIPELINE_URL=http://localhost:8000
ARG NEXT_PUBLIC_STT_URL=http://localhost:8001
ARG NEXT_PUBLIC_LLM_URL=http://localhost:8002

ENV NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_GATEWAY_URL=$NEXT_PUBLIC_GATEWAY_URL \
    NEXT_PUBLIC_PIPELINE_URL=$NEXT_PUBLIC_PIPELINE_URL \
    NEXT_PUBLIC_STT_URL=$NEXT_PUBLIC_STT_URL \
    NEXT_PUBLIC_LLM_URL=$NEXT_PUBLIC_LLM_URL

RUN npm run build

# ─── Stage 3 : Runtime (minimal, non-root) ─────────────────────────────────────
FROM node:22-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Non-root user (already present in node images: node, UID 1000)
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
# Ensure all static assets are world-readable (macOS may copy with 600 perms)
RUN chmod -R a+r ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

CMD ["node", "server.js"]
