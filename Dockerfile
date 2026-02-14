# ─── Stage 1: Builder ─────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install production dependencies only for backend
COPY Backend/package.json Backend/package-lock.json* ./Backend/
RUN cd Backend && npm ci --omit=dev

# Install and build frontend
COPY Frontend/package.json Frontend/package-lock.json* ./Frontend/
RUN cd Frontend && npm ci
COPY Frontend/ ./Frontend/
RUN cd Frontend && npm run build

# ─── Stage 2: Production image ────────────────────────────────
FROM node:20-alpine AS production

# Security: run as non-root
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

# Copy backend + production deps
COPY --from=builder /app/Backend/node_modules ./Backend/node_modules
COPY Backend/ ./Backend/

# Copy built frontend to a static folder
COPY --from=builder /app/Frontend/dist ./Frontend/dist

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5001/api/health || exit 1

USER appuser

EXPOSE 5001

ENV NODE_ENV=production

CMD ["node", "Backend/server.js"]
