# ─── Base image with corepack (pnpm) ──────────────────────────
FROM node:20-alpine AS base
RUN corepack enable
WORKDIR /app

# ─── Install workspace deps (build context only) ──────────────
FROM base AS deps
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY apps/api/package.json apps/api/package.json
COPY apps/web/package.json apps/web/package.json
COPY packages ./packages
RUN pnpm install --frozen-lockfile --ignore-scripts

# ─── Build frontend ───────────────────────────────────────────
FROM deps AS build
COPY . .
RUN pnpm --filter @fitflex/web build

# ─── Production deps for API only ─────────────────────────────
FROM base AS api_deps
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
COPY apps/api/package.json apps/api/package.json
COPY packages ./packages
RUN pnpm install --frozen-lockfile --prod --filter @fitflex/api --ignore-scripts

# ─── Final image ──────────────────────────────────────────────
FROM node:20-alpine AS production
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
WORKDIR /app
RUN corepack enable

# Copy node_modules (production only) and app code
COPY --from=api_deps /app/node_modules ./node_modules
COPY --from=build /app/apps/api ./apps/api
COPY --from=build /app/apps/web/dist ./apps/web/dist

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5001/api/health || exit 1

ENV NODE_ENV=production
EXPOSE 5001
USER appuser

CMD ["node", "apps/api/server.js"]
