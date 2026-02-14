# FitFlex — Gym Management System

> Full-stack gym management platform built with **React 18 + Vite** on the frontend and **Node.js + Express + MongoDB** on the backend. Production-grade architecture with Controller → Service → Repository layering, comprehensive security, Docker support, CI/CD, and monitoring.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Feature Flags](#feature-flags)
- [Code Quality](#code-quality)

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Frontend   │────▶│   Backend    │────▶│   MongoDB    │
│  React/Vite  │     │  Express API │     │   Mongoose   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                     ┌──────┴──────┐
                     │  Architecture│
                     ├─────────────┤
                     │  Routes      │  ← HTTP layer (validation, auth)
                     │  Services    │  ← Business logic
                     │  Repositories│  ← Data access (generic CRUD)
                     └─────────────┘
```

| Layer | Responsibility | Example |
|-------|---------------|---------|
| **Routes** | HTTP handling, validation, auth guards | `routes/core.js` |
| **Services** | Business logic, orchestration | `AuthService`, `MemberService` |
| **Repositories** | Database abstraction via `BaseRepository` | `repos.user.findById()` |
| **Models** | Mongoose schemas, indexes | `models/index.js` |

---

## Tech Stack

**Backend**: Node.js 20+, Express 4.18, MongoDB 7 + Mongoose 8, JWT (access + refresh rotation), Zod, Winston, Helmet, compression, rate limiting

**Frontend**: React 18.2, Vite 4.5, Tailwind CSS 3.3, Framer Motion 10, Lucide React, Axios

**DevOps**: Docker (multi-stage), Docker Compose, Nginx reverse proxy, GitHub Actions CI/CD

---

## Getting Started

### Prerequisites

- Node.js ≥ 18 (20 recommended) · MongoDB ≥ 6.0 · npm ≥ 9

### Quick Start

```bash
cd "Gym Management System"
npm install
cp .env.example Backend/.env
# Edit Backend/.env — set MONGODB_URI and AUTH_SECRET

# Start backend (hot reload)
npm run dev:backend      # or: cd Backend && npm run dev

# Start frontend (separate terminal)
npm run dev:frontend     # or: cd Frontend && npm run dev
```

- Frontend: `http://localhost:3000` (proxies `/api` → backend)
- Backend: `http://localhost:5001`
- Health check: `GET http://localhost:5001/api/health`

### Demo Logins

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Trainer | `trainer` | `trainer123` |
| Member | `member` | `member123` |

### Environment Variables

See [`.env.example`](.env.example) for all variables. Key ones:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | **Yes** | — | MongoDB connection string |
| `AUTH_SECRET` | **Yes** | — | JWT signing secret (min 8 chars) |
| `PORT` | No | `5001` | Backend port |
| `NODE_ENV` | No | `development` | Environment mode |
| `JWT_EXPIRES_IN` | No | `2h` | Access token expiry |
| `JWT_REFRESH_EXPIRES_IN` | No | `7d` | Refresh token expiry |
| `CORS_ORIGIN` | No | `http://localhost:3000` | Allowed origins |
| `SENTRY_DSN` | No | — | Sentry error tracking |
| `FEATURE_FLAGS` | No | — | Comma-separated flags |

---

## Docker

```bash
docker compose up --build -d     # Build & start
docker compose logs -f app       # View logs
docker compose down -v           # Stop & cleanup
```

Production-like (secure defaults: only Nginx exposed):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

| Service | Port | Description |
|---------|------|-------------|
| `app` | 5001 | Node.js backend |
| `ml` | 8001 | FastAPI ML/AI service (chat, insights, anomalies, clusters) |
| `mongo` | 27017 | MongoDB 7 |
| `redis` | 6379 | Redis 7 (caching) |
| `nginx` | 80/443 | Reverse proxy |
| `ollama` | 11434 | Local LLM runtime (optional) |

In Docker, Nginx proxies:

- `/api/*` → backend
- `/ml/*` → ML API (JWT-protected; uses the same `AUTH_SECRET`)

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/login` | Login |
| POST | `/api/register` | Register |
| POST | `/api/auth/refresh` | Refresh tokens |

### Member (`/api/me/*`) — attendance, invoices, progress, classes, tickets, notifications

### Trainer (`/api/trainer/*`) — sessions, classes, schedules, profile, members

### Admin (`/api/admin/*`) — CRUD, analytics, trainers, notifications, support

### General — `GET /api/stats`, `GET /api/users`, `GET /api/profile`, `GET /api/health`, `GET /metrics`

---

## Testing

```bash
cd Backend
npm test                 # Run all tests
npm run test:coverage    # With coverage report
npm run test:watch       # Watch mode
```

Tests use **Jest** with fully mocked repositories — no database required.

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):

1. **Lint** — ESLint + Prettier
2. **Test** — Jest + MongoDB service container + coverage
3. **Build** — Frontend Vite production build
4. **Docker** — Build & cache image (main branch only)

---

## Monitoring & Observability

- **Prometheus** metrics at `GET /metrics` (request duration, counts, active connections, Node.js internals)
- **Winston** structured logging with JSON format in production
- **Morgan** HTTP access logs with request IDs
- **Sentry** error tracking (set `SENTRY_DSN` to enable)

---

## Security

Helmet · CORS · Rate limiting (global + auth) · NoSQL injection prevention · HPP · JWT refresh token rotation · Zod validation · 1 MB body limit · Gzip compression · Request timeout

---

## Feature Flags

```bash
FEATURE_FLAGS=REDIS_CACHE,ADVANCED_ANALYTICS
```

Flags: `REDIS_CACHE`, `ML_RECOMMENDATIONS`, `AI_CHATBOT`, `ADVANCED_ANALYTICS`, `A_B_TESTING`

---

## Code Quality

```bash
npx eslint .              # Lint
npx prettier --check .    # Format check
npx prettier --write .    # Auto-fix
npm run smoke             # Smoke test (DB + API)
```

---

## Notes

- Seeds run automatically once when the database is empty
- MongoDB database is defined by `MONGODB_URI`
- For production: set `NODE_ENV=production`, strong `AUTH_SECRET`, hosted `MONGODB_URI`, and `CORS_ORIGIN` to your frontend URL
# Gym-Management-System
