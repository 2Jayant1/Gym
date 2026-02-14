/**
 * FitFlex Gym Management — Express Server (production-grade)
 *
 * Architecture: Controller → Service → Repository
 *
 * Boot order:
 *   1. Validate environment variables (crash early on bad config).
 *   2. Connect to MongoDB.
 *   3. Initialize repositories + services.
 *   4. Apply security & performance middleware.
 *   5. Mount routes (inject services via factory).
 *   6. Global error handler (no try/catch needed in routes).
 *   7. Graceful shutdown on SIGTERM / SIGINT.
 */
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const mongoSanitize = require('express-mongo-sanitize');
const hpp = require('hpp');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const path = require('path');

const config       = require('./src/config/env');
const logger       = require('./src/config/logger');
const featureFlags = require('./src/config/featureFlags');
const { initSentry, captureException, requestHandler: sentryRequestHandler, errorHandler: sentryErrorHandler } = require('./src/config/sentry');
const { connectDb } = require('./src/db/mongoose');
const models       = require('./src/models');
const { seedIfEmpty } = require('./src/seed/seed.mongo');

// Architecture layers
const createRepositories = require('./src/repositories');
const createServices     = require('./src/services');

// Middleware
const { requireAuth, requireRole } = require('./src/middleware/auth');
const requestId      = require('./src/middleware/requestId');
const requestTimeout = require('./src/middleware/requestTimeout');
const createMetricsMiddleware = require('./src/middleware/metrics');
const { AppError }   = require('./src/utils/AppError');

// Initialise Sentry (no-op if SENTRY_DSN is unset or @sentry/node not installed)
initSentry(config);

// Routes
const coreRoutes    = require('./src/routes/core');
const adminRoutes   = require('./src/routes/admin');
const trainerRoutes = require('./src/routes/trainer');
const meRoutes      = require('./src/routes/me');

/* ─── Initialize Repository → Service layers ─────────── */
const repos    = createRepositories(models);
const services = createServices({ repos, config, models });

const app = express();

/* ─── Trust proxy (needed behind Nginx/ALB for correct IP) ── */
app.set('trust proxy', 1);

/* ─── Sentry request handler (must be first middleware) ────── */
app.use(sentryRequestHandler());

/* ─── Prometheus metrics ─────────────────────────────────────── */
const { middleware: metricsMiddleware, metricsHandler } = createMetricsMiddleware();
app.use(metricsMiddleware);
app.get('/metrics', metricsHandler);

/* ═══════════════════════════════════════════════════════════════
   REQUEST ID + TIMEOUT
   ═══════════════════════════════════════════════════════════════ */
app.use(requestId);
app.use(requestTimeout(config.REQUEST_TIMEOUT_MS));

/* ═══════════════════════════════════════════════════════════════
   SECURITY MIDDLEWARE
   ═══════════════════════════════════════════════════════════════ */

// HTTP security headers (XSS, MIME sniffing, clickjacking, etc.)
app.use(helmet());

// CORS — locked to configured origin(s)
app.use(
  cors({
    origin: config.CORS_ORIGIN === '*'
      ? true
      : config.CORS_ORIGIN.split(',').map((v) => v.trim()),
    credentials: true,
  }),
);

// Body parsers with size limits (prevents payload DOS)
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: false, limit: '1mb' }));

// Sanitize req.body/query/params against NoSQL injection ($gt, $ne, etc.)
app.use(mongoSanitize());

// Prevent HTTP parameter pollution (?sort=name&sort=email → only last)
app.use(hpp());

// Gzip/Brotli compression for all responses
app.use(compression());

/* ═══════════════════════════════════════════════════════════════
   RATE LIMITING
   ═══════════════════════════════════════════════════════════════ */

// Global rate limit
app.use(
  rateLimit({
    windowMs: config.RATE_LIMIT_WINDOW_MS,
    max: config.RATE_LIMIT_MAX,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests, please try again later.' },
  }),
);

// Aggressive limit on auth endpoints (prevent brute-force)
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many login attempts. Try again in 15 minutes.' },
});

/* ═══════════════════════════════════════════════════════════════
   LOGGING
   ═══════════════════════════════════════════════════════════════ */

// Pipe Morgan HTTP logs through Winston (include request ID)
const morganStream = { write: (msg) => logger.http(msg.trim()) };
app.use(morgan(
  config.NODE_ENV === 'production'
    ? ':remote-addr :method :url :status :res[content-length] - :response-time ms'
    : 'dev',
  { stream: morganStream },
));

/* ═══════════════════════════════════════════════════════════════
   HEALTH CHECK  (deep — pings the DB + reports feature flags)
   ═══════════════════════════════════════════════════════════════ */
app.get('/api/health', async (_req, res) => {
  const dbState = mongoose.connection.readyState;
  const dbOk = dbState === 1;
  const status = dbOk ? 200 : 503;
  res.status(status).json({
    ok: dbOk,
    service: 'gym-management-backend',
    version: '2.0.0',
    time: new Date().toISOString(),
    db: dbOk ? 'connected' : 'disconnected',
    uptime: Math.round(process.uptime()),
    features: featureFlags.getAll(),
  });
});

/* ═══════════════════════════════════════════════════════════════
   AUTH MIDDLEWARE INSTANCES
   ═══════════════════════════════════════════════════════════════ */
const authMw = requireAuth({ secret: config.AUTH_SECRET });

/* ═══════════════════════════════════════════════════════════════
   ROUTES — services are injected into route factories
   ═══════════════════════════════════════════════════════════════ */
app.use(
  '/api',
  coreRoutes({
    services,
    models,
    requireAuth: authMw,
    requireRole,
    authLimiter,
  }),
);

app.use('/api/admin',   adminRoutes({ services, models, requireAuth: authMw, requireRole }));
app.use('/api/trainer', trainerRoutes({ services, requireAuth: authMw, requireRole }));
app.use('/api/me',      meRoutes({ services, requireAuth: authMw }));

/* ═══════════════════════════════════════════════════════════════
   FRONTEND STATIC (production)
   Serve the Vite build output bundled into the Docker image.
   ═══════════════════════════════════════════════════════════════ */
if (config.NODE_ENV === 'production') {
  const distPath = path.join(__dirname, '..', 'web', 'dist');
  app.use(express.static(distPath));
  app.get('*', (req, res, next) => {
    // Keep API/metrics routes handled by Express routers
    if (req.originalUrl.startsWith('/api') || req.originalUrl === '/metrics') return next();
    return res.sendFile(path.join(distPath, 'index.html'));
  });
}

/* ─── Dev-only reseed endpoint ──────────────────────────────── */
if (config.NODE_ENV === 'development') {
  const { asyncHandler } = require('./src/utils/AppError');
  app.post('/api/dev/reseed', authMw, requireRole(['admin']), asyncHandler(async (_req, res) => {
    const collections = await mongoose.connection.db.collections();
    for (const c of collections) await c.deleteMany({});
    await seedIfEmpty(models);
    res.json({ ok: true, message: 'Database reseeded successfully' });
  }));
}

/* ═══════════════════════════════════════════════════════════════
   404 HANDLER — must come after all routes
   ═══════════════════════════════════════════════════════════════ */
app.all('*', (req, _res, next) => {
  next(new AppError(`Route ${req.method} ${req.originalUrl} not found`, 404));
});

/* ─── Sentry error handler (before our global error handler) ── */
app.use(sentryErrorHandler());

/* ═══════════════════════════════════════════════════════════════
   GLOBAL ERROR HANDLER
   —————————————————————————————————————————————————————————————
   Every thrown error or next(err) ends up here.
   • Operational errors (AppError) → send message to client.
   • Mongoose validation errors    → formatted nicely.
   • Duplicate key errors          → user-friendly message.
   • Everything else               → generic 500.
   ═══════════════════════════════════════════════════════════════ */
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, _next) => {
  // Mongoose validation error → 400
  if (err.name === 'ValidationError') {
    const messages = Object.values(err.errors).map((e) => e.message);
    return res.status(400).json({ error: messages.join('. ') });
  }

  // Mongoose duplicate key → 409
  if (err.code === 11000) {
    const field = Object.keys(err.keyPattern || {})[0] || 'field';
    return res.status(409).json({ error: `Duplicate value for "${field}". That value already exists.` });
  }

  // Mongoose bad ObjectId → 400
  if (err.name === 'CastError' && err.kind === 'ObjectId') {
    return res.status(400).json({ error: `Invalid ID: ${err.value}` });
  }

  // JWT errors → 401
  if (err.name === 'JsonWebTokenError' || err.name === 'TokenExpiredError') {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }

  // Our own AppError
  const statusCode = err.statusCode || 500;
  const message = err.isOperational ? err.message : 'Internal server error';

  // Log non-operational (unexpected) errors fully (with request context)
  if (!err.isOperational) {
    logger.error('Unhandled error', {
      err: err.message,
      stack: err.stack,
      requestId: req.id,
      method: req.method,
      url: req.originalUrl,
    });
    captureException(err);
  }

  res.status(statusCode).json({ error: message });
});

/* ═══════════════════════════════════════════════════════════════
   START
   ═══════════════════════════════════════════════════════════════ */
let server;

(async () => {
  try {
    await connectDb({ uri: config.MONGODB_URI });
    logger.info('Connected to MongoDB');
    await seedIfEmpty(models);

    server = app.listen(config.PORT, () => {
      logger.info(`Server running on http://localhost:${config.PORT} [${config.NODE_ENV}]`);
      logger.info('Architecture: Controller → Service → Repository');
    });

    server.on('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        logger.error(`Port ${config.PORT} is already in use. Set the PORT env var or free the port.`);
      } else {
        logger.error('Server error', { err: err.message });
      }
      process.exit(1);
    });
  } catch (err) {
    logger.error('Startup failed', { err: err.message });
    process.exit(1);
  }
})();

/* ═══════════════════════════════════════════════════════════════
   GRACEFUL SHUTDOWN
   ═══════════════════════════════════════════════════════════════ */
function gracefulShutdown(signal) {
  logger.info(`${signal} received — shutting down gracefully…`);
  server.close(async () => {
    try {
      await mongoose.connection.close();
      logger.info('MongoDB connection closed.');
    } catch { /* swallow */ }
    process.exit(0);
  });

  // Force-kill after 10 seconds if connections hang
  setTimeout(() => {
    logger.warn('Forcing shutdown after 10s timeout.');
    process.exit(1);
  }, 10_000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT',  () => gracefulShutdown('SIGINT'));

// Catch unhandled rejections/exceptions — log and exit
process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled Rejection', { err: reason });
  server.close(() => process.exit(1));
});

process.on('uncaughtException', (err) => {
  logger.error('Uncaught Exception', { err: err.message, stack: err.stack });
  server.close(() => process.exit(1));
});
