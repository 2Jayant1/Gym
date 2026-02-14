/**
 * Prometheus-compatible metrics middleware.
 *
 * Exposes request duration, request count, and active connections.
 * Metrics are available at GET /metrics in Prometheus text exposition format.
 *
 * If `prom-client` is not installed the middleware is a no-op (graceful degradation).
 */

let client;
try {
  client = require('prom-client');
} catch {
  // prom-client not installed – export stubs
}

function createMetricsMiddleware() {
  if (!client) {
    return {
      middleware: (_req, _res, next) => next(),
      metricsHandler: (_req, res) => res.status(503).json({ message: 'prom-client not installed' }),
    };
  }

  // Create a dedicated registry so we don't pollute the global one
  const register = new client.Registry();

  // Collect default Node.js metrics (CPU, memory, event loop lag, etc.)
  client.collectDefaultMetrics({ register, prefix: 'gym_' });

  // ── Custom metrics ────────────────────────────────────────────────────────────

  const httpRequestDuration = new client.Histogram({
    name: 'gym_http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code'],
    buckets: [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5],
    registers: [register],
  });

  const httpRequestTotal = new client.Counter({
    name: 'gym_http_requests_total',
    help: 'Total number of HTTP requests',
    labelNames: ['method', 'route', 'status_code'],
    registers: [register],
  });

  const httpActiveRequests = new client.Gauge({
    name: 'gym_http_active_requests',
    help: 'Number of in-flight HTTP requests',
    registers: [register],
  });

  // ── Middleware ─────────────────────────────────────────────────────────────────

  function middleware(req, res, next) {
    // Skip metrics endpoint itself
    if (req.path === '/metrics') return next();

    httpActiveRequests.inc();
    const end = httpRequestDuration.startTimer();

    res.on('finish', () => {
      const route = req.route?.path || req.path || 'unknown';
      const labels = {
        method: req.method,
        route,
        status_code: res.statusCode,
      };

      end(labels);
      httpRequestTotal.inc(labels);
      httpActiveRequests.dec();
    });

    next();
  }

  // ── /metrics handler ──────────────────────────────────────────────────────────

  async function metricsHandler(_req, res) {
    try {
      res.set('Content-Type', register.contentType);
      res.end(await register.metrics());
    } catch (err) {
      res.status(500).end(err.message);
    }
  }

  return { middleware, metricsHandler, register };
}

module.exports = createMetricsMiddleware;
