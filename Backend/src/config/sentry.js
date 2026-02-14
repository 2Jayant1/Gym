/**
 * Sentry error-tracking placeholder.
 *
 * When SENTRY_DSN is set this module initialises the Sentry SDK.
 * Until @sentry/node is added as a dependency this is a graceful no-op.
 */

let Sentry;

function initSentry(config) {
  if (!config.SENTRY_DSN) return;

  try {
    Sentry = require('@sentry/node');
    Sentry.init({
      dsn: config.SENTRY_DSN,
      environment: config.NODE_ENV,
      tracesSampleRate: config.NODE_ENV === 'production' ? 0.2 : 1.0,
      integrations: [],
    });
    console.log('[sentry] initialised');
  } catch {
    console.warn('[sentry] @sentry/node not installed – skipping');
  }
}

function captureException(err) {
  if (Sentry) Sentry.captureException(err);
}

function requestHandler() {
  if (Sentry) return Sentry.Handlers?.requestHandler?.() ?? ((_r, _s, n) => n());
  return (_req, _res, next) => next();
}

function errorHandler() {
  if (Sentry) return Sentry.Handlers?.errorHandler?.() ?? ((_e, _r, _s, n) => n());
  return (_err, _req, _res, next) => next(_err);
}

module.exports = { initSentry, captureException, requestHandler, errorHandler };
