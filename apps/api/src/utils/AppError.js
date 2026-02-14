/**
 * AppError — custom error class with HTTP status.
 *
 * Throw this anywhere in a route and the global error handler
 * will format the response automatically. No if-else chains needed.
 *
 *   throw new AppError('Not found', 404);
 *   throw AppError.notFound('User');
 *   throw AppError.forbidden();
 */
class AppError extends Error {
  constructor(message, statusCode = 500) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true; // distinguishes from programming bugs
    Error.captureStackTrace(this, this.constructor);
  }

  /* ── Convenience factories ─────────────────────────── */
  static badRequest(msg = 'Bad request') {
    return new AppError(msg, 400);
  }
  static unauthorized(msg = 'Unauthorized') {
    return new AppError(msg, 401);
  }
  static forbidden(msg = 'Forbidden') {
    return new AppError(msg, 403);
  }
  static notFound(resource = 'Resource') {
    return new AppError(`${resource} not found`, 404);
  }
  static conflict(msg = 'Conflict') {
    return new AppError(msg, 409);
  }
  static tooMany(msg = 'Too many requests') {
    return new AppError(msg, 429);
  }
}

/**
 * asyncHandler — wraps an async route so thrown errors flow to
 * Express's error middleware without explicit try/catch.
 *
 *   router.get('/foo', asyncHandler(async (req, res) => { … }));
 */
function asyncHandler(fn) {
  return (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);
}

module.exports = { AppError, asyncHandler };
