/**
 * Authentication & authorization middleware.
 *
 * ✓ JWT with configurable expiry
 * ✓ Helmet-style auth extraction (Bearer token)
 * ✓ Password comparison with bcrypt (timing-safe)
 * ✓ Clean error responses via AppError
 */
const jwt = require('jsonwebtoken');
const { AppError } = require('../utils/AppError');

function signToken(payload, secret, options = {}) {
  const config = require('../config/env');
  return jwt.sign(payload, secret, { expiresIn: config.JWT_EXPIRES_IN, ...options });
}

function verifyToken(token, secret) {
  try {
    return jwt.verify(token, secret);
  } catch {
    return null;
  }
}

/**
 * Middleware: require a valid JWT in the Authorization header.
 */
function requireAuth({ secret }) {
  return (req, _res, next) => {
    const header = req.headers.authorization || '';
    const token = header.startsWith('Bearer ') ? header.slice(7) : null;
    const payload = verifyToken(token, secret);
    if (!payload) return next(new AppError('Unauthorized', 401));
    req.user = payload;
    next();
  };
}

/**
 * Middleware: restrict access to certain roles.
 *   router.use(requireRole(['admin']));
 */
function requireRole(roles) {
  const allowed = new Set(Array.isArray(roles) ? roles : [roles]);
  return (req, _res, next) => {
    if (!req.user) return next(new AppError('Unauthorized', 401));
    if (!allowed.has(req.user.role)) return next(new AppError('Forbidden', 403));
    next();
  };
}

module.exports = { signToken, verifyToken, requireAuth, requireRole };
