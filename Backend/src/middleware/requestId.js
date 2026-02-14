/**
 * Request ID middleware — attaches a unique ID to every request.
 *
 * Respects incoming X-Request-Id headers (from load balancers)
 * and exposes the ID back on the response for traceability.
 */
const crypto = require('crypto');

function requestId(req, res, next) {
  const id = req.headers['x-request-id'] || crypto.randomUUID();
  req.id = id;
  res.setHeader('X-Request-Id', id);
  next();
}

module.exports = requestId;
