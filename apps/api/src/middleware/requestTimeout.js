/**
 * Request timeout middleware.
 *
 * Sends a 408 if the handler doesn't respond within the limit.
 * Default: 30 seconds (configurable via TIMEOUT_MS env var).
 */
function requestTimeout(ms = 30_000) {
  return (req, res, next) => {
    const timer = setTimeout(() => {
      if (!res.headersSent) {
        res.status(408).json({ error: 'Request timed out' });
      }
    }, ms);

    // Clear timer when the response finishes
    res.on('finish', () => clearTimeout(timer));
    res.on('close', () => clearTimeout(timer));
    next();
  };
}

module.exports = requestTimeout;
