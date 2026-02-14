/**
 * Environment configuration — validated at startup with Zod.
 *
 * If any required variable is missing or malformed the process
 * prints a friendly error and exits immediately. This prevents
 * "it worked on my machine" surprises in production.
 */
const { z } = require('zod');
require('dotenv').config();

const envSchema = z.object({
  PORT:          z.coerce.number().default(5001),
  NODE_ENV:      z.enum(['development', 'production', 'test']).default('development'),
  MONGODB_URI:   z.string().min(1, 'MONGODB_URI is required'),
  AUTH_SECRET:   z.string().min(8, 'AUTH_SECRET must be ≥ 8 characters'),
  CORS_ORIGIN:   z.string().default('http://localhost:3000'),

  // JWT
  JWT_EXPIRES_IN:         z.string().default('2h'),
  JWT_REFRESH_EXPIRES_IN: z.string().default('7d'),

  // Rate-limiting
  RATE_LIMIT_WINDOW_MS:  z.coerce.number().default(15 * 60 * 1000),
  RATE_LIMIT_MAX:        z.coerce.number().default(300),

  // Request timeout
  REQUEST_TIMEOUT_MS: z.coerce.number().default(30_000),

  // Redis (optional — caching is graceful-degrading)
  REDIS_URL: z.string().optional(),

  // Monitoring (optional)
  SENTRY_DSN: z.string().optional(),

  // Logging
  LOG_LEVEL: z.enum(['error', 'warn', 'info', 'http', 'debug']).default('info'),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error('❌  Invalid environment variables:');
  parsed.error.issues.forEach((issue) => {
    console.error(`   ${issue.path.join('.')} — ${issue.message}`);
  });
  process.exit(1);
}

const config = Object.freeze(parsed.data);

module.exports = config;
