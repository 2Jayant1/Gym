/**
 * Application logger powered by Winston.
 *
 * • Development → pretty-print with colors to the console.
 * • Production  → JSON lines for log aggregators (ELK, Datadog…).
 *
 * Usage:
 *   const logger = require('./logger');
 *   logger.info('Server started', { port: 5001 });
 *   logger.error('DB failed', { err });
 */
const { createLogger, format, transports } = require('winston');
const config = require('./env');

const isDev = config.NODE_ENV === 'development';

const logger = createLogger({
  level: config.LOG_LEVEL,
  defaultMeta: { service: 'gym-api' },
  format: format.combine(
    format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    format.errors({ stack: true }),
    isDev
      ? format.combine(
          format.colorize(),
          format.printf(({ timestamp, level, message, ...meta }) => {
            const extra = Object.keys(meta).length > 1 ? ` ${JSON.stringify(meta)}` : '';
            return `${timestamp} ${level}: ${message}${extra}`;
          }),
        )
      : format.json(),
  ),
  transports: [new transports.Console()],
});

module.exports = logger;
