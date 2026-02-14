/**
 * Feature Flags — simple in-memory flag system.
 *
 * Placeholder for a real feature flag service (LaunchDarkly, Unleash, etc.).
 * Flags are loaded from environment or a static config. Services/routes
 * check flags before enabling experimental features.
 *
 * Usage:
 *   const flags = require('./featureFlags');
 *   if (flags.isEnabled('NEW_DASHBOARD')) { ... }
 */

// Default flags — override via FEATURE_FLAGS env (comma-separated)
const DEFAULT_FLAGS = {
  REDIS_CACHE:       false,   // Enable Redis caching layer
  ML_RECOMMENDATIONS:false,   // Enable ML-based workout recommendations
  AI_CHATBOT:        false,   // Enable AI gym chatbot
  ADVANCED_ANALYTICS:false,   // Enable advanced analytics dashboard
  A_B_TESTING:       false,   // Enable A/B test framework
};

// Parse runtime overrides from env: FEATURE_FLAGS=REDIS_CACHE,AI_CHATBOT
const envFlags = (process.env.FEATURE_FLAGS || '').split(',').map(s => s.trim()).filter(Boolean);
const flags = { ...DEFAULT_FLAGS };
for (const f of envFlags) flags[f] = true;

module.exports = {
  /** Check if a flag is enabled */
  isEnabled(flag) {
    return flags[flag] === true;
  },

  /** Get all flags (for /health or admin UI) */
  getAll() {
    return { ...flags };
  },

  /** Runtime toggle (for testing or admin override) */
  set(flag, value) {
    flags[flag] = !!value;
  },
};
