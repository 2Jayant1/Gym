module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.js'],
  coverageDirectory: 'coverage',
  coveragePathIgnorePatterns: ['/node_modules/', '/seed/', '/scripts/'],
  collectCoverageFrom: ['src/**/*.js', '!src/seed/**', '!src/scripts/**'],
  coverageThreshold: {
    global: {
      branches: 15,
      functions: 15,
      lines: 15,
      statements: 15,
    },
  },
  testTimeout: 15000,
  verbose: true,
  forceExit: true,
  detectOpenHandles: true,
};
