module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.js'],
  coverageDirectory: 'coverage',
  coveragePathIgnorePatterns: ['/node_modules/', '/seed/', '/scripts/'],
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/seed/**',
    '!src/scripts/**',
  ],
  coverageThreshold: {
    global: {
      branches: 60,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
  testTimeout: 15000,
  verbose: true,
  forceExit: true,
  detectOpenHandles: true,
};
