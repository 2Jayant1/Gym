/**
 * Test helpers — factories, mocks, and utilities for unit and integration tests.
 */

/**
 * Create a mock repository with all BaseRepository methods stubbed.
 */
function createMockRepo(overrides = {}) {
  return {
    findById: jest.fn().mockResolvedValue(null),
    findOne: jest.fn().mockResolvedValue(null),
    find: jest.fn().mockResolvedValue([]),
    create: jest.fn().mockImplementation((data) => Promise.resolve({ _id: 'mock-id', ...data })),
    update: jest.fn().mockResolvedValue(null),
    updateById: jest.fn().mockResolvedValue(null),
    updateOne: jest.fn().mockResolvedValue(null),
    updateMany: jest.fn().mockResolvedValue({ modifiedCount: 0 }),
    deleteById: jest.fn().mockResolvedValue(null),
    deleteOne: jest.fn().mockResolvedValue(null),
    softDelete: jest.fn().mockResolvedValue(null),
    restore: jest.fn().mockResolvedValue(null),
    count: jest.fn().mockResolvedValue(0),
    estimatedCount: jest.fn().mockResolvedValue(0),
    paginate: jest.fn().mockResolvedValue({ data: [], pagination: {} }),
    aggregate: jest.fn().mockResolvedValue([]),
    bulkCreate: jest.fn().mockResolvedValue([]),
    bulkUpdate: jest.fn().mockResolvedValue({}),
    distinct: jest.fn().mockResolvedValue([]),
    withTransaction: jest.fn().mockImplementation((fn) => fn()),
    ...overrides,
  };
}

/**
 * Create the full mock repos object that services expect.
 */
function createMockRepos(overrides = {}) {
  const repoNames = [
    'user',
    'membershipPlan',
    'attendance',
    'accessLog',
    'invoice',
    'payment',
    'trainer',
    'trainingSession',
    'workoutPlan',
    'exercise',
    'workoutExercise',
    'equipment',
    'equipmentMaintenance',
    'fitnessClass',
    'classSchedule',
    'classRegistration',
    'supportCategory',
    'supportTicket',
    'ticketReply',
    'memberProgress',
    'gymUsageStat',
    'notification',
    'auditLog',
    'refreshToken',
  ];

  const repos = {};
  for (const name of repoNames) {
    repos[name] = createMockRepo(overrides[name] || {});
  }
  return repos;
}

/**
 * Default test config object mimicking env.js output.
 */
const mockConfig = {
  PORT: 5001,
  NODE_ENV: 'test',
  MONGODB_URI: 'mongodb://localhost:27017/gym_test',
  AUTH_SECRET: 'test-secret-min-8-chars',
  CORS_ORIGIN: 'http://localhost:3000',
  JWT_EXPIRES_IN: '2h',
  JWT_REFRESH_EXPIRES_IN: '7d',
  REFRESH_TOKEN_TTL_DAYS: 7,
  RATE_LIMIT_WINDOW_MS: 900000,
  RATE_LIMIT_MAX: 300,
  REQUEST_TIMEOUT_MS: 30000,
  LOG_LEVEL: 'error',
};

/**
 * User factory — generates a user-like object.
 */
function buildUser(overrides = {}) {
  return {
    _id: 'user-id-1',
    id: 'user-id-1',
    username: 'testuser',
    email: 'test@example.com',
    name: 'Test User',
    role: 'member',
    status: 'active',
    password: '$2a$12$hashedpassword',
    phone: '1234567890',
    joinDate: new Date('2024-01-01'),
    membershipPlan: { _id: 'plan-id', name: 'Basic' },
    heightCm: 175,
    weightKg: 70,
    fitnessGoals: 'Stay healthy',
    medicalConditions: '',
    save: jest.fn().mockResolvedValue(true),
    ...overrides,
  };
}

function buildTrainer(overrides = {}) {
  return {
    _id: 'trainer-id-1',
    user: 'user-id-1',
    specialization: 'Strength',
    certification: 'ACE',
    hourlyRate: 50,
    experienceYears: 5,
    bio: 'Experienced trainer',
    status: 'active',
    ...overrides,
  };
}

module.exports = {
  createMockRepo,
  createMockRepos,
  mockConfig,
  buildUser,
  buildTrainer,
};
