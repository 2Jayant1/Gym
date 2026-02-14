/**
 * Unit tests — AuthService
 *
 * Tests login, register, refresh, and token operations
 * WITHOUT a real database (all repos are mocked).
 */
const bcrypt = require('bcryptjs');
const AuthService = require('../../src/services/AuthService');
const { createMockRepos, mockConfig, buildUser } = require('../helpers');

describe('AuthService', () => {
  let authService;
  let repos;

  beforeEach(() => {
    repos = createMockRepos();
    authService = new AuthService({ repos, config: mockConfig });
  });

  /* ═══════════════ signAccessToken / signRefreshToken ═══════════════ */

  describe('signAccessToken', () => {
    it('should return a JWT string', () => {
      const token = authService.signAccessToken({ id: '123', role: 'member' });
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3); // header.payload.signature
    });
  });

  describe('signRefreshToken', () => {
    it('should return a JWT string', () => {
      const token = authService.signRefreshToken({ id: '123' });
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3);
    });
  });

  describe('verifyToken', () => {
    it('should verify a valid token', () => {
      const token = authService.signAccessToken({ id: '123', role: 'admin' });
      const payload = authService.verifyToken(token);
      expect(payload).toBeDefined();
      expect(payload.id).toBe('123');
      expect(payload.role).toBe('admin');
    });

    it('should return null for an invalid token', () => {
      const payload = authService.verifyToken('invalid.token.here');
      expect(payload).toBeNull();
    });
  });

  /* ═══════════════ login ═══════════════ */

  describe('login', () => {
    it('should login successfully with email and password', async () => {
      const hashedPw = bcrypt.hashSync('Password1', 12);
      const mockUser = buildUser({ password: hashedPw });
      repos.user.findOne.mockResolvedValue(mockUser);

      const result = await authService.login({ email: 'test@example.com', password: 'Password1' });

      expect(result).toHaveProperty('token');
      expect(result).toHaveProperty('refreshToken');
      expect(result.user.email).toBe('test@example.com');
      expect(repos.user.findOne).toHaveBeenCalledWith(
        { email: 'test@example.com' },
        expect.objectContaining({ select: '+password' }),
      );
    });

    it('should login successfully with username', async () => {
      const hashedPw = bcrypt.hashSync('Password1', 12);
      const mockUser = buildUser({ password: hashedPw });
      repos.user.findOne.mockResolvedValue(mockUser);

      const result = await authService.login({ username: 'testuser', password: 'Password1' });

      expect(result).toHaveProperty('token');
      expect(repos.user.findOne).toHaveBeenCalledWith(
        { username: 'testuser' },
        expect.objectContaining({ select: '+password' }),
      );
    });

    it('should throw on missing credentials', async () => {
      await expect(authService.login({})).rejects.toThrow('email or username and password are required');
    });

    it('should throw on invalid user', async () => {
      repos.user.findOne.mockResolvedValue(null);
      await expect(authService.login({ email: 'nope@example.com', password: 'Password1' })).rejects.toThrow(
        'Invalid credentials',
      );
    });

    it('should throw on wrong password', async () => {
      const hashedPw = bcrypt.hashSync('CorrectPass1', 12);
      const mockUser = buildUser({ password: hashedPw });
      repos.user.findOne.mockResolvedValue(mockUser);

      await expect(authService.login({ email: 'test@example.com', password: 'WrongPass1' })).rejects.toThrow(
        'Invalid credentials',
      );
    });
  });

  /* ═══════════════ register ═══════════════ */

  describe('register', () => {
    const validData = {
      name: 'New User',
      username: 'newuser',
      email: 'new@example.com',
      password: 'Password1',
    };

    it('should register a new user successfully', async () => {
      repos.user.findOne.mockResolvedValue(null); // no existing user
      repos.membershipPlan.findOne.mockResolvedValue({ _id: 'plan-1' });
      repos.user.create.mockResolvedValue(
        buildUser({
          name: 'New User',
          username: 'newuser',
          email: 'new@example.com',
        }),
      );
      repos.notification.create.mockResolvedValue({});

      const result = await authService.register(validData);

      expect(result).toHaveProperty('token');
      expect(result).toHaveProperty('refreshToken');
      expect(result.user.name).toBe('New User');
      expect(repos.user.create).toHaveBeenCalled();
    });

    it('should throw on duplicate email', async () => {
      repos.user.findOne.mockResolvedValue({ email: 'new@example.com' });

      await expect(authService.register(validData)).rejects.toThrow('A user with that email already exists');
    });

    it('should throw on duplicate username', async () => {
      repos.user.findOne.mockResolvedValue({ email: 'other@example.com', username: 'newuser' });

      await expect(authService.register(validData)).rejects.toThrow('A user with that username already exists');
    });
  });

  /* ═══════════════ refresh ═══════════════ */

  describe('refresh', () => {
    it('should return new tokens with a valid refresh token', async () => {
      const tokenFamily = 'family-1';
      const refreshToken = authService.signRefreshToken({ id: 'user-id-1', tokenFamily, jti: 'jti-1' });
      const tokenHash = authService.hashToken(refreshToken);
      repos.refreshToken.findOne.mockResolvedValue({
        _id: 'rt-1',
        user: 'user-id-1',
        tokenFamily,
        tokenHash,
        expiresAt: new Date(Date.now() + 3600_000),
      });
      repos.user.findById.mockResolvedValue(buildUser());

      const result = await authService.refresh(refreshToken, { ip: '1.1.1.1', userAgent: 'jest' });

      expect(result).toHaveProperty('token');
      expect(result).toHaveProperty('refreshToken');
      expect(result.refreshToken).not.toBe(refreshToken); // rotated
    });

    it('should throw on invalid refresh token', async () => {
      await expect(authService.refresh('bad-token')).rejects.toThrow('Invalid refresh token');
    });

    it('should throw on suspended account', async () => {
      const refreshToken = authService.signRefreshToken({ id: 'user-id-1', tokenFamily: 'abc', jti: 'jti-1' });
      const tokenHash = authService.hashToken(refreshToken);
      repos.refreshToken.findOne.mockResolvedValue({
        _id: 'rt-1',
        user: 'user-id-1',
        tokenFamily: 'abc',
        tokenHash,
        expiresAt: new Date(Date.now() + 3600_000),
      });
      repos.user.findById.mockResolvedValue(buildUser({ status: 'suspended' }));

      await expect(authService.refresh(refreshToken)).rejects.toThrow('Account not found or suspended');
    });
  });
});
