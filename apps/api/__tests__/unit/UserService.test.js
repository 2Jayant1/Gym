/**
 * Unit tests — UserService
 *
 * Tests profile, password change, user listing, and stats.
 */
const bcrypt = require('bcryptjs');
const UserService = require('../../src/services/UserService');
const { createMockRepos, buildUser } = require('../helpers');

describe('UserService', () => {
  let userService;
  let repos;

  beforeEach(() => {
    repos = createMockRepos();
    userService = new UserService({ repos });
  });

  /* ═══════════════ getProfile ═══════════════ */

  describe('getProfile', () => {
    it('should return a formatted user profile', async () => {
      repos.user.findById.mockResolvedValue(buildUser());

      const profile = await userService.getProfile('user-id-1');

      expect(profile).toHaveProperty('id');
      expect(profile).toHaveProperty('username', 'testuser');
      expect(profile).toHaveProperty('email', 'test@example.com');
      expect(profile).toHaveProperty('plan_name', 'Basic');
    });

    it('should throw if user not found', async () => {
      repos.user.findById.mockResolvedValue(null);
      await expect(userService.getProfile('nope')).rejects.toThrow('User not found');
    });
  });

  /* ═══════════════ updateProfile ═══════════════ */

  describe('updateProfile', () => {
    it('should update and return the user', async () => {
      repos.user.update.mockResolvedValue(buildUser({ name: 'Updated' }));

      const result = await userService.updateProfile('user-id-1', { name: 'Updated' });
      expect(result.name).toBe('Updated');
      expect(repos.user.update).toHaveBeenCalledWith('user-id-1', { name: 'Updated' }, expect.any(Object));
    });

    it('should throw if user not found', async () => {
      repos.user.update.mockResolvedValue(null);
      await expect(userService.updateProfile('nope', {})).rejects.toThrow('User not found');
    });
  });

  /* ═══════════════ changePassword ═══════════════ */

  describe('changePassword', () => {
    it('should change password when current password matches', async () => {
      const hashedPw = bcrypt.hashSync('OldPass1', 12);
      const user = buildUser({ password: hashedPw });
      repos.user.findById.mockResolvedValue(user);

      const result = await userService.changePassword('user-id-1', {
        currentPassword: 'OldPass1',
        newPassword: 'NewPass1',
      });

      expect(result.ok).toBe(true);
      expect(user.save).toHaveBeenCalled();
    });

    it('should throw when current password is incorrect', async () => {
      const hashedPw = bcrypt.hashSync('CorrectPass1', 12);
      repos.user.findById.mockResolvedValue(buildUser({ password: hashedPw }));

      await expect(
        userService.changePassword('user-id-1', {
          currentPassword: 'WrongPass1',
          newPassword: 'NewPass1',
        }),
      ).rejects.toThrow('Current password is incorrect');
    });

    it('should throw if user not found', async () => {
      repos.user.findById.mockResolvedValue(null);
      await expect(userService.changePassword('nope', { currentPassword: 'a', newPassword: 'b' })).rejects.toThrow(
        'User not found',
      );
    });
  });

  /* ═══════════════ listUsers ═══════════════ */

  describe('listUsers', () => {
    it('should return mapped users and total', async () => {
      repos.user.find.mockResolvedValue([buildUser()]);
      repos.user.count.mockResolvedValue(1);

      const result = await userService.listUsers({ page: 1, limit: 10, skip: 0 });

      expect(result.mapped).toHaveLength(1);
      expect(result.total).toBe(1);
      expect(result.mapped[0]).toHaveProperty('id');
      expect(result.mapped[0]).toHaveProperty('plan_name', 'Basic');
    });
  });

  /* ═══════════════ getStats ═══════════════ */

  describe('getStats', () => {
    it('should return counts', async () => {
      repos.user.estimatedCount.mockResolvedValue(100);
      repos.user.count.mockResolvedValue(80);
      repos.membershipPlan.estimatedCount.mockResolvedValue(5);

      const stats = await userService.getStats();

      expect(stats).toEqual({ totalUsers: 100, activeUsers: 80, totalPlans: 5 });
    });
  });
});
