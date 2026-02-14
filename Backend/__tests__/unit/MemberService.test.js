/**
 * Unit tests — MemberService
 *
 * Tests attendance, progress, classes, tickets, and notifications.
 */
const bcrypt = require('bcryptjs');
const MemberService = require('../../src/services/MemberService');
const { createMockRepos, buildUser } = require('../helpers');

describe('MemberService', () => {
  let memberService;
  let repos;

  beforeEach(() => {
    repos = createMockRepos();
    memberService = new MemberService({ repos });
  });

  /* ═══════════════ ATTENDANCE ═══════════════ */

  describe('checkIn', () => {
    it('should create attendance record when no existing check-in', async () => {
      repos.attendance.findOne.mockResolvedValue(null);
      repos.attendance.create.mockResolvedValue({ _id: 'att-1', user: 'u1', checkIn: new Date() });
      repos.accessLog.create.mockResolvedValue({});

      const result = await memberService.checkIn('u1');
      expect(result).toHaveProperty('_id', 'att-1');
      expect(repos.attendance.create).toHaveBeenCalled();
    });

    it('should throw if already checked in', async () => {
      repos.attendance.findOne.mockResolvedValue({ _id: 'existing' });
      await expect(memberService.checkIn('u1')).rejects.toThrow('Already checked in');
    });
  });

  describe('checkOut', () => {
    it('should update attendance record', async () => {
      const mockRecord = { checkOut: null, save: jest.fn().mockResolvedValue(true) };
      repos.attendance.findOne.mockResolvedValue(mockRecord);
      repos.accessLog.create.mockResolvedValue({});

      const result = await memberService.checkOut('u1');
      expect(mockRecord.save).toHaveBeenCalled();
      expect(mockRecord.checkOut).toBeInstanceOf(Date);
    });

    it('should throw if no active check-in', async () => {
      repos.attendance.findOne.mockResolvedValue(null);
      await expect(memberService.checkOut('u1')).rejects.toThrow('No active check-in found');
    });
  });

  /* ═══════════════ PROGRESS ═══════════════ */

  describe('addProgress', () => {
    it('should create progress record with BMI', async () => {
      repos.user.findById.mockResolvedValue({ heightCm: 180 });
      repos.memberProgress.create.mockResolvedValue({ _id: 'prog-1', bmi: 24.69 });

      const result = await memberService.addProgress('u1', { weightKg: 80 });
      expect(repos.memberProgress.create).toHaveBeenCalled();
      const createArg = repos.memberProgress.create.mock.calls[0][0];
      expect(createArg.bmi).toBeDefined();
      expect(createArg.user).toBe('u1');
    });
  });

  /* ═══════════════ PROFILE ═══════════════ */

  describe('changePassword', () => {
    it('should change password with correct current password', async () => {
      const hashedPw = bcrypt.hashSync('OldPass1', 12);
      const user = buildUser({ password: hashedPw });
      repos.user.findById.mockResolvedValue(user);

      const result = await memberService.changePassword('u1', {
        currentPassword: 'OldPass1',
        newPassword: 'NewPass1',
      });
      expect(result.ok).toBe(true);
    });

    it('should throw on wrong current password', async () => {
      const hashedPw = bcrypt.hashSync('CorrectPass1', 12);
      repos.user.findById.mockResolvedValue(buildUser({ password: hashedPw }));

      await expect(
        memberService.changePassword('u1', { currentPassword: 'Wrong', newPassword: 'New1' }),
      ).rejects.toThrow('Current password is incorrect');
    });
  });

  /* ═══════════════ CLASSES ═══════════════ */

  describe('registerForClass', () => {
    it('should register when class has capacity', async () => {
      repos.classSchedule.findById.mockResolvedValue({
        fitnessClass: { maxParticipants: 20 },
      });
      repos.classRegistration.count.mockResolvedValue(5);
      repos.classRegistration.create.mockResolvedValue({ _id: 'reg-1' });

      const result = await memberService.registerForClass('u1', 'sched-1');
      expect(result).toHaveProperty('_id', 'reg-1');
    });

    it('should throw when class is full', async () => {
      repos.classSchedule.findById.mockResolvedValue({
        fitnessClass: { maxParticipants: 10 },
      });
      repos.classRegistration.count.mockResolvedValue(10);

      await expect(memberService.registerForClass('u1', 'sched-1')).rejects.toThrow('Class is full');
    });

    it('should throw when classScheduleId is missing', async () => {
      await expect(memberService.registerForClass('u1', null)).rejects.toThrow('classScheduleId is required');
    });
  });

  /* ═══════════════ TICKETS ═══════════════ */

  describe('createTicket', () => {
    it('should create a ticket', async () => {
      repos.supportTicket.create.mockResolvedValue({ _id: 'ticket-1', title: 'Help' });

      const result = await memberService.createTicket('u1', {
        title: 'Help',
        message: 'I need help',
      });
      expect(result.title).toBe('Help');
    });
  });

  /* ═══════════════ NOTIFICATIONS ═══════════════ */

  describe('getUnreadCount', () => {
    it('should return unread count', async () => {
      repos.notification.count.mockResolvedValue(5);
      const count = await memberService.getUnreadCount('u1');
      expect(count).toBe(5);
    });
  });

  describe('markAllRead', () => {
    it('should mark all notifications as read', async () => {
      repos.notification.updateMany.mockResolvedValue({ modifiedCount: 3 });
      const result = await memberService.markAllRead('u1');
      expect(result.ok).toBe(true);
    });
  });
});
