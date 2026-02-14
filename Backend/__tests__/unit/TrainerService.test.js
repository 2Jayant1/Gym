/**
 * Unit tests — TrainerService
 */
const TrainerService = require('../../src/services/TrainerService');
const { createMockRepos, buildTrainer } = require('../helpers');

describe('TrainerService', () => {
  let trainerService;
  let repos;

  beforeEach(() => {
    repos = createMockRepos();
    trainerService = new TrainerService({ repos });
  });

  describe('resolveTrainerId', () => {
    it('should return trainer _id', async () => {
      repos.trainer.findOne.mockResolvedValue({ _id: 'trainer-1' });
      const id = await trainerService.resolveTrainerId('user-1');
      expect(id).toBe('trainer-1');
    });

    it('should throw if trainer not found', async () => {
      repos.trainer.findOne.mockResolvedValue(null);
      await expect(trainerService.resolveTrainerId('user-1')).rejects.toThrow('Trainer profile not found');
    });
  });

  describe('listSessions', () => {
    it('should return sessions with total', async () => {
      repos.trainingSession.find.mockResolvedValue([{ _id: 's1', user: { name: 'John', email: 'j@e.com' } }]);
      repos.trainingSession.count.mockResolvedValue(1);

      const result = await trainerService.listSessions('trainer-1', { skip: 0, limit: 10 });
      expect(result.rows).toHaveLength(1);
      expect(result.total).toBe(1);
      expect(result.rows[0]).toHaveProperty('member_name', 'John');
    });
  });

  describe('createSession', () => {
    it('should create a session', async () => {
      repos.trainingSession.create.mockResolvedValue({ _id: 's1' });
      const result = await trainerService.createSession('trainer-1', {
        userId: 'u1',
        sessionDate: '2024-01-15',
        startTime: '10:00',
        endTime: '11:00',
        durationMinutes: 60,
      });
      expect(result._id).toBe('s1');
    });

    it('should throw on missing fields', async () => {
      await expect(trainerService.createSession('trainer-1', {})).rejects.toThrow(
        'userId, sessionDate, startTime, endTime, durationMinutes are required',
      );
    });
  });

  describe('createClass', () => {
    it('should create a class', async () => {
      repos.fitnessClass.create.mockResolvedValue({ _id: 'c1', name: 'Yoga' });
      const result = await trainerService.createClass('trainer-1', { name: 'Yoga', durationMinutes: 60 });
      expect(result.name).toBe('Yoga');
    });

    it('should throw on missing fields', async () => {
      await expect(trainerService.createClass('trainer-1', {})).rejects.toThrow(
        'name and durationMinutes are required',
      );
    });
  });

  describe('getProfile', () => {
    it('should return trainer profile', async () => {
      repos.trainer.findById.mockResolvedValue(buildTrainer());
      const result = await trainerService.getProfile('trainer-1');
      expect(result).toHaveProperty('specialization', 'Strength');
    });

    it('should throw if not found', async () => {
      repos.trainer.findById.mockResolvedValue(null);
      await expect(trainerService.getProfile('nope')).rejects.toThrow('Trainer not found');
    });
  });

  describe('listMembers', () => {
    it('should return members with sessions', async () => {
      repos.trainingSession.distinct.mockResolvedValue(['u1', 'u2']);
      repos.user.find.mockResolvedValue([
        { _id: 'u1', name: 'User 1' },
        { _id: 'u2', name: 'User 2' },
      ]);

      const result = await trainerService.listMembers('trainer-1');
      expect(result).toHaveLength(2);
    });
  });
});
