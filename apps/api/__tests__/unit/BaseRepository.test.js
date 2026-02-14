/**
 * Unit tests — BaseRepository
 *
 * Tests generic CRUD operations with a mock Mongoose model.
 */
const BaseRepository = require('../../src/repositories/BaseRepository');

// Create a mock Mongoose model
function createMockModel() {
  const chainable = {
    select: jest.fn().mockReturnThis(),
    populate: jest.fn().mockReturnThis(),
    sort: jest.fn().mockReturnThis(),
    skip: jest.fn().mockReturnThis(),
    limit: jest.fn().mockReturnThis(),
    lean: jest.fn().mockResolvedValue(null),
  };

  const model = {
    findById: jest.fn().mockReturnValue({ ...chainable }),
    findOne: jest.fn().mockReturnValue({ ...chainable }),
    find: jest.fn().mockReturnValue({ ...chainable, lean: jest.fn().mockResolvedValue([]) }),
    create: jest.fn().mockResolvedValue({ _id: 'new-id' }),
    findByIdAndUpdate: jest.fn().mockResolvedValue({ _id: 'updated-id' }),
    findOneAndUpdate: jest.fn().mockResolvedValue({ _id: 'updated-id' }),
    updateMany: jest.fn().mockResolvedValue({ modifiedCount: 1 }),
    findByIdAndDelete: jest.fn().mockResolvedValue({ _id: 'deleted-id' }),
    findOneAndDelete: jest.fn().mockResolvedValue({ _id: 'deleted-id' }),
    countDocuments: jest.fn().mockResolvedValue(0),
    estimatedDocumentCount: jest.fn().mockResolvedValue(0),
    aggregate: jest.fn().mockResolvedValue([]),
    insertMany: jest.fn().mockResolvedValue([]),
    bulkWrite: jest.fn().mockResolvedValue({}),
    distinct: jest.fn().mockResolvedValue([]),
    startSession: jest.fn().mockResolvedValue({
      startTransaction: jest.fn(),
      commitTransaction: jest.fn(),
      abortTransaction: jest.fn(),
      endSession: jest.fn(),
    }),
  };

  return model;
}

describe('BaseRepository', () => {
  let model;
  let repo;

  beforeEach(() => {
    model = createMockModel();
    repo = new BaseRepository(model);
  });

  describe('create', () => {
    it('should delegate to model.create', async () => {
      await repo.create({ name: 'Test' });
      expect(model.create).toHaveBeenCalledWith({ name: 'Test' }, {});
    });
  });

  describe('deleteById', () => {
    it('should delegate to findByIdAndDelete', async () => {
      await repo.deleteById('id-1');
      expect(model.findByIdAndDelete).toHaveBeenCalledWith('id-1');
    });
  });

  describe('count', () => {
    it('should delegate to countDocuments', async () => {
      model.countDocuments.mockResolvedValue(42);
      const result = await repo.count({ status: 'active' });
      expect(result).toBe(42);
      expect(model.countDocuments).toHaveBeenCalledWith({ status: 'active' });
    });
  });

  describe('estimatedCount', () => {
    it('should delegate to estimatedDocumentCount', async () => {
      model.estimatedDocumentCount.mockResolvedValue(100);
      const result = await repo.estimatedCount();
      expect(result).toBe(100);
    });
  });

  describe('aggregate', () => {
    it('should delegate to model.aggregate', async () => {
      const pipeline = [{ $group: { _id: null, total: { $sum: 1 } } }];
      await repo.aggregate(pipeline);
      expect(model.aggregate).toHaveBeenCalledWith(pipeline);
    });
  });

  describe('distinct', () => {
    it('should delegate to model.distinct', async () => {
      model.distinct.mockResolvedValue(['a', 'b']);
      const result = await repo.distinct('field', { active: true });
      expect(result).toEqual(['a', 'b']);
    });
  });

  describe('bulkCreate', () => {
    it('should delegate to insertMany', async () => {
      const docs = [{ name: 'A' }, { name: 'B' }];
      await repo.bulkCreate(docs);
      expect(model.insertMany).toHaveBeenCalledWith(docs, { ordered: false });
    });
  });

  describe('withTransaction', () => {
    it('should start, commit, and end a session', async () => {
      const session = await model.startSession();
      const fn = jest.fn().mockResolvedValue('result');

      const result = await repo.withTransaction(fn);

      expect(model.startSession).toHaveBeenCalled();
      expect(fn).toHaveBeenCalledWith(session);
    });
  });
});
