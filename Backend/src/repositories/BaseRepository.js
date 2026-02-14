/**
 * Base Repository — reusable CRUD operations for any Mongoose model.
 *
 * Every domain repository extends this and inherits:
 *   findById, findOne, find, create, update, softDelete, restore,
 *   count, estimatedCount, paginate, aggregate, bulkCreate, bulkUpdate,
 *   withTransaction
 *
 * This keeps Mongoose calls out of services entirely.
 */
class BaseRepository {
  /** @param {import('mongoose').Model} model */
  constructor(model) {
    this.model = model;
  }

  /* ─── Read ────────────────────────────────────────────── */

  findById(id, options = {}) {
    let q = this.model.findById(id);
    if (options.select) q = q.select(options.select);
    if (options.populate) q = q.populate(options.populate);
    return options.lean !== false ? q.lean() : q;
  }

  findOne(filter, options = {}) {
    let q = this.model.findOne(filter);
    if (options.select) q = q.select(options.select);
    if (options.populate) q = q.populate(options.populate);
    return options.lean !== false ? q.lean() : q;
  }

  find(filter = {}, options = {}) {
    let q = this.model.find(filter);
    if (options.select)  q = q.select(options.select);
    if (options.populate) q = q.populate(options.populate);
    if (options.sort)    q = q.sort(options.sort);
    if (options.skip)    q = q.skip(options.skip);
    if (options.limit)   q = q.limit(options.limit);
    return options.lean !== false ? q.lean() : q;
  }

  /* ─── Write ───────────────────────────────────────────── */

  create(data, options = {}) {
    return this.model.create(data, options);
  }

  update(id, data, options = {}) {
    return this.model.findByIdAndUpdate(id, data, {
      new: true,
      runValidators: true,
      ...options,
    });
  }

  updateOne(filter, data, options = {}) {
    return this.model.findOneAndUpdate(filter, data, {
      new: true,
      runValidators: true,
      ...options,
    });
  }

  updateMany(filter, data) {
    return this.model.updateMany(filter, data);
  }

  deleteById(id) {
    return this.model.findByIdAndDelete(id);
  }

  deleteOne(filter) {
    return this.model.findOneAndDelete(filter);
  }

  /* ─── Soft Delete ─────────────────────────────────────── */

  softDelete(id) {
    return this.update(id, { deletedAt: new Date() });
  }

  restore(id) {
    return this.update(id, { deletedAt: null });
  }

  /* ─── Count ───────────────────────────────────────────── */

  count(filter = {}) {
    return this.model.countDocuments(filter);
  }

  estimatedCount() {
    return this.model.estimatedDocumentCount();
  }

  /* ─── Pagination helper ───────────────────────────────── */

  async paginate(filter, { page = 1, limit = 50, sort = { _id: -1 }, select, populate } = {}) {
    const skip = (page - 1) * limit;
    const [data, total] = await Promise.all([
      this.find(filter, { sort, skip, limit, select, populate }),
      this.count(filter),
    ]);
    const totalPages = Math.ceil(total / limit) || 1;
    return {
      data,
      pagination: { page, limit, total, totalPages, hasNext: page < totalPages, hasPrev: page > 1 },
    };
  }

  /* ─── Aggregate ───────────────────────────────────────── */

  aggregate(pipeline) {
    return this.model.aggregate(pipeline);
  }

  /* ─── Bulk operations ─────────────────────────────────── */

  bulkCreate(docs) {
    return this.model.insertMany(docs, { ordered: false });
  }

  bulkUpdate(operations) {
    return this.model.bulkWrite(operations);
  }

  /* ─── Transaction helper ──────────────────────────────── */

  async withTransaction(fn) {
    const session = await this.model.startSession();
    try {
      session.startTransaction();
      const result = await fn(session);
      await session.commitTransaction();
      return result;
    } catch (err) {
      await session.abortTransaction();
      throw err;
    } finally {
      session.endSession();
    }
  }

  /* ─── Distinct ────────────────────────────────────────── */

  distinct(field, filter = {}) {
    return this.model.distinct(field, filter);
  }
}

module.exports = BaseRepository;
