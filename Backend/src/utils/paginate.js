/**
 * Pagination utility — offset-based with sensible defaults.
 *
 * Works with any Mongoose query. Reads `page` and `limit` from
 * req.query so callers don't have to parse or validate them.
 *
 * Example:
 *   const { paginate, paginationParams } = require('../utils/paginate');
 *   router.get('/items', asyncHandler(async (req, res) => {
 *     const { page, limit, skip } = paginationParams(req);
 *     const [rows, total] = await Promise.all([
 *       Model.find(filter).skip(skip).limit(limit).lean(),
 *       Model.countDocuments(filter),
 *     ]);
 *     res.json(paginate(rows, total, page, limit));
 *   }));
 *
 * The response shape:
 *   { data: [...], pagination: { page, limit, total, totalPages, hasNext, hasPrev } }
 */

const MAX_LIMIT = 200;
const DEFAULT_LIMIT = 50;

function paginationParams(req) {
  const page  = Math.max(1, parseInt(req.query.page, 10) || 1);
  const limit = Math.min(MAX_LIMIT, Math.max(1, parseInt(req.query.limit, 10) || DEFAULT_LIMIT));
  const skip  = (page - 1) * limit;
  return { page, limit, skip };
}

function paginate(data, total, page, limit) {
  const totalPages = Math.ceil(total / limit) || 1;
  return {
    data,
    pagination: {
      page,
      limit,
      total,
      totalPages,
      hasNext: page < totalPages,
      hasPrev: page > 1,
    },
  };
}

module.exports = { paginationParams, paginate };
