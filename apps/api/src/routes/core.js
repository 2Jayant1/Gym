/**
 * Core routes — auth, registration, public endpoints.
 *
 * Architecture: Routes → Service layer (no direct model access).
 *
 * ✓ asyncHandler   → no try/catch needed
 * ✓ AppError       → throw instead of res.status()
 * ✓ Zod validation → validate() middleware
 * ✓ Pagination     → on /users list
 * ✓ Auth limiter   → on login/register
 * ✓ JWT refresh    → token rotation endpoint
 */
const express = require('express');
const { loginSchema, registerSchema, validate } = require('../utils/validate');
const { asyncHandler } = require('../utils/AppError');
const { paginationParams, paginate } = require('../utils/paginate');

function coreRoutes({ services, models, requireAuth, requireRole, authLimiter }) {
  const router = express.Router();

  /* ─── Auth — Login ──────────────────────────────────── */
  router.post('/auth/login', authLimiter, validate(loginSchema), asyncHandler(async (req, res) => {
    const result = await services.auth.login(req.validated, { ip: req.ip, userAgent: req.headers['user-agent'] });
    res.json(result);
  }));

  /* ─── Auth — Register ───────────────────────────────── */
  router.post('/auth/register', authLimiter, validate(registerSchema), asyncHandler(async (req, res) => {
    const result = await services.auth.register(req.validated, { ip: req.ip, userAgent: req.headers['user-agent'] });
    res.status(201).json(result);
  }));

  /* ─── Auth — Refresh Token (rotation) ───────────────── */
  router.post('/auth/refresh', asyncHandler(async (req, res) => {
    const { refreshToken } = req.body || {};
    const result = await services.auth.refresh(refreshToken, { ip: req.ip, userAgent: req.headers['user-agent'] });
    res.json(result);
  }));

  /* ─── Current user (from JWT payload) ───────────────── */
  router.get('/me', requireAuth, (req, res) => {
    res.json({ user: req.user });
  });

  /* ─── Stats (dashboard counts) ──────────────────────── */
  router.get('/stats', requireAuth, asyncHandler(async (_req, res) => {
    const stats = await services.user.getStats();
    res.json(stats);
  }));

  /* ─── Users list (admin/trainer, paginated) ─────────── */
  router.get('/users', requireAuth, requireRole(['admin', 'trainer']), asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { mapped, total } = await services.user.listUsers({ page, limit, skip });
    res.json(paginate(mapped, total, page, limit));
  }));

  /* ─── Me profile (detailed) ─────────────────────────── */
  router.get('/users/me', requireAuth, asyncHandler(async (req, res) => {
    const profile = await services.user.getProfile(req.user.id);
    res.json(profile);
  }));

  /* ─── Membership plans (public, cacheable) ──────────── */
  router.get('/membership-plans', asyncHandler(async (_req, res) => {
    const rows = await models.MembershipPlan.find().lean();
    res.json(rows);
  }));

  router.get('/membership-plans/:id', asyncHandler(async (req, res) => {
    const { AppError } = require('../utils/AppError');
    const row = await models.MembershipPlan.findById(req.params.id).lean();
    if (!row) throw AppError.notFound('Plan');
    res.json(row);
  }));

  return router;
}

module.exports = coreRoutes;
