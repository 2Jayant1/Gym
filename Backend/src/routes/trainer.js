/**
 * Trainer routes — sessions, classes, schedules, profile.
 *
 * Architecture: Routes → TrainerService (no direct model access).
 *
 * ✓ asyncHandler everywhere (zero try/catch)
 * ✓ AppError for guard clauses
 * ✓ Pagination on lists
 */
const express = require('express');
const { asyncHandler } = require('../utils/AppError');
const { sessionCreateSchema, classCreateSchema, validate } = require('../utils/validate');
const { paginationParams, paginate } = require('../utils/paginate');

function trainerRoutes({ services, requireAuth, requireRole }) {
  const router = express.Router();

  router.use(requireAuth);
  router.use(requireRole(['trainer']));

  // Middleware — resolves trainer ID for the current user
  const loadTrainer = asyncHandler(async (req, _res, next) => {
    req.trainerId = await services.trainer.resolveTrainerId(req.user.id);
    next();
  });

  /* ═══════════════════════════════════════════════════════════
     SESSIONS
     ═══════════════════════════════════════════════════════════ */
  router.get('/sessions', loadTrainer, asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.trainer.listSessions(req.trainerId, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.post('/sessions', loadTrainer, validate(sessionCreateSchema), asyncHandler(async (req, res) => {
    const row = await services.trainer.createSession(req.trainerId, req.validated);
    res.status(201).json(row);
  }));

  router.patch('/sessions/:id', loadTrainer, asyncHandler(async (req, res) => {
    const row = await services.trainer.updateSession(req.trainerId, req.params.id, req.body);
    res.json(row);
  }));

  router.delete('/sessions/:id', loadTrainer, asyncHandler(async (req, res) => {
    const result = await services.trainer.deleteSession(req.trainerId, req.params.id);
    res.json(result);
  }));

  /* ═══════════════════════════════════════════════════════════
     CLASSES
     ═══════════════════════════════════════════════════════════ */
  router.get('/classes', loadTrainer, asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.trainer.listClasses(req.trainerId, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.post('/classes', loadTrainer, validate(classCreateSchema), asyncHandler(async (req, res) => {
    const row = await services.trainer.createClass(req.trainerId, req.validated);
    res.status(201).json(row);
  }));

  router.patch('/classes/:id', loadTrainer, asyncHandler(async (req, res) => {
    const row = await services.trainer.updateClass(req.trainerId, req.params.id, req.body);
    res.json(row);
  }));

  router.delete('/classes/:id', loadTrainer, asyncHandler(async (req, res) => {
    const result = await services.trainer.deleteClass(req.trainerId, req.params.id);
    res.json(result);
  }));

  /* ─── Schedules ─────────────────────────────────────────── */
  router.get('/classes/:id/schedules', loadTrainer, asyncHandler(async (req, res) => {
    const schedules = await services.trainer.listSchedules(req.trainerId, req.params.id);
    res.json(schedules);
  }));

  router.post('/classes/:id/schedules', loadTrainer, asyncHandler(async (req, res) => {
    const schedule = await services.trainer.createSchedule(req.trainerId, req.params.id, req.body);
    res.status(201).json(schedule);
  }));

  /* ─── Profile ───────────────────────────────────────────── */
  router.get('/profile', loadTrainer, asyncHandler(async (req, res) => {
    const trainer = await services.trainer.getProfile(req.trainerId);
    res.json(trainer);
  }));

  router.patch('/profile', loadTrainer, asyncHandler(async (req, res) => {
    const trainer = await services.trainer.updateProfile(req.trainerId, req.body);
    res.json(trainer);
  }));

  /* ─── My members (those with sessions) ──────────────────── */
  router.get('/members', loadTrainer, asyncHandler(async (req, res) => {
    const members = await services.trainer.listMembers(req.trainerId);
    res.json(members);
  }));

  return router;
}

module.exports = trainerRoutes;
