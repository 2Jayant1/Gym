/**
 * Admin routes — full CRUD, analytics, ticket management.
 *
 * Architecture: Routes → AdminService (no direct model access).
 *
 * ✓ asyncHandler everywhere
 * ✓ Pagination on list endpoints
 * ✓ Audit logging on mutations (via service)
 * ✓ Aggregation pipelines for analytics (via service)
 */
const express = require('express');
const { asyncHandler } = require('../utils/AppError');
const { paginationParams, paginate } = require('../utils/paginate');

function adminRoutes({ services, models, requireAuth, requireRole }) {
  const router = express.Router();

  router.use(requireAuth);
  router.use(requireRole(['admin']));

  /* ─── Tables overview (with counts) ─────────────────────── */
  router.get('/tables', asyncHandler(async (_req, res) => {
    const results = await services.admin.getTablesOverview();
    res.json(results);
  }));

  /* ─── Generic table view (paginated) ────────────────────── */
  router.get('/table/:name', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.admin.getTableData(req.params.name, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  /* ─── Training sessions (populated) ─────────────────────── */
  router.get('/training-sessions', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.admin.getTrainingSessions({ skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  /* ─── Generic CRUD ──────────────────────────────────────── */
  router.get('/crud/:model', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.admin.crudList(req.params.model, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.get('/crud/:model/:id', asyncHandler(async (req, res) => {
    const row = await services.admin.crudGet(req.params.model, req.params.id);
    res.json(row);
  }));

  router.post('/crud/:model', asyncHandler(async (req, res) => {
    const row = await services.admin.crudCreate(req.params.model, req.body, req);
    res.status(201).json(row);
  }));

  router.patch('/crud/:model/:id', asyncHandler(async (req, res) => {
    const row = await services.admin.crudUpdate(req.params.model, req.params.id, req.body, req);
    res.json(row);
  }));

  router.delete('/crud/:model/:id', asyncHandler(async (req, res) => {
    const result = await services.admin.crudDelete(req.params.model, req.params.id, req);
    res.json(result);
  }));

  /* ─── Analytics overview ────────────────────────────────── */
  router.get('/analytics/overview', asyncHandler(async (_req, res) => {
    const data = await services.admin.getAnalyticsOverview();
    res.json(data);
  }));

  /* ─── Analytics details (aggregation pipelines) ─────────── */
  router.get('/analytics/details', asyncHandler(async (_req, res) => {
    const data = await services.admin.getAnalyticsDetails();
    res.json(data);
  }));

  /* ─── Trainer management ────────────────────────────────── */
  router.post('/trainers', asyncHandler(async (req, res) => {
    const trainer = await services.admin.createTrainer(req.body, req);
    res.status(201).json(trainer);
  }));

  /* ─── Ticket management ─────────────────────────────────── */
  router.patch('/tickets/:id/assign', asyncHandler(async (req, res) => {
    const ticket = await services.admin.assignTicket(req.params.id, req.body, req);
    res.json(ticket);
  }));

  router.get('/tickets/:id/replies', asyncHandler(async (req, res) => {
    const replies = await services.admin.getTicketReplies(req.params.id);
    res.json(replies);
  }));

  router.post('/tickets/:id/replies', asyncHandler(async (req, res) => {
    const reply = await services.admin.addTicketReply(req.params.id, req.user.id, req.body, req);
    res.status(201).json(reply);
  }));

  /* ─── Member stats ──────────────────────────────────────── */
  router.get('/members/:id/stats', asyncHandler(async (req, res) => {
    const stats = await services.admin.getMemberStats(req.params.id);
    res.json(stats);
  }));

  /* ─── Notifications ─────────────────────────────────────── */
  router.post('/notifications', asyncHandler(async (req, res) => {
    const notification = await services.admin.sendNotification(req.body, req);
    res.status(201).json(notification);
  }));

  router.post('/notifications/broadcast', asyncHandler(async (req, res) => {
    const result = await services.admin.broadcastNotification(req.body, req);
    res.status(201).json(result);
  }));

  return router;
}

module.exports = adminRoutes;
