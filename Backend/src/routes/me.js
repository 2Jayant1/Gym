/**
 * Member ("me") routes — attendance, invoices, progress, classes, tickets, notifications.
 *
 * Architecture: Routes → MemberService (no direct model access).
 *
 * ✓ asyncHandler everywhere (zero try/catch)
 * ✓ AppError for guard clauses
 * ✓ Pagination on all list endpoints
 * ✓ Zod validation on mutations
 */
const express = require('express');
const { profileUpdateSchema, passwordChangeSchema, ticketCreateSchema, ticketReplySchema, progressSchema, validate } = require('../utils/validate');
const { asyncHandler } = require('../utils/AppError');
const { paginationParams, paginate } = require('../utils/paginate');

function meRoutes({ services, requireAuth }) {
  const router = express.Router();

  router.use(requireAuth);

  /* ═══════════════════════════════════════════════════════════
     ATTENDANCE
     ═══════════════════════════════════════════════════════════ */
  router.get('/attendance', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listAttendance(req.user.id, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.post('/check-in', asyncHandler(async (req, res) => {
    const record = await services.member.checkIn(req.user.id);
    res.status(201).json(record);
  }));

  router.post('/check-out', asyncHandler(async (req, res) => {
    const record = await services.member.checkOut(req.user.id);
    res.json(record);
  }));

  /* ═══════════════════════════════════════════════════════════
     INVOICES
     ═══════════════════════════════════════════════════════════ */
  router.get('/invoices', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listInvoices(req.user.id, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  /* ═══════════════════════════════════════════════════════════
     PROGRESS
     ═══════════════════════════════════════════════════════════ */
  router.get('/progress', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listProgress(req.user.id, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.post('/progress', validate(progressSchema), asyncHandler(async (req, res) => {
    const record = await services.member.addProgress(req.user.id, req.validated);
    res.status(201).json(record);
  }));

  /* ═══════════════════════════════════════════════════════════
     PROFILE
     ═══════════════════════════════════════════════════════════ */
  router.patch('/profile', validate(profileUpdateSchema), asyncHandler(async (req, res) => {
    const u = await services.member.updateProfile(req.user.id, req.validated);
    res.json(u);
  }));

  router.patch('/password', validate(passwordChangeSchema), asyncHandler(async (req, res) => {
    const result = await services.member.changePassword(req.user.id, req.validated);
    res.json(result);
  }));

  /* ═══════════════════════════════════════════════════════════
     CLASSES
     ═══════════════════════════════════════════════════════════ */
  router.get('/classes', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listScheduledClasses({ skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.get('/registrations', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listRegistrations(req.user.id, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.post('/classes/register', asyncHandler(async (req, res) => {
    const reg = await services.member.registerForClass(req.user.id, req.body?.classScheduleId);
    res.status(201).json(reg);
  }));

  router.patch('/registrations/:id/cancel', asyncHandler(async (req, res) => {
    const reg = await services.member.cancelRegistration(req.user.id, req.params.id);
    res.json(reg);
  }));

  /* ═══════════════════════════════════════════════════════════
     WORKOUT PLANS
     ═══════════════════════════════════════════════════════════ */
  router.get('/workout-plans', asyncHandler(async (req, res) => {
    const plans = await services.member.listWorkoutPlans(req.user.id);
    res.json(plans);
  }));

  /* ═══════════════════════════════════════════════════════════
     SUPPORT CATEGORIES
     ═══════════════════════════════════════════════════════════ */
  router.get('/support-categories', asyncHandler(async (_req, res) => {
    const rows = await services.member.listSupportCategories();
    res.json(rows);
  }));

  /* ═══════════════════════════════════════════════════════════
     TICKETS
     ═══════════════════════════════════════════════════════════ */
  router.get('/tickets', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listTickets(req.user.id, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.post('/tickets', validate(ticketCreateSchema), asyncHandler(async (req, res) => {
    const ticket = await services.member.createTicket(req.user.id, req.validated);
    res.status(201).json(ticket);
  }));

  router.get('/tickets/:id/replies', asyncHandler(async (req, res) => {
    const replies = await services.member.listTicketReplies(req.user.id, req.params.id);
    res.json(replies);
  }));

  router.post('/tickets/:id/replies', validate(ticketReplySchema), asyncHandler(async (req, res) => {
    const reply = await services.member.addTicketReply(req.user.id, req.params.id, req.validated);
    res.status(201).json(reply);
  }));

  /* ═══════════════════════════════════════════════════════════
     NOTIFICATIONS
     ═══════════════════════════════════════════════════════════ */
  router.get('/notifications', asyncHandler(async (req, res) => {
    const { page, limit, skip } = paginationParams(req);
    const { rows, total } = await services.member.listNotifications(req.user.id, { skip, limit });
    res.json(paginate(rows, total, page, limit));
  }));

  router.get('/notifications/unread-count', asyncHandler(async (req, res) => {
    const count = await services.member.getUnreadCount(req.user.id);
    res.json({ count });
  }));

  router.patch('/notifications/:id/read', asyncHandler(async (req, res) => {
    const n = await services.member.markNotificationRead(req.user.id, req.params.id);
    res.json(n);
  }));

  router.patch('/notifications/read-all', asyncHandler(async (req, res) => {
    const result = await services.member.markAllRead(req.user.id);
    res.json(result);
  }));

  return router;
}

module.exports = meRoutes;
