/**
 * Admin Service — CRUD, analytics, ticket management, notifications.
 */
const bcrypt = require('bcryptjs');
const { AppError } = require('../utils/AppError');

class AdminService {
  constructor({ repos, models }) {
    this.repos = repos;
    this.models = models;
  }

  /* ─── Model map for generic CRUD ────────────────────── */
  get modelMap() {
    return {
      membership_plans: this.models.MembershipPlan,
      users: this.models.User,
      trainers: this.models.Trainer,
      training_sessions: this.models.TrainingSession,
      workout_plans: this.models.WorkoutPlan,
      exercises: this.models.Exercise,
      workout_exercises: this.models.WorkoutExercise,
      attendance: this.models.Attendance,
      access_logs: this.models.AccessLog,
      fitness_classes: this.models.FitnessClass,
      class_schedule: this.models.ClassSchedule,
      class_registrations: this.models.ClassRegistration,
      equipment: this.models.Equipment,
      equipment_maintenance: this.models.EquipmentMaintenance,
      support_categories: this.models.SupportCategory,
      support_tickets: this.models.SupportTicket,
      ticket_replies: this.models.TicketReply,
      member_progress: this.models.MemberProgress,
      gym_usage_stats: this.models.GymUsageStat,
      invoices: this.models.Invoice,
      payments: this.models.Payment,
      notifications: this.models.Notification,
      audit_logs: this.models.AuditLog,
    };
  }

  resolveModel(name) {
    const Model = this.modelMap[name];
    if (!Model) throw AppError.badRequest(`Unknown model: ${name}`);
    return Model;
  }

  logAudit({ action, targetModel, targetId, metadata, req }) {
    return this.repos.auditLog
      .create({
        actor: req.user?.id,
        action,
        targetModel,
        targetId,
        metadata,
        ip: req.ip,
      })
      .catch(() => {});
  }

  /* ═══════════════ TABLES OVERVIEW ═══════════════ */

  async getTablesOverview() {
    const tables = Object.keys(this.modelMap);
    const results = await Promise.all(
      tables.map(async (t) => {
        const count = await this.modelMap[t].estimatedDocumentCount().catch(() => null);
        return { table: t, count };
      }),
    );
    results.sort((a, b) => a.table.localeCompare(b.table));
    return results;
  }

  /* ═══════════════ GENERIC TABLE VIEW ═══════════════ */

  async getTableData(name, { skip, limit }) {
    const Model = this.resolveModel(name);
    const [rows, total] = await Promise.all([
      Model.find().sort({ _id: -1 }).skip(skip).limit(limit).lean(),
      Model.countDocuments(),
    ]);
    return { rows, total };
  }

  /* ═══════════════ TRAINING SESSIONS ═══════════════ */

  async getTrainingSessions({ skip, limit }) {
    const [rows, total] = await Promise.all([
      this.repos.trainingSession.find(
        {},
        {
          sort: { sessionDate: -1, startTime: -1 },
          skip,
          limit,
          populate: [
            { path: 'trainer', populate: { path: 'user', select: 'name' } },
            { path: 'user', select: 'name' },
          ],
        },
      ),
      this.repos.trainingSession.count({}),
    ]);

    const mapped = rows.map((row) => ({
      ...row,
      trainer_id: row.trainer?._id,
      trainer_name: row.trainer?.user?.name,
      member_name: row.user?.name,
    }));
    return { rows: mapped, total };
  }

  /* ═══════════════ GENERIC CRUD ═══════════════ */

  async crudList(modelName, { skip, limit }) {
    const Model = this.resolveModel(modelName);
    const [rows, total] = await Promise.all([
      Model.find().sort({ _id: -1 }).skip(skip).limit(limit).lean(),
      Model.countDocuments(),
    ]);
    return { rows, total };
  }

  async crudGet(modelName, id) {
    const Model = this.resolveModel(modelName);
    const row = await Model.findById(id).lean();
    if (!row) throw AppError.notFound(modelName);
    return row;
  }

  async crudCreate(modelName, body, req) {
    const Model = this.resolveModel(modelName);
    const row = await Model.create(body);
    this.logAudit({ action: 'create', targetModel: modelName, targetId: row.id, metadata: { body }, req });
    return row;
  }

  async crudUpdate(modelName, id, body, req) {
    const Model = this.resolveModel(modelName);
    const payload = { ...body };
    if (modelName === 'users' && payload.password) {
      payload.password = bcrypt.hashSync(payload.password, 12);
    }
    const row = await Model.findByIdAndUpdate(id, payload, { new: true, runValidators: true });
    if (!row) throw AppError.notFound(modelName);
    this.logAudit({ action: 'update', targetModel: modelName, targetId: row.id, metadata: { body }, req });
    return row;
  }

  async crudDelete(modelName, id, req) {
    const Model = this.resolveModel(modelName);
    const row = await Model.findByIdAndDelete(id);
    if (!row) throw AppError.notFound(modelName);
    this.logAudit({ action: 'delete', targetModel: modelName, targetId: row.id, metadata: {}, req });
    return { ok: true };
  }

  /* ═══════════════ ANALYTICS ═══════════════ */

  async getAnalyticsOverview() {
    const [totalUsers, activeUsers, totalVisits, revenueAgg] = await Promise.all([
      this.repos.user.estimatedCount(),
      this.repos.user.count({ status: 'active' }),
      this.repos.attendance.estimatedCount(),
      this.repos.invoice.aggregate([{ $group: { _id: null, totalRevenue: { $sum: '$totalAmount' } } }]),
    ]);
    return {
      totalUsers,
      activeUsers,
      totalVisits,
      totalRevenue: revenueAgg?.[0]?.totalRevenue || 0,
    };
  }

  async getAnalyticsDetails() {
    const [usersByRole, planDistribution, revenueByStatus, paymentsByMethod, monthlyRevenue] = await Promise.all([
      this.repos.user.aggregate([
        { $group: { _id: '$role', count: { $sum: 1 } } },
        { $project: { role: '$_id', count: 1, _id: 0 } },
      ]),
      this.repos.user.aggregate([
        { $group: { _id: '$membershipPlan', count: { $sum: 1 } } },
        { $lookup: { from: 'membershipplans', localField: '_id', foreignField: '_id', as: 'plan' } },
        { $unwind: { path: '$plan', preserveNullAndEmptyArrays: true } },
        { $project: { name: '$plan.name', count: 1, _id: 0 } },
      ]),
      this.repos.invoice.aggregate([
        { $group: { _id: '$status', total: { $sum: '$totalAmount' } } },
        { $project: { status: '$_id', total: 1, _id: 0 } },
      ]),
      this.repos.payment.aggregate([
        { $group: { _id: '$paymentMethod', total: { $sum: '$amount' } } },
        { $project: { method: '$_id', total: 1, _id: 0 } },
      ]),
      this.repos.invoice.aggregate([
        {
          $group: { _id: { $dateToString: { format: '%Y-%m', date: '$issueDate' } }, total: { $sum: '$totalAmount' } },
        },
        { $project: { month: '$_id', total: 1, _id: 0 } },
        { $sort: { month: -1 } },
        { $limit: 6 },
        { $sort: { month: 1 } },
      ]),
    ]);

    return { usersByRole, planDistribution, revenueByStatus, paymentsByMethod, monthlyRevenue };
  }

  /* ═══════════════ TRAINER MANAGEMENT ═══════════════ */

  async createTrainer(data, req) {
    const { userId, specialization, certification, hourlyRate, experienceYears, bio } = data;
    if (!userId) throw AppError.badRequest('userId is required');

    const user = await this.repos.user.findById(userId, { lean: false });
    if (!user) throw AppError.notFound('User');

    user.role = 'trainer';
    await user.save();

    const trainer = await this.repos.trainer.create({
      user: userId,
      specialization,
      certification,
      hourlyRate: hourlyRate ? Number(hourlyRate) : undefined,
      experienceYears: experienceYears ? Number(experienceYears) : undefined,
      bio,
    });

    this.logAudit({
      action: 'create_trainer',
      targetModel: 'trainers',
      targetId: trainer.id,
      metadata: { userId },
      req,
    });
    return trainer;
  }

  /* ═══════════════ TICKETS ═══════════════ */

  async assignTicket(ticketId, { assignedTo, status }, req) {
    const update = {};
    if (assignedTo) update.assignedTo = assignedTo;
    if (status) update.status = status;

    const ticket = await this.repos.supportTicket.update(ticketId, update, {
      populate: [
        { path: 'user', select: 'name email' },
        { path: 'assignedTo', select: 'name' },
      ],
    });
    if (!ticket) throw AppError.notFound('Ticket');

    if (assignedTo) {
      this.repos.notification
        .create({
          user: assignedTo,
          title: 'Ticket Assigned',
          message: `You have been assigned ticket: ${ticket.title}`,
          type: 'info',
          link: '/support',
        })
        .catch(() => {});
    }

    return ticket;
  }

  async getTicketReplies(ticketId) {
    return this.repos.ticketReply.find(
      { ticket: ticketId },
      { sort: { createdAt: 1 }, populate: { path: 'user', select: 'name role' } },
    );
  }

  async addTicketReply(ticketId, userId, { message, isInternal }, req) {
    if (!message) throw AppError.badRequest('message is required');

    const reply = await this.repos.ticketReply.create({
      ticket: ticketId,
      user: userId,
      message,
      isInternal: !!isInternal,
    });

    if (!isInternal) {
      const ticket = await this.repos.supportTicket.findById(ticketId);
      if (ticket) {
        this.repos.notification
          .create({
            user: ticket.user,
            title: 'New reply on your ticket',
            message: `Your ticket "${ticket.title}" received a new reply.`,
            type: 'info',
          })
          .catch(() => {});
      }
    }

    return reply;
  }

  /* ═══════════════ MEMBER STATS ═══════════════ */

  async getMemberStats(userId) {
    const [attendanceCount, invoiceCount, progressCount, ticketCount, latestProgress] = await Promise.all([
      this.repos.attendance.count({ user: userId }),
      this.repos.invoice.count({ user: userId }),
      this.repos.memberProgress.count({ user: userId }),
      this.repos.supportTicket.count({ user: userId }),
      this.repos.memberProgress.findOne({ user: userId }, { sort: { recordDate: -1 } }),
    ]);
    return { attendanceCount, invoiceCount, progressCount, ticketCount, latestProgress };
  }

  /* ═══════════════ NOTIFICATIONS ═══════════════ */

  async sendNotification(data, req) {
    const { userId, title, message, type } = data;
    if (!userId || !title || !message) throw AppError.badRequest('userId, title, and message are required');

    const notification = await this.repos.notification.create({
      user: userId,
      title,
      message,
      type: type || 'info',
    });
    this.logAudit({
      action: 'send_notification',
      targetModel: 'notifications',
      targetId: notification.id,
      metadata: { userId },
      req,
    });
    return notification;
  }

  async broadcastNotification(data, req) {
    const { title, message, type, role } = data;
    if (!title || !message) throw AppError.badRequest('title and message are required');

    const filter = role ? { role, deletedAt: null } : { deletedAt: null };
    const userIds = await this.repos.user.find(filter, { select: '_id' });

    const docs = userIds.map((u) => ({ user: u._id, title, message, type: type || 'info' }));
    if (docs.length) await this.repos.notification.bulkCreate(docs);

    return { ok: true, count: docs.length };
  }
}

module.exports = AdminService;
