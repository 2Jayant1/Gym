/**
 * Member Service — attendance, progress, classes, tickets, notifications, workout plans.
 *
 * Every method here was previously inlined inside me.js routes.
 * Moving logic into a service enables unit testing + reuse.
 */
const bcrypt = require('bcryptjs');
const { AppError } = require('../utils/AppError');

class MemberService {
  constructor({ repos }) {
    this.repos = repos;
  }

  /* ═══════════════ ATTENDANCE ═══════════════ */

  async listAttendance(userId, { skip, limit, sort = { date: -1, checkIn: -1 } }) {
    const filter = { user: userId };
    const [rows, total] = await Promise.all([
      this.repos.attendance.find(filter, { sort, skip, limit }),
      this.repos.attendance.count(filter),
    ]);
    return { rows, total };
  }

  async checkIn(userId) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const existing = await this.repos.attendance.findOne({
      user: userId,
      date: { $gte: today },
      checkOut: null,
    });
    if (existing) throw AppError.badRequest('Already checked in. Please check out first.');

    const record = await this.repos.attendance.create({
      user: userId,
      checkIn: new Date(),
      date: new Date(),
    });

    // Fire-and-forget access log
    this.repos.accessLog.create({ user: userId, accessType: 'check_in' }).catch(() => {});
    return record;
  }

  async checkOut(userId) {
    const record = await this.repos.attendance.findOne(
      { user: userId, checkOut: null },
      { sort: { checkIn: -1 }, lean: false },
    );
    if (!record) throw AppError.badRequest('No active check-in found.');

    record.checkOut = new Date();
    await record.save();

    this.repos.accessLog.create({ user: userId, accessType: 'check_out' }).catch(() => {});
    return record;
  }

  /* ═══════════════ INVOICES ═══════════════ */

  async listInvoices(userId, { skip, limit }) {
    const filter = { user: userId };
    const [rows, total] = await Promise.all([
      this.repos.invoice.find(filter, {
        sort: { issueDate: -1 },
        skip,
        limit,
        populate: { path: 'membershipPlan', select: 'name' },
      }),
      this.repos.invoice.count(filter),
    ]);
    const mapped = rows.map((r) => ({ ...r, planName: r.membershipPlan?.name }));
    return { rows: mapped, total };
  }

  /* ═══════════════ PROGRESS ═══════════════ */

  async listProgress(userId, { skip, limit }) {
    const filter = { user: userId };
    const [rows, total] = await Promise.all([
      this.repos.memberProgress.find(filter, { sort: { recordDate: -1 }, skip, limit }),
      this.repos.memberProgress.count(filter),
    ]);
    return { rows, total };
  }

  async addProgress(userId, data) {
    const { weightKg, bodyFatPercentage, muscleMassKg, chestCm, waistCm, hipsCm, bicepsCm, thighsCm, notes } = data;
    const recordDate = data.recordDate ? new Date(data.recordDate) : new Date();
    const userDoc = await this.repos.user.findById(userId, { select: 'heightCm' });
    const heightCm = userDoc?.heightCm || 170;
    const bmi = weightKg ? +(weightKg / (heightCm / 100) ** 2).toFixed(2) : undefined;

    return this.repos.memberProgress.create({
      user: userId,
      recordDate,
      weightKg,
      bodyFatPercentage,
      muscleMassKg,
      chestCm,
      waistCm,
      hipsCm,
      bicepsCm,
      thighsCm,
      bmi,
      notes,
    });
  }

  /* ═══════════════ PROFILE ═══════════════ */

  async updateProfile(userId, data) {
    const u = await this.repos.user.update(userId, data, {
      select:
        'username email role name phone address fitnessGoals medicalConditions timezone emergencyContact emergencyPhone heightCm weightKg dateOfBirth gender avatarUrl',
    });
    if (!u) throw AppError.notFound('User');
    return u;
  }

  async changePassword(userId, { currentPassword, newPassword }) {
    const user = await this.repos.user.findById(userId, { select: '+password', lean: false });
    if (!user) throw AppError.notFound('User');

    const match = bcrypt.compareSync(currentPassword, user.password);
    if (!match) throw AppError.badRequest('Current password is incorrect');

    user.password = newPassword;
    await user.save();
    return { ok: true, message: 'Password updated successfully' };
  }

  /* ═══════════════ CLASSES ═══════════════ */

  async listScheduledClasses({ skip, limit }) {
    const filter = { status: 'scheduled' };
    const [rows, total] = await Promise.all([
      this.repos.classSchedule.find(filter, {
        sort: { classDate: 1, startTime: 1 },
        skip,
        limit,
        populate: {
          path: 'fitnessClass',
          populate: { path: 'trainer', populate: { path: 'user', select: 'name' } },
        },
      }),
      this.repos.classSchedule.count(filter),
    ]);

    const mapped = rows.map((r) => ({
      ...r,
      className: r.fitnessClass?.name,
      classCategory: r.fitnessClass?.category,
      classDuration: r.fitnessClass?.durationMinutes,
      classLevel: r.fitnessClass?.difficultyLevel,
      trainerName: r.fitnessClass?.trainer?.user?.name,
      maxParticipants: r.fitnessClass?.maxParticipants,
    }));
    return { rows: mapped, total };
  }

  async listRegistrations(userId, { skip, limit }) {
    const filter = { user: userId };
    const [rows, total] = await Promise.all([
      this.repos.classRegistration.find(filter, {
        sort: { registrationDate: -1 },
        skip,
        limit,
        populate: {
          path: 'classSchedule',
          populate: { path: 'fitnessClass', select: 'name category durationMinutes' },
        },
      }),
      this.repos.classRegistration.count(filter),
    ]);
    return { rows, total };
  }

  async registerForClass(userId, classScheduleId) {
    if (!classScheduleId) throw AppError.badRequest('classScheduleId is required');

    const schedule = await this.repos.classSchedule.findById(classScheduleId, {
      populate: { path: 'fitnessClass', select: 'maxParticipants' },
      lean: false,
    });
    if (!schedule) throw AppError.notFound('Class schedule');

    if (schedule.fitnessClass?.maxParticipants) {
      const count = await this.repos.classRegistration.count({
        classSchedule: classScheduleId,
        status: { $in: ['registered', 'attended'] },
      });
      if (count >= schedule.fitnessClass.maxParticipants) {
        throw AppError.badRequest('Class is full');
      }
    }

    return this.repos.classRegistration.create({
      classSchedule: classScheduleId,
      user: userId,
      status: 'registered',
    });
  }

  async cancelRegistration(userId, registrationId) {
    const reg = await this.repos.classRegistration.updateOne(
      { _id: registrationId, user: userId },
      { status: 'cancelled' },
    );
    if (!reg) throw AppError.notFound('Registration');
    return reg;
  }

  /* ═══════════════ WORKOUT PLANS ═══════════════ */

  async listWorkoutPlans(userId) {
    const plans = await this.repos.workoutPlan.find(
      { user: userId },
      {
        sort: { createdAt: -1 },
        populate: { path: 'trainer', populate: { path: 'user', select: 'name' } },
      },
    );

    const enriched = await Promise.all(
      plans.map(async (p) => {
        const exercises = await this.repos.workoutExercise.find(
          { workoutPlan: p._id },
          {
            populate: { path: 'exercise', select: 'name muscleGroup equipmentNeeded difficultyLevel' },
            sort: { dayOfWeek: 1, sortOrder: 1 },
          },
        );
        return { ...p, exercises, trainerName: p.trainer?.user?.name };
      }),
    );
    return enriched;
  }

  /* ═══════════════ SUPPORT ═══════════════ */

  async listSupportCategories() {
    return this.repos.supportCategory.find();
  }

  async listTickets(userId, { skip, limit }) {
    const filter = { user: userId };
    const [rows, total] = await Promise.all([
      this.repos.supportTicket.find(filter, {
        sort: { createdAt: -1 },
        skip,
        limit,
        populate: [
          { path: 'category', select: 'name' },
          { path: 'assignedTo', select: 'name' },
        ],
      }),
      this.repos.supportTicket.count(filter),
    ]);

    const mapped = rows.map((row) => ({
      ...row,
      category_name: row.category?.name,
      assigned_name: row.assignedTo?.name,
    }));
    return { rows: mapped, total };
  }

  async createTicket(userId, { title, message, categoryId, priority }) {
    return this.repos.supportTicket.create({
      user: userId,
      category: categoryId || undefined,
      title,
      message,
      priority: priority || 'medium',
      status: 'open',
    });
  }

  async listTicketReplies(userId, ticketId) {
    const ticket = await this.repos.supportTicket.findOne({ _id: ticketId, user: userId });
    if (!ticket) throw AppError.notFound('Ticket');

    return this.repos.ticketReply.find(
      { ticket: ticketId, isInternal: false },
      { sort: { createdAt: 1 }, populate: { path: 'user', select: 'name role' } },
    );
  }

  async addTicketReply(userId, ticketId, { message }) {
    const ticket = await this.repos.supportTicket.findOne({ _id: ticketId, user: userId }, { lean: false });
    if (!ticket) throw AppError.notFound('Ticket');

    const reply = await this.repos.ticketReply.create({
      ticket: ticketId,
      user: userId,
      message,
      isInternal: false,
    });

    // Reopen if resolved/closed
    if (['resolved', 'closed'].includes(ticket.status)) {
      ticket.status = 'open';
      await ticket.save();
    }

    return reply;
  }

  /* ═══════════════ NOTIFICATIONS ═══════════════ */

  async listNotifications(userId, { skip, limit }) {
    const filter = { user: userId };
    const [rows, total] = await Promise.all([
      this.repos.notification.find(filter, { sort: { createdAt: -1 }, skip, limit }),
      this.repos.notification.count(filter),
    ]);
    return { rows, total };
  }

  async getUnreadCount(userId) {
    return this.repos.notification.count({ user: userId, read: false });
  }

  async markNotificationRead(userId, notificationId) {
    const n = await this.repos.notification.updateOne({ _id: notificationId, user: userId }, { read: true });
    if (!n) throw AppError.notFound('Notification');
    return n;
  }

  async markAllRead(userId) {
    await this.repos.notification.updateMany({ user: userId, read: false }, { read: true });
    return { ok: true };
  }
}

module.exports = MemberService;
