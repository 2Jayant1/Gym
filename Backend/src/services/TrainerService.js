/**
 * Trainer Service — sessions, classes, schedules, profile, members.
 */
const { AppError } = require('../utils/AppError');

class TrainerService {
  constructor({ repos }) {
    this.repos = repos;
  }

  async resolveTrainerId(userId) {
    const row = await this.repos.trainer.findOne({ user: userId }, { select: '_id' });
    if (!row) throw AppError.notFound('Trainer profile');
    return row._id;
  }

  /* ═══════════════ SESSIONS ═══════════════ */

  async listSessions(trainerId, { skip, limit }) {
    const filter = { trainer: trainerId };
    const [rows, total] = await Promise.all([
      this.repos.trainingSession.find(filter, {
        sort: { sessionDate: -1, startTime: -1 },
        skip, limit,
        populate: { path: 'user', select: 'name email' },
      }),
      this.repos.trainingSession.count(filter),
    ]);
    const mapped = rows.map((r) => ({ ...r, member_name: r.user?.name, member_email: r.user?.email }));
    return { rows: mapped, total };
  }

  async createSession(trainerId, data) {
    const { userId, sessionDate, startTime, endTime, durationMinutes, sessionType, status, notes, cost } = data;
    if (!userId || !sessionDate || !startTime || !endTime || !durationMinutes) {
      throw AppError.badRequest('userId, sessionDate, startTime, endTime, durationMinutes are required');
    }
    return this.repos.trainingSession.create({
      trainer: trainerId,
      user: userId,
      sessionDate: new Date(sessionDate),
      startTime, endTime,
      durationMinutes: Number(durationMinutes),
      sessionType,
      status: status || 'scheduled',
      notes,
      cost: cost ? Number(cost) : undefined,
    });
  }

  async updateSession(trainerId, sessionId, data) {
    const payload = { ...data };
    if (payload.sessionDate) payload.sessionDate = new Date(payload.sessionDate);
    if (payload.durationMinutes) payload.durationMinutes = Number(payload.durationMinutes);
    if (payload.cost) payload.cost = Number(payload.cost);

    const row = await this.repos.trainingSession.updateOne(
      { _id: sessionId, trainer: trainerId },
      payload,
    );
    if (!row) throw AppError.notFound('Session');
    return row;
  }

  async deleteSession(trainerId, sessionId) {
    const row = await this.repos.trainingSession.deleteOne({ _id: sessionId, trainer: trainerId });
    if (!row) throw AppError.notFound('Session');
    return { ok: true };
  }

  /* ═══════════════ CLASSES ═══════════════ */

  async listClasses(trainerId, { skip, limit }) {
    const filter = { trainer: trainerId };
    const [rows, total] = await Promise.all([
      this.repos.fitnessClass.find(filter, { sort: { _id: -1 }, skip, limit }),
      this.repos.fitnessClass.count(filter),
    ]);
    return { rows, total };
  }

  async createClass(trainerId, data) {
    const { name, description, maxParticipants, durationMinutes, difficultyLevel, category, status } = data;
    if (!name || !durationMinutes) throw AppError.badRequest('name and durationMinutes are required');
    return this.repos.fitnessClass.create({
      name, description,
      trainer: trainerId,
      maxParticipants: maxParticipants ? Number(maxParticipants) : undefined,
      durationMinutes: Number(durationMinutes),
      difficultyLevel, category,
      status: status || 'active',
    });
  }

  async updateClass(trainerId, classId, data) {
    const payload = { ...data };
    if (payload.maxParticipants) payload.maxParticipants = Number(payload.maxParticipants);
    if (payload.durationMinutes) payload.durationMinutes = Number(payload.durationMinutes);

    const row = await this.repos.fitnessClass.updateOne(
      { _id: classId, trainer: trainerId },
      payload,
    );
    if (!row) throw AppError.notFound('Class');
    return row;
  }

  async deleteClass(trainerId, classId) {
    const row = await this.repos.fitnessClass.deleteOne({ _id: classId, trainer: trainerId });
    if (!row) throw AppError.notFound('Class');
    return { ok: true };
  }

  /* ═══════════════ SCHEDULES ═══════════════ */

  async listSchedules(trainerId, classId) {
    const cls = await this.repos.fitnessClass.findOne({ _id: classId, trainer: trainerId });
    if (!cls) throw AppError.notFound('Class');
    return this.repos.classSchedule.find(
      { fitnessClass: classId },
      { sort: { classDate: 1, startTime: 1 } },
    );
  }

  async createSchedule(trainerId, classId, data) {
    const cls = await this.repos.fitnessClass.findOne({ _id: classId, trainer: trainerId });
    if (!cls) throw AppError.notFound('Class');
    const { classDate, startTime, endTime, room } = data;
    if (!classDate || !startTime || !endTime) throw AppError.badRequest('classDate, startTime, endTime are required');
    return this.repos.classSchedule.create({
      fitnessClass: classId,
      classDate: new Date(classDate),
      startTime, endTime, room,
    });
  }

  /* ═══════════════ PROFILE ═══════════════ */

  async getProfile(trainerId) {
    const trainer = await this.repos.trainer.findById(trainerId, {
      populate: { path: 'user', select: 'name email phone' },
    });
    if (!trainer) throw AppError.notFound('Trainer');
    return trainer;
  }

  async updateProfile(trainerId, data) {
    const allowed = ['specialization', 'certification', 'hourlyRate', 'experienceYears', 'bio'];
    const payload = {};
    for (const k of allowed) {
      if (data[k] !== undefined) payload[k] = data[k];
    }
    const trainer = await this.repos.trainer.update(trainerId, payload);
    if (!trainer) throw AppError.notFound('Trainer');
    return trainer;
  }

  /* ═══════════════ MEMBERS ═══════════════ */

  async listMembers(trainerId) {
    const userIds = await this.repos.trainingSession.distinct('user', { trainer: trainerId });
    return this.repos.user.find(
      { _id: { $in: userIds } },
      { select: 'name email phone fitnessGoals status' },
    );
  }
}

module.exports = TrainerService;
