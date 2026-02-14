/**
 * User Service — profile, password, user lookups.
 */
const bcrypt = require('bcryptjs');
const { AppError } = require('../utils/AppError');

class UserService {
  constructor({ repos }) {
    this.repos = repos;
  }

  async getProfile(userId) {
    const u = await this.repos.user.findById(userId, {
      select: 'username email role name phone joinDate membershipPlan status heightCm weightKg fitnessGoals medicalConditions',
      populate: { path: 'membershipPlan', select: 'name' },
    });
    if (!u) throw AppError.notFound('User');

    return {
      id: u._id, username: u.username, email: u.email, role: u.role,
      name: u.name, phone: u.phone, join_date: u.joinDate,
      membership_plan_id: u.membershipPlan?._id,
      plan_name: u.membershipPlan?.name,
      status: u.status, height_cm: u.heightCm, weight_kg: u.weightKg,
      fitness_goals: u.fitnessGoals, medical_conditions: u.medicalConditions,
    };
  }

  async updateProfile(userId, data) {
    const u = await this.repos.user.update(userId, data, {
      select: 'username email role name phone address fitnessGoals medicalConditions timezone emergencyContact emergencyPhone heightCm weightKg dateOfBirth gender avatarUrl',
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

  async listUsers({ page, limit, skip }) {
    const filter = { deletedAt: null };
    const [rows, total] = await Promise.all([
      this.repos.user.find(filter, {
        populate: { path: 'membershipPlan', select: 'name' },
        select: 'username email role name phone joinDate membershipPlan status heightCm weightKg fitnessGoals medicalConditions',
        sort: { joinDate: -1 },
        skip, limit,
      }),
      this.repos.user.count(filter),
    ]);

    const mapped = rows.map((u) => ({
      id: u._id, username: u.username, email: u.email, role: u.role,
      name: u.name, phone: u.phone, join_date: u.joinDate,
      membership_plan_id: u.membershipPlan?._id,
      plan_name: u.membershipPlan?.name,
      status: u.status, height_cm: u.heightCm, weight_kg: u.weightKg,
      fitness_goals: u.fitnessGoals, medical_conditions: u.medicalConditions,
    }));

    return { mapped, total };
  }

  async getStats() {
    const [totalUsers, activeUsers, totalPlans] = await Promise.all([
      this.repos.user.estimatedCount(),
      this.repos.user.count({ status: 'active' }),
      this.repos.membershipPlan.estimatedCount(),
    ]);
    return { totalUsers, activeUsers, totalPlans };
  }
}

module.exports = UserService;
