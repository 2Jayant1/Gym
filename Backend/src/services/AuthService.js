/**
 * Auth Service — handles authentication business logic.
 *
 * Separated from routes so it can be unit-tested independently
 * and reused across different entry points (REST, WebSocket, CLI).
 */
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const { AppError } = require('../utils/AppError');

class AuthService {
  /**
   * @param {object} deps
   * @param {object} deps.repos — repository collection
   * @param {object} deps.config — env config
   */
  constructor({ repos, config }) {
    this.repos = repos;
    this.config = config;
  }

  /* ─── Token helpers ───────────────────────────────────── */

  signAccessToken(payload) {
    return jwt.sign(payload, this.config.AUTH_SECRET, {
      expiresIn: this.config.JWT_EXPIRES_IN,
    });
  }

  signRefreshToken(payload) {
    return jwt.sign(payload, this.config.AUTH_SECRET, {
      expiresIn: this.config.JWT_REFRESH_EXPIRES_IN || '7d',
    });
  }

  verifyToken(token) {
    try { return jwt.verify(token, this.config.AUTH_SECRET); }
    catch { return null; }
  }

  /* ─── Login ───────────────────────────────────────────── */

  async login({ email, username, password }) {
    if (!password || (!email && !username)) {
      throw AppError.badRequest('email or username and password are required');
    }

    const query = email ? { email: email.toLowerCase() } : { username };
    const user = await this.repos.user.findOne(query, { select: '+password', lean: false });
    if (!user) throw AppError.unauthorized('Invalid credentials');

    const ok = bcrypt.compareSync(password, user.password);
    if (!ok) throw AppError.unauthorized('Invalid credentials');

    // Update last login (fire-and-forget)
    user.lastLoginAt = new Date();
    user.save().catch(() => {});

    const tokenPayload = {
      id: user.id, role: user.role, name: user.name,
      email: user.email, username: user.username,
    };

    const accessToken = this.signAccessToken(tokenPayload);
    const refreshToken = this.signRefreshToken({ id: user.id, tokenFamily: crypto.randomUUID() });

    return {
      token: accessToken,
      refreshToken,
      user: tokenPayload,
    };
  }

  /* ─── Register ────────────────────────────────────────── */

  async register(data) {
    const { name, username, email, password, phone, dateOfBirth, gender, fitnessGoals } = data;

    const existing = await this.repos.user.findOne({
      $or: [{ email: email.toLowerCase() }, { username }],
    });
    if (existing) {
      const field = existing.email === email.toLowerCase() ? 'email' : 'username';
      throw AppError.conflict(`A user with that ${field} already exists`);
    }

    const defaultPlan = await this.repos.membershipPlan.findOne(
      { status: 'active' },
      { sort: { monthlyFee: 1 } },
    );

    const user = await this.repos.user.create({
      name, username,
      email: email.toLowerCase(),
      password,
      role: 'member',
      phone,
      dateOfBirth: dateOfBirth ? new Date(dateOfBirth) : undefined,
      gender: gender || undefined,
      fitnessGoals,
      membershipPlan: defaultPlan?._id,
    });

    // Welcome notification (fire-and-forget)
    this.repos.notification.create({
      user: user._id,
      title: 'Welcome to FitFlex!',
      message: 'Your account has been created. Explore your dashboard to get started.',
      type: 'success',
    }).catch(() => {});

    const tokenPayload = {
      id: user.id, role: user.role, name: user.name,
      email: user.email, username: user.username,
    };

    const accessToken = this.signAccessToken(tokenPayload);
    const refreshToken = this.signRefreshToken({ id: user.id, tokenFamily: crypto.randomUUID() });

    return { token: accessToken, refreshToken, user: tokenPayload };
  }

  /* ─── Refresh ─────────────────────────────────────────── */

  async refresh(refreshToken) {
    const payload = this.verifyToken(refreshToken);
    if (!payload?.id) throw AppError.unauthorized('Invalid refresh token');

    const user = await this.repos.user.findById(payload.id, {
      select: 'role name email username status',
    });
    if (!user || user.status === 'suspended') {
      throw AppError.unauthorized('Account not found or suspended');
    }

    const tokenPayload = {
      id: user._id, role: user.role, name: user.name,
      email: user.email, username: user.username,
    };

    const newAccessToken = this.signAccessToken(tokenPayload);
    const newRefreshToken = this.signRefreshToken({ id: user._id, tokenFamily: crypto.randomUUID() });

    return { token: newAccessToken, refreshToken: newRefreshToken, user: tokenPayload };
  }
}

module.exports = AuthService;
