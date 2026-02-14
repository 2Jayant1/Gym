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

  /* ─── Helpers ─────────────────────────────────────────── */

  refreshTtlMs() {
    return (this.config.REFRESH_TOKEN_TTL_DAYS || 7) * 24 * 60 * 60 * 1000;
  }

  hashToken(token) {
    return crypto.createHash('sha256').update(token).digest('hex');
  }

  buildUserPayload(user) {
    return {
      id: user.id || user._id,
      role: user.role,
      name: user.name,
      email: user.email,
      username: user.username,
    };
  }

  async persistRefreshToken({ userId, tokenFamily, jti, token, context }) {
    const tokenHash = this.hashToken(token);
    const expiresAt = new Date(Date.now() + this.refreshTtlMs());
    await this.repos.refreshToken.create({
      user: userId,
      tokenFamily,
      jti,
      tokenHash,
      expiresAt,
      ip: context?.ip,
      userAgent: context?.userAgent,
    });
  }

  async revokeFamily({ userId, tokenFamily, reason }) {
    await this.repos.refreshToken.updateMany(
      { user: userId, tokenFamily, revokedAt: { $exists: false } },
      { revokedAt: new Date(), revokedReason: reason || 'revoked' },
    );
  }

  issueAccessToken(payload) {
    return jwt.sign(payload, this.config.AUTH_SECRET, {
      expiresIn: this.config.JWT_EXPIRES_IN,
    });
  }

  issueRefreshToken({ userId, tokenFamily, jti }) {
    return jwt.sign({ id: userId, tokenFamily, jti }, this.config.AUTH_SECRET, {
      expiresIn: this.config.JWT_REFRESH_EXPIRES_IN || '7d',
    });
  }

  verifyToken(token) {
    try { return jwt.verify(token, this.config.AUTH_SECRET); }
    catch { return null; }
  }

  /* ─── Login ───────────────────────────────────────────── */

  async login({ email, username, password }, context = {}) {
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

    const tokenPayload = this.buildUserPayload(user);
    const tokenFamily = crypto.randomUUID();
    const jti = crypto.randomUUID();

    const accessToken = this.issueAccessToken(tokenPayload);
    const refreshToken = this.issueRefreshToken({ userId: user.id, tokenFamily, jti });
    await this.persistRefreshToken({ userId: user.id, tokenFamily, jti, token: refreshToken, context });

    return { token: accessToken, refreshToken, user: tokenPayload };
  }

  /* ─── Register ────────────────────────────────────────── */

  async register(data, context = {}) {
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

    const tokenPayload = this.buildUserPayload(user);
    const tokenFamily = crypto.randomUUID();
    const jti = crypto.randomUUID();

    const accessToken = this.issueAccessToken(tokenPayload);
    const refreshToken = this.issueRefreshToken({ userId: user.id, tokenFamily, jti });
    await this.persistRefreshToken({ userId: user.id, tokenFamily, jti, token: refreshToken, context });

    return { token: accessToken, refreshToken, user: tokenPayload };
  }

  /* ─── Refresh ─────────────────────────────────────────── */

  async refresh(refreshToken, context = {}) {
    const payload = this.verifyToken(refreshToken);
    if (!payload?.id || !payload?.tokenFamily || !payload?.jti) {
      throw AppError.unauthorized('Invalid refresh token');
    }

    const tokenHash = this.hashToken(refreshToken);
    const stored = await this.repos.refreshToken.findOne({ tokenHash }, { lean: false });

    // Reuse detection: token not found -> revoke entire family
    if (!stored) {
      await this.revokeFamily({ userId: payload.id, tokenFamily: payload.tokenFamily, reason: 'reuse-detected' });
      throw AppError.unauthorized('Invalid refresh token');
    }

    // Expired or already revoked -> revoke family and reject
    const now = new Date();
    if (stored.revokedAt || stored.expiresAt < now) {
      await this.revokeFamily({ userId: stored.user, tokenFamily: stored.tokenFamily, reason: stored.revokedAt ? stored.revokedReason || 'revoked' : 'expired' });
      throw AppError.unauthorized('Refresh token expired');
    }

    const user = await this.repos.user.findById(payload.id, {
      select: 'role name email username status',
    });
    if (!user || user.status === 'suspended') {
      await this.revokeFamily({ userId: stored.user, tokenFamily: stored.tokenFamily, reason: 'revoked' });
      throw AppError.unauthorized('Account not found or suspended');
    }

    // Rotate: revoke current token, issue new pair with same family
    const newJti = crypto.randomUUID();
    await this.repos.refreshToken.update(stored._id, {
      revokedAt: now,
      revokedReason: 'rotated',
      replacedBy: newJti,
    });

    const tokenPayload = this.buildUserPayload(user);
    const newAccessToken = this.issueAccessToken(tokenPayload);
    const newRefreshToken = this.issueRefreshToken({ userId: user._id, tokenFamily: stored.tokenFamily, jti: newJti });
    await this.persistRefreshToken({ userId: user._id, tokenFamily: stored.tokenFamily, jti: newJti, token: newRefreshToken, context });

    return { token: newAccessToken, refreshToken: newRefreshToken, user: tokenPayload };
  }
}

module.exports = AuthService;
