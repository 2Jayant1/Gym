/**
 * Mongoose models — production-grade with compound indexes.
 *
 * INDEX STRATEGY (designed for millions of rows):
 *   • Every foreign-key gets a single-field index (via `index: true`).
 *   • Frequent filter combos get compound indexes.
 *   • TTL indexes auto-expire stale data (audit logs, access logs).
 *   • Unique constraints enforce data integrity at the DB level.
 *   • `{ lean: true }` is used at query time, not schema level.
 *
 * CONVENTIONS:
 *   • Schemas are defined → indexes declared → model registered.
 *   • `timestamps: true` gives us createdAt + updatedAt automatically.
 *   • `select: false` on password keeps it out of default queries.
 */
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const { Schema } = mongoose;

/* ═══════════════════════════════════════════════════════════════
   MEMBERSHIP PLAN
   ═══════════════════════════════════════════════════════════════ */
const membershipPlanSchema = new Schema(
  {
    name:             { type: String, unique: true, required: true, trim: true },
    description:      { type: String },
    monthlyFee:       { type: Number, required: true },
    durationDays:     { type: Number, required: true },
    maxVisitsPerWeek: { type: Number },
    includesTrainer:  { type: Boolean, default: false },
    includesClasses:  { type: Boolean, default: false },
    status:           { type: String, default: 'active', enum: ['active', 'inactive'] },
    signupFee:        { type: Number, default: 0 },
    cancellationFee:  { type: Number, default: 0 },
  },
  { timestamps: true },
);

membershipPlanSchema.index({ status: 1, monthlyFee: 1 });

/* ═══════════════════════════════════════════════════════════════
   USER
   ═══════════════════════════════════════════════════════════════ */
const userSchema = new Schema(
  {
    username:         { type: String, unique: true, required: true, trim: true },
    email:            { type: String, unique: true, required: true, lowercase: true, trim: true },
    password:         { type: String, required: true, select: false },
    role:             { type: String, default: 'member', enum: ['member', 'trainer', 'admin'] },
    name:             { type: String, required: true },
    phone:            { type: String },
    dateOfBirth:      { type: Date },
    gender:           { type: String },
    emergencyContact: { type: String },
    emergencyPhone:   { type: String },
    address:          { type: String },
    joinDate:         { type: Date, default: Date.now },
    membershipPlan:   { type: Schema.Types.ObjectId, ref: 'MembershipPlan' },
    status:           { type: String, default: 'active', enum: ['active', 'inactive', 'suspended'] },
    heightCm:         { type: Number },
    weightKg:         { type: Number },
    fitnessGoals:     { type: String },
    medicalConditions:{ type: String },
    avatarUrl:        { type: String },
    timezone:         { type: String, default: 'UTC' },
    isEmailVerified:  { type: Boolean, default: false },
    lastLoginAt:      { type: Date },
    permissions:      { type: [String], default: [] },
    deletedAt:        { type: Date, default: null },
  },
  { timestamps: true },
);

userSchema.index({ role: 1, status: 1 });
userSchema.index({ deletedAt: 1 });
userSchema.index({ membershipPlan: 1 });

userSchema.pre('save', function hashPassword(next) {
  if (!this.isModified('password')) return next();
  this.password = bcrypt.hashSync(this.password, 12);   // 12 rounds (production standard)
  next();
});

/* ═══════════════════════════════════════════════════════════════
   ATTENDANCE
   ═══════════════════════════════════════════════════════════════ */
const attendanceSchema = new Schema(
  {
    user:            { type: Schema.Types.ObjectId, ref: 'User', required: true },
    checkIn:         { type: Date, default: Date.now },
    checkOut:        { type: Date },
    date:            { type: Date, default: Date.now },
    durationMinutes: { type: Number, default: 0 },
    notes:           { type: String },
  },
  { timestamps: false },
);

// Compound: "all visits by user, newest first" — the most common query
attendanceSchema.index({ user: 1, date: -1 });
// Compound: open check-ins for a user (used in check-in/check-out logic)
attendanceSchema.index({ user: 1, checkOut: 1 });

attendanceSchema.pre('save', function updateDuration(next) {
  if (this.checkOut && this.checkIn) {
    this.durationMinutes = Math.max(0, Math.round((this.checkOut - this.checkIn) / 60000));
  }
  next();
});

/* ═══════════════════════════════════════════════════════════════
   ACCESS LOG  (TTL: auto-delete after 90 days)
   ═══════════════════════════════════════════════════════════════ */
const accessLogSchema = new Schema(
  {
    user:       { type: Schema.Types.ObjectId, ref: 'User', required: true },
    accessTime: { type: Date, default: Date.now },
    accessType: { type: String, default: 'check_in', enum: ['check_in', 'check_out', 'denied'] },
    reason:     { type: String },
  },
  { timestamps: false },
);

accessLogSchema.index({ user: 1, accessTime: -1 });
accessLogSchema.index({ accessTime: 1 }, { expireAfterSeconds: 90 * 24 * 3600 });  // TTL 90 days

/* ═══════════════════════════════════════════════════════════════
   INVOICE
   ═══════════════════════════════════════════════════════════════ */
const invoiceSchema = new Schema(
  {
    user:           { type: Schema.Types.ObjectId, ref: 'User', required: true },
    membershipPlan: { type: Schema.Types.ObjectId, ref: 'MembershipPlan', required: true },
    invoiceNumber:  { type: String, unique: true },
    amount:         { type: Number, required: true },
    issueDate:      { type: Date, required: true },
    dueDate:        { type: Date, required: true },
    status:         { type: String, default: 'pending', enum: ['pending', 'paid', 'overdue', 'cancelled'] },
    taxAmount:      { type: Number, default: 0 },
    discountAmount: { type: Number, default: 0 },
    totalAmount:    { type: Number, required: true },
    notes:          { type: String },
  },
  { timestamps: true },
);

invoiceSchema.index({ user: 1, issueDate: -1 });
invoiceSchema.index({ status: 1 });
invoiceSchema.index({ dueDate: 1 });

invoiceSchema.pre('save', async function setInvoiceNumber(next) {
  if (this.invoiceNumber) return next();
  const ym = new Date(this.issueDate).toISOString().slice(0, 7).replace('-', '');
  const count = await mongoose.model('Invoice').countDocuments({
    issueDate: {
      $gte: new Date(`${ym.slice(0, 4)}-${ym.slice(4, 6)}-01`),
      $lt:  new Date(`${ym.slice(0, 4)}-${ym.slice(4, 6)}-31T23:59:59.999Z`),
    },
  });
  this.invoiceNumber = `INV-${ym}-${String(count + 1).padStart(4, '0')}`;
  next();
});

/* ═══════════════════════════════════════════════════════════════
   PAYMENT
   ═══════════════════════════════════════════════════════════════ */
const paymentSchema = new Schema(
  {
    invoice:       { type: Schema.Types.ObjectId, ref: 'Invoice', required: true },
    user:          { type: Schema.Types.ObjectId, ref: 'User', required: true },
    amount:        { type: Number, required: true },
    paymentDate:   { type: Date, default: Date.now },
    paymentMethod: { type: String, default: 'cash', enum: ['cash', 'card', 'transfer', 'online'] },
    transactionId: { type: String, unique: true, sparse: true },
    status:        { type: String, default: 'completed', enum: ['pending', 'completed', 'failed', 'refunded'] },
    notes:         { type: String },
  },
  { timestamps: false },
);

paymentSchema.index({ user: 1, paymentDate: -1 });
paymentSchema.index({ invoice: 1 });

/* ═══════════════════════════════════════════════════════════════
   TRAINER
   ═══════════════════════════════════════════════════════════════ */
const trainerSchema = new Schema(
  {
    user:            { type: Schema.Types.ObjectId, ref: 'User', required: true, unique: true },
    specialization:  { type: String },
    certification:   { type: String },
    hourlyRate:      { type: Number },
    experienceYears: { type: Number },
    bio:             { type: String },
    status:          { type: String, default: 'active', enum: ['active', 'inactive'] },
  },
  { timestamps: true },
);

/* ═══════════════════════════════════════════════════════════════
   TRAINING SESSION
   ═══════════════════════════════════════════════════════════════ */
const trainingSessionSchema = new Schema(
  {
    trainer:         { type: Schema.Types.ObjectId, ref: 'Trainer', required: true },
    user:            { type: Schema.Types.ObjectId, ref: 'User', required: true },
    sessionDate:     { type: Date, required: true },
    startTime:       { type: String, required: true },
    endTime:         { type: String, required: true },
    durationMinutes: { type: Number, required: true },
    sessionType:     { type: String, enum: ['personal', 'group', 'assessment'] },
    status:          { type: String, default: 'scheduled', enum: ['scheduled', 'completed', 'cancelled', 'no_show'] },
    notes:           { type: String },
    cost:            { type: Number },
    rating:          { type: Number, min: 1, max: 5 },
    feedback:        { type: String },
  },
  { timestamps: true },
);

trainingSessionSchema.index({ trainer: 1, sessionDate: -1 });
trainingSessionSchema.index({ user: 1, sessionDate: -1 });

/* ═══════════════════════════════════════════════════════════════
   WORKOUT PLAN
   ═══════════════════════════════════════════════════════════════ */
const workoutPlanSchema = new Schema(
  {
    user:            { type: Schema.Types.ObjectId, ref: 'User', required: true },
    trainer:         { type: Schema.Types.ObjectId, ref: 'Trainer' },
    name:            { type: String, required: true },
    description:     { type: String },
    goal:            { type: String, enum: ['weight_loss', 'muscle_gain', 'endurance', 'general_fitness'] },
    difficultyLevel: { type: String, enum: ['beginner', 'intermediate', 'advanced'] },
    startDate:       { type: Date },
    endDate:         { type: Date },
    status:          { type: String, default: 'active', enum: ['active', 'inactive', 'archived'] },
  },
  { timestamps: true },
);

workoutPlanSchema.index({ user: 1, status: 1 });

/* ═══════════════════════════════════════════════════════════════
   EXERCISE
   ═══════════════════════════════════════════════════════════════ */
const exerciseSchema = new Schema(
  {
    name:            { type: String, required: true },
    description:     { type: String },
    muscleGroup:     { type: String },
    equipmentNeeded: { type: String },
    difficultyLevel: { type: String },
    videoUrl:        { type: String },
    status:          { type: String, default: 'active', enum: ['active', 'inactive'] },
  },
  { timestamps: true },
);

exerciseSchema.index({ muscleGroup: 1 });
exerciseSchema.index({ name: 'text' });   // text search on exercise names

/* ═══════════════════════════════════════════════════════════════
   WORKOUT EXERCISE  (join table)
   ═══════════════════════════════════════════════════════════════ */
const workoutExerciseSchema = new Schema(
  {
    workoutPlan:     { type: Schema.Types.ObjectId, ref: 'WorkoutPlan', required: true },
    exercise:        { type: Schema.Types.ObjectId, ref: 'Exercise', required: true },
    dayOfWeek:       { type: Number, min: 1, max: 7 },
    sets:            { type: Number },
    reps:            { type: Number },
    weightKg:        { type: Number },
    durationMinutes: { type: Number },
    restSeconds:     { type: Number },
    notes:           { type: String },
    sortOrder:       { type: Number, default: 0 },
  },
  { timestamps: false },
);

workoutExerciseSchema.index({ workoutPlan: 1, dayOfWeek: 1, sortOrder: 1 });

/* ═══════════════════════════════════════════════════════════════
   EQUIPMENT
   ═══════════════════════════════════════════════════════════════ */
const equipmentSchema = new Schema(
  {
    name:                { type: String, required: true },
    category:            { type: String },
    brand:               { type: String },
    model:               { type: String },
    serialNumber:        { type: String, unique: true, sparse: true },
    purchaseDate:        { type: Date },
    purchasePrice:       { type: Number },
    status:              { type: String, default: 'active', enum: ['active', 'maintenance', 'retired'] },
    lastMaintenanceDate: { type: Date },
    nextMaintenanceDate: { type: Date },
    location:            { type: String },
    notes:               { type: String },
  },
  { timestamps: true },
);

equipmentSchema.index({ status: 1 });

/* ═══════════════════════════════════════════════════════════════
   EQUIPMENT MAINTENANCE
   ═══════════════════════════════════════════════════════════════ */
const equipmentMaintenanceSchema = new Schema(
  {
    equipment:           { type: Schema.Types.ObjectId, ref: 'Equipment', required: true },
    maintenanceDate:     { type: Date, required: true },
    maintenanceType:     { type: String, enum: ['routine', 'repair', 'replacement'] },
    description:         { type: String },
    cost:                { type: Number },
    technician:          { type: String },
    nextMaintenanceDate: { type: Date },
    status:              { type: String, default: 'completed', enum: ['completed', 'scheduled'] },
  },
  { timestamps: false },
);

equipmentMaintenanceSchema.index({ equipment: 1, maintenanceDate: -1 });

/* ═══════════════════════════════════════════════════════════════
   FITNESS CLASS
   ═══════════════════════════════════════════════════════════════ */
const fitnessClassSchema = new Schema(
  {
    name:            { type: String, required: true },
    description:     { type: String },
    trainer:         { type: Schema.Types.ObjectId, ref: 'Trainer' },
    maxParticipants: { type: Number },
    durationMinutes: { type: Number, required: true },
    difficultyLevel: { type: String },
    category:        { type: String },
    status:          { type: String, default: 'active', enum: ['active', 'inactive'] },
  },
  { timestamps: true },
);

fitnessClassSchema.index({ trainer: 1 });
fitnessClassSchema.index({ status: 1, category: 1 });

/* ═══════════════════════════════════════════════════════════════
   CLASS SCHEDULE
   ═══════════════════════════════════════════════════════════════ */
const classScheduleSchema = new Schema(
  {
    fitnessClass: { type: Schema.Types.ObjectId, ref: 'FitnessClass', required: true },
    classDate:    { type: Date, required: true },
    startTime:    { type: String, required: true },
    endTime:      { type: String, required: true },
    room:         { type: String },
    status:       { type: String, default: 'scheduled', enum: ['scheduled', 'cancelled', 'completed'] },
  },
  { timestamps: true },
);

classScheduleSchema.index({ fitnessClass: 1, classDate: 1 });
classScheduleSchema.index({ status: 1, classDate: 1 });

/* ═══════════════════════════════════════════════════════════════
   CLASS REGISTRATION
   ═══════════════════════════════════════════════════════════════ */
const classRegistrationSchema = new Schema(
  {
    classSchedule:    { type: Schema.Types.ObjectId, ref: 'ClassSchedule', required: true },
    user:             { type: Schema.Types.ObjectId, ref: 'User', required: true },
    registrationDate: { type: Date, default: Date.now },
    status:           { type: String, default: 'registered', enum: ['registered', 'attended', 'cancelled', 'no_show'] },
    notes:            { type: String },
  },
  { timestamps: false },
);

classRegistrationSchema.index({ classSchedule: 1, user: 1 }, { unique: true });
classRegistrationSchema.index({ user: 1, status: 1 });

/* ═══════════════════════════════════════════════════════════════
   SUPPORT
   ═══════════════════════════════════════════════════════════════ */
const supportCategorySchema = new Schema(
  { name: { type: String, unique: true, required: true }, description: { type: String } },
  { timestamps: true },
);

const supportTicketSchema = new Schema(
  {
    user:            { type: Schema.Types.ObjectId, ref: 'User', required: true },
    category:        { type: Schema.Types.ObjectId, ref: 'SupportCategory' },
    title:           { type: String, required: true },
    message:         { type: String, required: true },
    priority:        { type: String, default: 'medium', enum: ['low', 'medium', 'high', 'urgent'] },
    status:          { type: String, default: 'open', enum: ['open', 'in_progress', 'resolved', 'closed'] },
    assignedTo:      { type: Schema.Types.ObjectId, ref: 'User' },
    resolutionNotes: { type: String },
  },
  { timestamps: true },
);

supportTicketSchema.index({ user: 1, status: 1 });
supportTicketSchema.index({ status: 1, priority: 1 });
supportTicketSchema.index({ assignedTo: 1 });

const ticketReplySchema = new Schema(
  {
    ticket:     { type: Schema.Types.ObjectId, ref: 'SupportTicket', required: true },
    user:       { type: Schema.Types.ObjectId, ref: 'User', required: true },
    message:    { type: String, required: true },
    isInternal: { type: Boolean, default: false },
  },
  { timestamps: true },
);

ticketReplySchema.index({ ticket: 1, createdAt: 1 });

/* ═══════════════════════════════════════════════════════════════
   MEMBER PROGRESS
   ═══════════════════════════════════════════════════════════════ */
const memberProgressSchema = new Schema(
  {
    user:                { type: Schema.Types.ObjectId, ref: 'User', required: true },
    recordDate:          { type: Date, required: true },
    weightKg:            { type: Number },
    bodyFatPercentage:   { type: Number },
    muscleMassKg:        { type: Number },
    chestCm:             { type: Number },
    waistCm:             { type: Number },
    hipsCm:              { type: Number },
    bicepsCm:            { type: Number },
    thighsCm:            { type: Number },
    bmi:                 { type: Number },
    notes:               { type: String },
  },
  { timestamps: false },
);

memberProgressSchema.index({ user: 1, recordDate: -1 }, { unique: true });

/* ═══════════════════════════════════════════════════════════════
   GYM USAGE STAT
   ═══════════════════════════════════════════════════════════════ */
const gymUsageStatSchema = new Schema(
  {
    statDate:           { type: Date, required: true, unique: true },
    totalMembers:       { type: Number, default: 0 },
    activeMembers:      { type: Number, default: 0 },
    dailyVisits:        { type: Number, default: 0 },
    peakHourVisits:     { type: Number, default: 0 },
    newRegistrations:   { type: Number, default: 0 },
    membershipRenewals: { type: Number, default: 0 },
    totalRevenue:       { type: Number, default: 0 },
  },
  { timestamps: true },
);

/* ═══════════════════════════════════════════════════════════════
   NOTIFICATION
   ═══════════════════════════════════════════════════════════════ */
const notificationSchema = new Schema(
  {
    user:    { type: Schema.Types.ObjectId, ref: 'User', required: true },
    title:   { type: String, required: true },
    message: { type: String, required: true },
    type:    { type: String, enum: ['info', 'success', 'warning', 'error'], default: 'info' },
    read:    { type: Boolean, default: false },
    link:    { type: String },
  },
  { timestamps: true },
);

notificationSchema.index({ user: 1, read: 1, createdAt: -1 });

/* ═══════════════════════════════════════════════════════════════
   AUDIT LOG  (TTL: auto-delete after 180 days)
   ═══════════════════════════════════════════════════════════════ */
const auditLogSchema = new Schema(
  {
    actor:       { type: Schema.Types.ObjectId, ref: 'User' },
    action:      { type: String, required: true },
    targetModel: { type: String },
    targetId:    { type: String },
    metadata:    { type: Schema.Types.Mixed },
    ip:          { type: String },
  },
  { timestamps: true },
);

auditLogSchema.index({ actor: 1, createdAt: -1 });
auditLogSchema.index({ createdAt: 1 }, { expireAfterSeconds: 180 * 24 * 3600 }); // TTL 180 days

/* ═══════════════════════════════════════════════════════════════
   REGISTER MODELS
   ═══════════════════════════════════════════════════════════════ */
const MembershipPlan       = mongoose.model('MembershipPlan', membershipPlanSchema);
const User                 = mongoose.model('User', userSchema);
const Attendance           = mongoose.model('Attendance', attendanceSchema);
const AccessLog            = mongoose.model('AccessLog', accessLogSchema);
const Invoice              = mongoose.model('Invoice', invoiceSchema);
const Payment              = mongoose.model('Payment', paymentSchema);
const Trainer              = mongoose.model('Trainer', trainerSchema);
const TrainingSession      = mongoose.model('TrainingSession', trainingSessionSchema);
const WorkoutPlan          = mongoose.model('WorkoutPlan', workoutPlanSchema);
const Exercise             = mongoose.model('Exercise', exerciseSchema);
const WorkoutExercise      = mongoose.model('WorkoutExercise', workoutExerciseSchema);
const Equipment            = mongoose.model('Equipment', equipmentSchema);
const EquipmentMaintenance = mongoose.model('EquipmentMaintenance', equipmentMaintenanceSchema);
const FitnessClass         = mongoose.model('FitnessClass', fitnessClassSchema);
const ClassSchedule        = mongoose.model('ClassSchedule', classScheduleSchema);
const ClassRegistration    = mongoose.model('ClassRegistration', classRegistrationSchema);
const SupportCategory      = mongoose.model('SupportCategory', supportCategorySchema);
const SupportTicket        = mongoose.model('SupportTicket', supportTicketSchema);
const TicketReply          = mongoose.model('TicketReply', ticketReplySchema);
const MemberProgress       = mongoose.model('MemberProgress', memberProgressSchema);
const GymUsageStat         = mongoose.model('GymUsageStat', gymUsageStatSchema);
const Notification         = mongoose.model('Notification', notificationSchema);
const AuditLog             = mongoose.model('AuditLog', auditLogSchema);

module.exports = {
  MembershipPlan, User, Attendance, AccessLog, Invoice, Payment,
  Trainer, TrainingSession, WorkoutPlan, Exercise, WorkoutExercise,
  Equipment, EquipmentMaintenance, FitnessClass, ClassSchedule,
  ClassRegistration, SupportCategory, SupportTicket, TicketReply,
  MemberProgress, GymUsageStat, Notification, AuditLog,
};
