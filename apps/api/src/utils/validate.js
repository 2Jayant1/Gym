const { z } = require('zod');

// ─── Auth ────────────────────────────────────────────────────
// Strong password: min 8 chars, must contain uppercase, lowercase, digit
const strongPassword = z
  .string()
  .min(8, 'Password must be at least 8 characters')
  .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
  .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
  .regex(/[0-9]/, 'Password must contain at least one digit');

const loginSchema = z
  .object({
    email: z.string().email().optional(),
    username: z.string().min(2).optional(),
    password: z.string().min(1, 'Password is required'),
  })
  .refine((d) => d.email || d.username, { message: 'email or username is required' });

const registerSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters').max(100),
  username: z
    .string()
    .min(3, 'Username must be at least 3 characters')
    .max(30)
    .regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, underscores'),
  email: z.string().email('Invalid email address').max(254),
  password: strongPassword,
  phone: z.string().max(20).optional(),
  dateOfBirth: z.string().optional(),
  gender: z.enum(['male', 'female', 'other', '']).optional(),
  fitnessGoals: z.string().max(500).optional(),
});

// ─── Profile ─────────────────────────────────────────────────
const profileUpdateSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  phone: z.string().optional(),
  address: z.string().optional(),
  fitnessGoals: z.string().optional(),
  medicalConditions: z.string().optional(),
  timezone: z.string().optional(),
  emergencyContact: z.string().optional(),
  emergencyPhone: z.string().optional(),
  heightCm: z.coerce.number().positive().optional(),
  weightKg: z.coerce.number().positive().optional(),
  dateOfBirth: z.string().optional(),
  gender: z.enum(['male', 'female', 'other', '']).optional(),
});

const passwordChangeSchema = z.object({
  currentPassword: z.string().min(1, 'Current password is required'),
  newPassword: strongPassword,
});

// ─── Training ────────────────────────────────────────────────
const sessionCreateSchema = z.object({
  userId: z.string().min(1, 'userId is required'),
  sessionDate: z.string().min(1, 'sessionDate is required'),
  startTime: z.string().min(1, 'startTime is required'),
  endTime: z.string().min(1, 'endTime is required'),
  durationMinutes: z.number().int().positive('durationMinutes must be positive'),
  sessionType: z.enum(['personal', 'group', 'assessment']).optional(),
  status: z.enum(['scheduled', 'completed', 'cancelled', 'no_show']).optional(),
  notes: z.string().optional(),
  cost: z.number().min(0).optional(),
});

const classCreateSchema = z.object({
  name: z.string().min(1, 'name is required'),
  description: z.string().optional(),
  maxParticipants: z.number().int().positive().optional(),
  durationMinutes: z.number().int().positive('durationMinutes must be positive'),
  difficultyLevel: z.enum(['beginner', 'intermediate', 'advanced', '']).optional(),
  category: z.string().optional(),
  status: z.enum(['active', 'inactive']).optional(),
});

// ─── Progress ────────────────────────────────────────────────
const progressSchema = z
  .object({
    weightKg: z.coerce.number().positive('weightKg must be positive').optional(),
    bodyFatPercentage: z.coerce.number().min(0).max(100).optional(),
    muscleMassKg: z.coerce.number().positive().optional(),
    bmi: z.coerce.number().positive().optional(),
    notes: z.string().max(1000).optional(),
    measurements: z.record(z.coerce.number()).optional(),
    recordedAt: z.string().optional(),
  })
  .refine((d) => d.weightKg || d.bodyFatPercentage || d.muscleMassKg || d.bmi || d.notes || d.measurements, {
    message: 'At least one progress field is required',
  });

// ─── Tickets ─────────────────────────────────────────────────
const ticketCreateSchema = z.object({
  title: z.string().min(1, 'title is required'),
  message: z.string().min(1, 'message is required'),
  categoryId: z.string().optional(),
  priority: z.enum(['low', 'medium', 'high', 'urgent']).optional(),
});

const ticketReplySchema = z.object({
  message: z.string().min(1, 'message is required'),
});

// ─── Invoice ─────────────────────────────────────────────────
const invoiceCreateSchema = z.object({
  user: z.string().min(1, 'user is required'),
  membershipPlan: z.string().min(1, 'membershipPlan is required'),
  amount: z.number().min(0),
  taxAmount: z.number().min(0).optional(),
  discountAmount: z.number().min(0).optional(),
  totalAmount: z.number().min(0),
  issueDate: z.string().min(1),
  dueDate: z.string().min(1),
  notes: z.string().optional(),
});

// ─── Helpers ─────────────────────────────────────────────────

/**
 * Express middleware factory.
 * Usage:  router.post('/foo', validate(schema), handler)
 */
function validate(schema) {
  return (req, res, next) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      const messages = result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`);
      return res.status(400).json({ error: messages.join('; ') });
    }
    req.validated = result.data;
    next();
  };
}

module.exports = {
  loginSchema,
  registerSchema,
  profileUpdateSchema,
  passwordChangeSchema,
  progressSchema,
  sessionCreateSchema,
  classCreateSchema,
  ticketCreateSchema,
  ticketReplySchema,
  invoiceCreateSchema,
  validate,
};
