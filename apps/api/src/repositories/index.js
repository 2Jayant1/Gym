/**
 * Repository index — one-liner access to all repositories.
 *
 * Usage:
 *   const repos = require('./repositories')(models);
 *   const user = await repos.user.findById(id);
 */
const BaseRepository = require('./BaseRepository');

function createRepositories(models) {
  return {
    user:                new BaseRepository(models.User),
    membershipPlan:      new BaseRepository(models.MembershipPlan),
    attendance:          new BaseRepository(models.Attendance),
    accessLog:           new BaseRepository(models.AccessLog),
    refreshToken:        new BaseRepository(models.RefreshToken),
    invoice:             new BaseRepository(models.Invoice),
    payment:             new BaseRepository(models.Payment),
    trainer:             new BaseRepository(models.Trainer),
    trainingSession:     new BaseRepository(models.TrainingSession),
    workoutPlan:         new BaseRepository(models.WorkoutPlan),
    exercise:            new BaseRepository(models.Exercise),
    workoutExercise:     new BaseRepository(models.WorkoutExercise),
    equipment:           new BaseRepository(models.Equipment),
    equipmentMaintenance:new BaseRepository(models.EquipmentMaintenance),
    fitnessClass:        new BaseRepository(models.FitnessClass),
    classSchedule:       new BaseRepository(models.ClassSchedule),
    classRegistration:   new BaseRepository(models.ClassRegistration),
    supportCategory:     new BaseRepository(models.SupportCategory),
    supportTicket:       new BaseRepository(models.SupportTicket),
    ticketReply:         new BaseRepository(models.TicketReply),
    memberProgress:      new BaseRepository(models.MemberProgress),
    gymUsageStat:        new BaseRepository(models.GymUsageStat),
    notification:        new BaseRepository(models.Notification),
    auditLog:            new BaseRepository(models.AuditLog),
  };
}

module.exports = createRepositories;
