const bcrypt = require('bcryptjs');

function addDays(date, days) {
  const d = new Date(date);
  d.setDate(d.getDate() + days);
  return d;
}

function isoDate(d) {
  return new Date(d).toISOString().split('T')[0];
}

function rand(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

async function seedIfEmpty(models) {
  const {
    MembershipPlan, User, Trainer, FitnessClass, ClassSchedule, Exercise,
    WorkoutPlan, WorkoutExercise, SupportCategory, Attendance, AccessLog,
    Invoice, Payment, MemberProgress, SupportTicket, TicketReply,
    Equipment, EquipmentMaintenance, Notification, ClassRegistration,
  } = models;

  const existing = await User.countDocuments();
  if (existing > 0) {
    console.log('Seed skipped (users already exist).');
    return { skipped: true };
  }

  const now = new Date();

  // ─── Plans ───────────────────────────────────────────────
  const plans = await MembershipPlan.insertMany([
    { name: 'Basic', description: 'Access to gym floor and cardio equipment.', monthlyFee: 29.99, durationDays: 30, maxVisitsPerWeek: 7, includesTrainer: false, includesClasses: false },
    { name: 'Standard', description: 'Gym access plus group classes.', monthlyFee: 49.99, durationDays: 30, maxVisitsPerWeek: 14, includesTrainer: false, includesClasses: true },
    { name: 'Premium', description: 'Unlimited access, personal trainer, and all classes.', monthlyFee: 79.99, durationDays: 30, includesTrainer: true, includesClasses: true, signupFee: 10, cancellationFee: 25 },
  ]);
  const [basicPlan, standardPlan, premiumPlan] = plans;

  // ─── Users ───────────────────────────────────────────────
  const adminUser = await User.create({
    username: 'admin', email: 'admin@gym.local', password: 'Admin123',
    role: 'admin', name: 'Admin User', phone: '555-0100', membershipPlan: premiumPlan._id,
  });

  const trainerUser = await User.create({
    username: 'trainer', email: 'trainer@gym.local', password: 'Trainer123',
    role: 'trainer', name: 'Coach Alex', phone: '555-0200', membershipPlan: premiumPlan._id,
    fitnessGoals: 'Help members achieve their potential',
  });

  const trainerUser2 = await User.create({
    username: 'sarah', email: 'sarah@gym.local', password: 'Trainer123',
    role: 'trainer', name: 'Sarah Miller', phone: '555-0201', membershipPlan: premiumPlan._id,
    fitnessGoals: 'Yoga and mindfulness coaching',
  });

  const memberUser = await User.create({
    username: 'member', email: 'member@gym.local', password: 'Member123',
    role: 'member', name: 'Jane Smith', phone: '555-0300', membershipPlan: standardPlan._id,
    heightCm: 168, weightKg: 65, fitnessGoals: 'Improve general fitness and flexibility',
    gender: 'female', dateOfBirth: new Date('1995-06-15'),
  });

  // Additional members
  const memberNames = [
    { name: 'Mike Johnson', username: 'mikej', email: 'mike@gym.local', gender: 'male', height: 180, weight: 82, goal: 'muscle_gain' },
    { name: 'Emily Davis', username: 'emilyd', email: 'emily@gym.local', gender: 'female', height: 165, weight: 58, goal: 'weight_loss' },
    { name: 'David Wilson', username: 'davidw', email: 'david@gym.local', gender: 'male', height: 175, weight: 90, goal: 'endurance' },
    { name: 'Sophia Brown', username: 'sophiab', email: 'sophia@gym.local', gender: 'female', height: 170, weight: 62, goal: 'general_fitness' },
    { name: 'James Lee', username: 'jamesl', email: 'james@gym.local', gender: 'male', height: 185, weight: 78, goal: 'muscle_gain' },
    { name: 'Olivia Garcia', username: 'oliviag', email: 'olivia@gym.local', gender: 'female', height: 160, weight: 55, goal: 'weight_loss' },
    { name: 'Liam Martinez', username: 'liamm', email: 'liam@gym.local', gender: 'male', height: 178, weight: 85, goal: 'endurance' },
    { name: 'Ava Robinson', username: 'avar', email: 'ava@gym.local', gender: 'female', height: 172, weight: 67, goal: 'general_fitness' },
  ];

  const extraMembers = [];
  for (const m of memberNames) {
    const u = await User.create({
      username: m.username, email: m.email, password: 'Member123',
      role: 'member', name: m.name, phone: `555-0${300 + extraMembers.length + 1}`,
      membershipPlan: rand([basicPlan._id, standardPlan._id, premiumPlan._id]),
      heightCm: m.height, weightKg: m.weight, gender: m.gender,
      fitnessGoals: m.goal, dateOfBirth: new Date(`199${Math.floor(Math.random() * 5) + 2}-0${Math.floor(Math.random() * 9) + 1}-15`),
    });
    extraMembers.push(u);
  }

  const allMembers = [memberUser, ...extraMembers];

  // ─── Trainers ────────────────────────────────────────────
  const trainer = await Trainer.create({
    user: trainerUser._id, specialization: 'Strength & Conditioning',
    certification: 'NASM-CPT', hourlyRate: 40, experienceYears: 5,
    bio: 'Focused on sustainable strength and mobility for all fitness levels.',
  });

  const trainer2 = await Trainer.create({
    user: trainerUser2._id, specialization: 'Yoga & Pilates',
    certification: 'RYT-200', hourlyRate: 35, experienceYears: 8,
    bio: 'Certified yoga instructor emphasizing balance, flexibility, and mindfulness.',
  });

  // ─── Exercises ───────────────────────────────────────────
  const exercises = await Exercise.insertMany([
    { name: 'Back Squat', description: 'Barbell back squat', muscleGroup: 'legs', equipmentNeeded: 'barbell', difficultyLevel: 'intermediate' },
    { name: 'Bench Press', description: 'Flat barbell bench press', muscleGroup: 'chest', equipmentNeeded: 'barbell', difficultyLevel: 'intermediate' },
    { name: 'Deadlift', description: 'Conventional barbell deadlift', muscleGroup: 'back', equipmentNeeded: 'barbell', difficultyLevel: 'advanced' },
    { name: 'Pull-ups', description: 'Bodyweight pull-ups', muscleGroup: 'back', equipmentNeeded: 'pull-up bar', difficultyLevel: 'intermediate' },
    { name: 'Plank', description: 'Core stabilization exercise', muscleGroup: 'core', equipmentNeeded: 'none', difficultyLevel: 'beginner' },
    { name: 'Dumbbell Lunges', description: 'Walking lunges with dumbbells', muscleGroup: 'legs', equipmentNeeded: 'dumbbells', difficultyLevel: 'beginner' },
    { name: 'Overhead Press', description: 'Standing barbell overhead press', muscleGroup: 'shoulders', equipmentNeeded: 'barbell', difficultyLevel: 'intermediate' },
    { name: 'Bicep Curls', description: 'Dumbbell bicep curls', muscleGroup: 'arms', equipmentNeeded: 'dumbbells', difficultyLevel: 'beginner' },
    { name: 'Lat Pulldown', description: 'Cable lat pulldown', muscleGroup: 'back', equipmentNeeded: 'cable machine', difficultyLevel: 'beginner' },
    { name: 'Treadmill Run', description: '20-min steady state', muscleGroup: 'cardio', equipmentNeeded: 'treadmill', difficultyLevel: 'beginner' },
  ]);

  // ─── Fitness Classes ─────────────────────────────────────
  const yogaClass = await FitnessClass.create({
    name: 'Morning Yoga', description: 'Mobility and recovery class for all levels.',
    trainer: trainer2._id, durationMinutes: 60, difficultyLevel: 'beginner',
    category: 'yoga', maxParticipants: 20,
  });

  const hiitClass = await FitnessClass.create({
    name: 'HIIT Blast', description: 'High intensity interval training to torch calories.',
    trainer: trainer._id, durationMinutes: 45, difficultyLevel: 'intermediate',
    category: 'cardio', maxParticipants: 15,
  });

  const strengthClass = await FitnessClass.create({
    name: 'Power Lift', description: 'Strength-focused group class with barbells.',
    trainer: trainer._id, durationMinutes: 50, difficultyLevel: 'advanced',
    category: 'strength', maxParticipants: 12,
  });

  const pilatesClass = await FitnessClass.create({
    name: 'Core Pilates', description: 'Mat Pilates focusing on core strength and stability.',
    trainer: trainer2._id, durationMinutes: 55, difficultyLevel: 'beginner',
    category: 'yoga', maxParticipants: 18,
  });

  // ─── Class Schedules ─────────────────────────────────────
  const schedules = [];
  for (let i = -3; i < 7; i++) {
    const d = addDays(now, i);
    const s1 = await ClassSchedule.create({ fitnessClass: yogaClass._id, classDate: d, startTime: '07:00', endTime: '08:00', room: 'Studio A' });
    const s2 = await ClassSchedule.create({ fitnessClass: hiitClass._id, classDate: d, startTime: '09:00', endTime: '09:45', room: 'Gym Floor' });
    const s3 = await ClassSchedule.create({ fitnessClass: strengthClass._id, classDate: d, startTime: '17:00', endTime: '17:50', room: 'Weight Room' });
    if (i % 2 === 0) {
      const s4 = await ClassSchedule.create({ fitnessClass: pilatesClass._id, classDate: d, startTime: '11:00', endTime: '11:55', room: 'Studio B' });
      schedules.push(s4);
    }
    schedules.push(s1, s2, s3);
  }

  // ─── Class Registrations ─────────────────────────────────
  for (let i = 0; i < Math.min(6, schedules.length); i++) {
    const membersToReg = allMembers.slice(0, 3 + i);
    for (const m of membersToReg) {
      await ClassRegistration.create({
        classSchedule: schedules[i]._id,
        user: m._id,
        status: i < 3 ? rand(['registered', 'attended']) : 'registered',
      }).catch(() => null);
    }
  }

  // ─── Workout Plans ───────────────────────────────────────
  const plan1 = await WorkoutPlan.create({
    user: memberUser._id, trainer: trainer._id, name: 'Beginner Strength',
    description: '3-day full-body strength program', goal: 'general_fitness',
    difficultyLevel: 'beginner', startDate: addDays(now, -14), endDate: addDays(now, 42),
  });

  await WorkoutExercise.insertMany([
    { workoutPlan: plan1._id, exercise: exercises[0]._id, dayOfWeek: 1, sets: 3, reps: 8, weightKg: 40, restSeconds: 120, sortOrder: 0 },
    { workoutPlan: plan1._id, exercise: exercises[1]._id, dayOfWeek: 1, sets: 3, reps: 10, weightKg: 30, restSeconds: 90, sortOrder: 1 },
    { workoutPlan: plan1._id, exercise: exercises[4]._id, dayOfWeek: 1, sets: 3, reps: 1, durationMinutes: 1, restSeconds: 60, sortOrder: 2, notes: 'Hold for 1 minute' },
    { workoutPlan: plan1._id, exercise: exercises[3]._id, dayOfWeek: 3, sets: 3, reps: 6, restSeconds: 120, sortOrder: 0 },
    { workoutPlan: plan1._id, exercise: exercises[6]._id, dayOfWeek: 3, sets: 3, reps: 8, weightKg: 25, restSeconds: 90, sortOrder: 1 },
    { workoutPlan: plan1._id, exercise: exercises[8]._id, dayOfWeek: 3, sets: 3, reps: 12, weightKg: 35, restSeconds: 60, sortOrder: 2 },
    { workoutPlan: plan1._id, exercise: exercises[5]._id, dayOfWeek: 5, sets: 3, reps: 10, weightKg: 12, restSeconds: 90, sortOrder: 0 },
    { workoutPlan: plan1._id, exercise: exercises[7]._id, dayOfWeek: 5, sets: 3, reps: 12, weightKg: 10, restSeconds: 60, sortOrder: 1 },
    { workoutPlan: plan1._id, exercise: exercises[9]._id, dayOfWeek: 5, sets: 1, durationMinutes: 20, restSeconds: 0, sortOrder: 2, notes: 'Moderate pace' },
  ]);

  // ─── Training Sessions ───────────────────────────────────
  for (let i = 0; i < 15; i++) {
    const member = allMembers[i % allMembers.length];
    const dayOffset = -(i % 10);
    const date = addDays(now, dayOffset);
    const startH = 9 + (i % 6);
    await models.TrainingSession.create({
      trainer: trainer._id, user: member._id,
      sessionDate: date, startTime: `${String(startH).padStart(2, '0')}:00`,
      endTime: `${String(startH).padStart(2, '0')}:45`, durationMinutes: 45,
      sessionType: rand(['personal', 'assessment', 'group']),
      status: dayOffset < -2 ? 'completed' : 'scheduled',
      notes: 'Focus: technique + consistency.', cost: 30,
      rating: dayOffset < -2 ? (3 + Math.floor(Math.random() * 3)) : undefined,
      feedback: dayOffset < -2 ? rand(['Great session!', 'Really helped with form.', 'Tough but productive.']) : undefined,
    });
  }

  // ─── Attendance ──────────────────────────────────────────
  for (const member of allMembers) {
    for (let i = 0; i < 8 + Math.floor(Math.random() * 10); i++) {
      const d = addDays(now, -(i * 2 + Math.floor(Math.random() * 3)));
      const checkInH = 6 + Math.floor(Math.random() * 8);
      const checkIn = new Date(d);
      checkIn.setHours(checkInH, Math.floor(Math.random() * 60), 0, 0);
      const checkOut = new Date(checkIn);
      checkOut.setMinutes(checkOut.getMinutes() + 45 + Math.floor(Math.random() * 60));

      await Attendance.create({
        user: member._id, checkIn, checkOut, date: d,
        durationMinutes: Math.round((checkOut - checkIn) / 60000),
      }).catch(() => null);

      await AccessLog.create({ user: member._id, accessTime: checkIn, accessType: 'check_in' }).catch(() => null);
      await AccessLog.create({ user: member._id, accessTime: checkOut, accessType: 'check_out' }).catch(() => null);
    }
  }

  // ─── Invoices & Payments ─────────────────────────────────
  for (let i = 0; i < allMembers.length; i++) {
    const member = allMembers[i];
    for (let j = 0; j < 3; j++) {
      const issue = addDays(now, -(j * 30 + Math.floor(Math.random() * 5)));
      const due = addDays(issue, 7);
      const amount = [29.99, 49.99, 79.99][i % 3];
      const tax = +(amount * 0.08).toFixed(2);
      const total = +(amount + tax).toFixed(2);
      const status = j === 0 ? rand(['pending', 'paid']) : 'paid';
      const invoiceNumber = `INV-${isoDate(issue).replace(/-/g, '')}-${String(i * 3 + j + 1).padStart(4, '0')}`;

      const inv = await Invoice.create({
        user: member._id, membershipPlan: member.membershipPlan,
        invoiceNumber, amount, issueDate: issue, dueDate: due,
        status, taxAmount: tax, discountAmount: 0, totalAmount: total,
      }).catch(() => null);

      if (inv && status === 'paid') {
        await Payment.create({
          invoice: inv._id, user: member._id, amount: total,
          paymentMethod: rand(['cash', 'card', 'online', 'transfer']),
          transactionId: `TX-${inv._id}-${Date.now()}-${i}-${j}`,
          status: 'completed',
        }).catch(() => null);
      }
    }
  }

  // ─── Member Progress ─────────────────────────────────────
  for (const member of allMembers.slice(0, 6)) {
    for (let w = 0; w < 8; w++) {
      const d = addDays(now, -w * 7);
      const baseW = member.weightKg || 70;
      const weight = +(baseW - w * 0.3 + (Math.random() * 2 - 1)).toFixed(1);
      const bmi = +(weight / ((member.heightCm || 170) / 100) ** 2).toFixed(1);
      await MemberProgress.create({
        user: member._id, recordDate: d,
        weightKg: weight, bodyFatPercentage: +(18 + Math.random() * 8).toFixed(1),
        muscleMassKg: +(25 + Math.random() * 8).toFixed(1),
        chestCm: +(88 + Math.random() * 10).toFixed(1), waistCm: +(72 + Math.random() * 10).toFixed(1),
        hipsCm: +(92 + Math.random() * 8).toFixed(1), bmi,
      }).catch(() => null);
    }
  }

  // ─── Equipment ───────────────────────────────────────────
  const eq1 = await Equipment.create({
    name: 'Treadmill X', category: 'Cardio', brand: 'ProRun', model: 'XR-200',
    serialNumber: 'SN-TR-0001', purchaseDate: addDays(now, -400), purchasePrice: 2500,
    status: 'active', lastMaintenanceDate: addDays(now, -30), nextMaintenanceDate: addDays(now, 60), location: 'Cardio Zone',
  });
  await Equipment.create({
    name: 'Smith Machine', category: 'Strength', brand: 'IronForge', model: 'SM-500',
    serialNumber: 'SN-SM-0001', purchaseDate: addDays(now, -300), purchasePrice: 4500,
    status: 'active', location: 'Weight Room',
  });
  await Equipment.create({
    name: 'Rowing Machine', category: 'Cardio', brand: 'AquaRow', model: 'AR-100',
    serialNumber: 'SN-RM-0001', purchaseDate: addDays(now, -200), purchasePrice: 1800,
    status: 'maintenance', location: 'Cardio Zone', notes: 'Chain needs replacement',
  });

  await EquipmentMaintenance.create({
    equipment: eq1._id, maintenanceDate: addDays(now, -30),
    maintenanceType: 'routine', description: 'Belt inspection and lubrication',
    cost: 120, technician: 'Technician A', nextMaintenanceDate: addDays(now, 60), status: 'completed',
  });

  // ─── Support Categories & Tickets ────────────────────────
  const cats = await SupportCategory.insertMany([
    { name: 'Billing', description: 'Invoices, payments, and refunds' },
    { name: 'Technical', description: 'App access and technical issues' },
    { name: 'Membership', description: 'Plan changes, renewals, and cancellations' },
    { name: 'Equipment', description: 'Report broken or missing equipment' },
    { name: 'General', description: 'Other inquiries' },
  ]);

  const tickets = [];
  const ticketTitles = [
    'Invoice not showing up', 'Can\'t log in to app', 'Want to upgrade plan',
    'Broken treadmill belt', 'Question about class schedule', 'Refund request',
    'Trainer availability', 'Locker room issue',
  ];

  for (let i = 0; i < Math.min(8, allMembers.length); i++) {
    const t = await SupportTicket.create({
      user: allMembers[i]._id, category: cats[i % cats.length]._id,
      title: ticketTitles[i], message: 'I need help with this issue. Please assist.',
      priority: rand(['low', 'medium', 'high']),
      status: i < 3 ? 'resolved' : rand(['open', 'in_progress']),
      assignedTo: i % 2 === 0 ? adminUser._id : trainerUser._id,
      resolutionNotes: i < 3 ? 'Issue has been resolved. Thank you for your patience.' : undefined,
    });
    tickets.push(t);
  }

  // Add replies to first few tickets
  for (let i = 0; i < Math.min(4, tickets.length); i++) {
    await TicketReply.create({
      ticket: tickets[i]._id, user: adminUser._id,
      message: 'We are looking into this issue. Will update you shortly.', isInternal: false,
    });
    await TicketReply.create({
      ticket: tickets[i]._id, user: allMembers[i]._id,
      message: 'Thank you for the quick response!', isInternal: false,
    });
    if (i < 2) {
      await TicketReply.create({
        ticket: tickets[i]._id, user: adminUser._id,
        message: 'Internal note: escalated to management.', isInternal: true,
      });
    }
  }

  // ─── Notifications ───────────────────────────────────────
  for (const member of allMembers) {
    await Notification.create({
      user: member._id, title: 'Welcome to FitFlex!',
      message: 'Your account has been created. Start exploring your dashboard!',
      type: 'success', read: true,
    }).catch(() => null);

    await Notification.create({
      user: member._id, title: 'New class available',
      message: 'Core Pilates has been added to the schedule. Register now!',
      type: 'info',
    }).catch(() => null);
  }

  await Notification.create({
    user: adminUser._id, title: 'System ready',
    message: 'Gym management system has been initialized with seed data.',
    type: 'success',
  }).catch(() => null);

  console.log('Seed complete. Users:', 3 + allMembers.length);
  return { ok: true, adminUserId: adminUser.id };
}

module.exports = { seedIfEmpty };
