import React, { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform, useSpring } from 'framer-motion';
import {
  Users, Calendar, Clock, ArrowRight, Flame, Dumbbell,
  Zap, TrendingUp, Play, ChevronRight, Sparkles, Timer, Sun, Moon, Sunrise,
} from 'lucide-react';
import PageShell from '../components/PageShell';
import api from '../services/api';

/* ─── Helpers ──────────────────────────────────────────────── */
const statusColors = {
  scheduled: 'bg-sky-100 text-sky-700 border-sky-200',
  'in-progress': 'bg-amber-100 text-amber-700 border-amber-200',
  completed: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  cancelled: 'bg-slate-100 text-slate-500 border-slate-200',
};

function getGreeting() {
  const h = new Date().getHours();
  if (h < 12) return { text: 'Good Morning', emoji: '🌅', Icon: Sunrise, color: 'from-amber-400 to-orange-500' };
  if (h < 17) return { text: 'Good Afternoon', emoji: '☀️', Icon: Sun, color: 'from-sky-400 to-blue-500' };
  return { text: 'Good Evening', emoji: '🌙', Icon: Moon, color: 'from-indigo-400 to-purple-500' };
}

const motivationalTips = [
  "Energy is contagious — bring yours to every session! 🔥",
  "Small wins build unstoppable momentum. Celebrate them! 🏆",
  "Your members mirror your energy. Shine bright today! ✨",
  "The best coaches never stop learning. Stay curious! 📚",
  "Progress isn't always linear — consistency is the real magic. 💪",
  "A great warm-up sets the tone for an epic session! 🚀",
];

/* ─── Floating Fitness Emojis (background delight) ─────────── */
const floatingEmojis = ['💪', '🔥', '⚡', '🏋️', '🎯', '✨', '🚀', '💎'];
function FloatingEmojis() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      {floatingEmojis.map((emoji, i) => (
        <motion.div
          key={i}
          className="absolute text-lg opacity-[0.06] select-none"
          style={{ left: `${10 + (i * 12) % 85}%`, top: `${5 + (i * 17) % 80}%` }}
          animate={{
            y: [0, -30, 0],
            x: [0, i % 2 === 0 ? 15 : -15, 0],
            rotate: [0, i % 2 === 0 ? 20 : -20, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{ repeat: Infinity, duration: 5 + i * 0.7, ease: 'easeInOut', delay: i * 0.4 }}
        >
          {emoji}
        </motion.div>
      ))}
    </div>
  );
}

/* ─── Animated Progress Ring (SVG) ─────────────────────────── */
function ProgressRing({ progress, size = 100, strokeWidth = 8, color = '#6366f1', label, value, delay = 0 }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const springProgress = useSpring(0, { stiffness: 40, damping: 15 });
  const strokeDashoffset = useTransform(springProgress, (v) => circumference - (v / 100) * circumference);

  useEffect(() => {
    const timer = setTimeout(() => springProgress.set(Math.min(progress, 100)), 300 + delay);
    return () => clearTimeout(timer);
  }, [progress, springProgress, delay]);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="currentColor"
            strokeWidth={strokeWidth} className="text-slate-100" />
          <motion.circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={color}
            strokeWidth={strokeWidth} strokeLinecap="round"
            style={{ strokeDasharray: circumference, strokeDashoffset }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-lg font-bold text-slate-800">{value}</span>
        </div>
      </div>
      <span className="text-xs font-medium text-slate-500">{label}</span>
    </div>
  );
}

/* ─── Countdown Timer ──────────────────────────────────────── */
function CountdownTimer({ targetDate }) {
  const [timeLeft, setTimeLeft] = useState('');
  useEffect(() => {
    const update = () => {
      const now = new Date();
      const diff = new Date(targetDate) - now;
      if (diff <= 0) { setTimeLeft('Now!'); return; }
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setTimeLeft(`${h > 0 ? h + 'h ' : ''}${m}m ${s}s`);
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [targetDate]);
  return <span className="tabular-nums font-mono">{timeLeft}</span>;
}

/* ─── Weekly Mini Calendar ─────────────────────────────────── */
function WeeklyMiniCalendar({ sessions }) {
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const today = new Date().getDay();

  const sessionsByDay = useMemo(() => {
    const counts = [0, 0, 0, 0, 0, 0, 0];
    const now = new Date();
    const weekStart = new Date(now);
    weekStart.setDate(now.getDate() - now.getDay());
    weekStart.setHours(0, 0, 0, 0);
    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekStart.getDate() + 7);
    sessions.forEach((s) => {
      const d = s.sessionDate ? new Date(s.sessionDate) : null;
      if (d && d >= weekStart && d < weekEnd) counts[d.getDay()]++;
    });
    return counts;
  }, [sessions]);

  const maxCount = Math.max(...sessionsByDay, 1);

  return (
    <div className="flex items-end gap-2 justify-between h-full">
      {days.map((day, i) => {
        const count = sessionsByDay[i];
        const heightPct = count > 0 ? Math.max(20, (count / maxCount) * 100) : 8;
        const isToday = i === today;
        return (
          <motion.div key={day} className="flex flex-col items-center gap-1.5 flex-1"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 * i }}>
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: `${heightPct}%` }}
              transition={{ delay: 0.3 + i * 0.08, type: 'spring', stiffness: 60 }}
              className={`w-full max-w-[28px] rounded-t-md min-h-[4px] relative ${
                isToday
                  ? 'bg-gradient-to-t from-indigo-500 to-violet-400 shadow-lg shadow-indigo-200'
                  : count > 0 ? 'bg-gradient-to-t from-slate-300 to-slate-200' : 'bg-slate-100'
              }`}
              style={{ minHeight: 4 }}
            >
              {count > 0 && (
                <motion.div initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.6 + i * 0.08 }}
                  className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-bold text-slate-600">
                  {count}
                </motion.div>
              )}
            </motion.div>
            <span className={`text-[10px] font-medium ${isToday ? 'text-indigo-600 font-bold' : 'text-slate-400'}`}>
              {day}
            </span>
            {isToday && <div className="h-1 w-1 rounded-full bg-indigo-500" />}
          </motion.div>
        );
      })}
    </div>
  );
}

/* ─── Quick Nav Card (3D tilt) ─────────────────────────────── */
function QuickNavCard({ icon: Icon, label, desc, gradient, onClick, delay = 0 }) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useSpring(useTransform(y, [-50, 50], [8, -8]), { stiffness: 200, damping: 20 });
  const rotateY = useSpring(useTransform(x, [-50, 50], [-8, 8]), { stiffness: 200, damping: 20 });

  const handleMouse = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    x.set(e.clientX - rect.left - rect.width / 2);
    y.set(e.clientY - rect.top - rect.height / 2);
  };
  const resetMouse = () => { x.set(0); y.set(0); };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ delay, type: 'spring', stiffness: 100 }}
      whileHover={{ y: -6, scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      onMouseMove={handleMouse}
      onMouseLeave={resetMouse}
      onClick={onClick}
      style={{ rotateX, rotateY, transformPerspective: 600 }}
      className="group relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-5 cursor-pointer hover:shadow-xl hover:border-slate-300 transition-shadow duration-300"
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-[0.06] transition-opacity duration-500`} />
      <div className="absolute -right-6 -top-6 h-20 w-20 rounded-full bg-gradient-to-br from-slate-50 to-white group-hover:scale-150 transition-transform duration-500 opacity-60" />
      <div className="relative">
        <motion.div whileHover={{ rotate: 12, scale: 1.1 }}
          className={`h-11 w-11 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center mb-3 shadow-md`}>
          <Icon className="h-5 w-5 text-white" />
        </motion.div>
        <div className="text-slate-800 font-bold text-sm">{label}</div>
        <div className="text-slate-400 text-xs mt-0.5">{desc}</div>
      </div>
      <motion.div className="absolute bottom-4 right-4"
        animate={{ x: [0, 3, 0] }} transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}>
        <ChevronRight className="h-4 w-4 text-slate-300 group-hover:text-slate-500 transition-colors" />
      </motion.div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function TrainerDashboard({ onTabChange }) {
  const [stats, setStats] = useState({});
  const [sessions, setSessions] = useState([]);
  const [classes, setClasses] = useState([]);
  const [tipIdx, setTipIdx] = useState(0);
  const greeting = getGreeting();

  useEffect(() => {
    let mounted = true;
    api.get('/api/stats').then((res) => { if (mounted) setStats(res.data); }).catch(() => null);
    Promise.all([api.get('/api/trainer/sessions'), api.get('/api/trainer/classes')])
      .then(([s, c]) => { if (mounted) { setSessions(s.data || []); setClasses(c.data || []); } })
      .catch(() => null);
    return () => { mounted = false; };
  }, []);

  // Rotate motivational tip
  useEffect(() => {
    const id = setInterval(() => setTipIdx((i) => (i + 1) % motivationalTips.length), 8000);
    return () => clearInterval(id);
  }, []);

  const todaySessions = useMemo(() => {
    const today = new Date().toISOString().split('T')[0];
    return sessions.filter((s) => {
      const d = s.sessionDate ? new Date(s.sessionDate).toISOString().split('T')[0] : '';
      return d === today;
    });
  }, [sessions]);

  const completedToday = useMemo(() => todaySessions.filter((s) => s.status === 'completed').length, [todaySessions]);

  const upcoming = useMemo(() => {
    const now = new Date();
    now.setHours(0, 0, 0, 0);
    return sessions
      .filter((s) => {
        const d = s.sessionDate ? new Date(s.sessionDate) : null;
        return d && d >= now && s.status !== 'completed' && s.status !== 'cancelled';
      })
      .sort((a, b) => new Date(a.sessionDate) - new Date(b.sessionDate))
      .slice(0, 5);
  }, [sessions]);

  const nextSession = useMemo(() => {
    const now = new Date();
    return sessions.find((s) => {
      const d = s.sessionDate ? new Date(s.sessionDate) : null;
      return d && d > now && s.status !== 'completed' && s.status !== 'cancelled';
    });
  }, [sessions]);

  const completionRate = useMemo(() => {
    if (sessions.length === 0) return 0;
    const completed = sessions.filter((s) => s.status === 'completed').length;
    return Math.round((completed / sessions.length) * 100);
  }, [sessions]);

  return (
    <PageShell title="Trainer Portal" subtitle="Your overview and today's schedule.">
      <FloatingEmojis />
      {/* ─── HERO WITH TIME-OF-DAY GREETING ─────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="card-hero relative overflow-hidden text-white shine"
        style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)' }}
      >
        <div className="absolute inset-0 opacity-20 animate-gradient-x" style={{ background: 'linear-gradient(90deg, transparent, rgba(245,158,11,0.3), transparent, rgba(16,185,129,0.3), transparent)', backgroundSize: '300% 100%' }} />
        <div className="absolute -right-20 -top-20 h-56 w-56 rounded-full bg-amber-500/10 blur-2xl" />
        <div className="absolute -left-10 -bottom-10 h-40 w-40 rounded-full bg-emerald-500/10 blur-2xl" />

        <div className="relative flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-1.5 text-xs text-white/60 bg-white/10 rounded-full px-3 py-1 border border-white/10">
              <Flame className="h-3 w-3" /> Training Hub
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
              className="text-2xl sm:text-3xl font-bold mt-3 flex items-center gap-3">
              {greeting.text}, Coach!{' '}
              <motion.span animate={{ rotate: [0, 14, -14, 0] }}
                transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                className="text-3xl">{greeting.emoji}</motion.span>
            </motion.div>
            <div className="text-white/60 mt-2">
              You have <span className="text-white font-semibold">{todaySessions.length}</span> sessions today
              {todaySessions.length > 0 && (
                <> — <span className="text-emerald-400 font-semibold">{completedToday}</span> completed</>
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Sessions', value: sessions.length, icon: '📋' },
              { label: 'Classes', value: classes.length, icon: '🏋️' },
              { label: 'Today', value: todaySessions.length, icon: '🎯' },
            ].map((item, i) => (
              <motion.div key={item.label}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                whileHover={{ scale: 1.05, y: -2 }}
                className="rounded-xl bg-white/[0.08] border border-white/[0.12] px-4 py-3 backdrop-blur-sm hover:bg-white/[0.14] transition-all cursor-default">
                <div className="text-xs text-white/50 flex items-center gap-1">
                  <span>{item.icon}</span> {item.label}
                </div>
                <div className="text-xl font-bold mt-0.5">{item.value}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ─── PROGRESS RINGS + NEXT SESSION COUNTDOWN ──────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }} className="card lg:col-span-2">
          <div className="flex items-center gap-2 mb-5">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center">
              <TrendingUp className="h-4 w-4 text-white" />
            </div>
            <h2 className="text-slate-800 text-lg font-bold">Performance Snapshot</h2>
          </div>
          <div className="flex items-center justify-around flex-wrap gap-6">
            <ProgressRing progress={completionRate} size={110} strokeWidth={10}
              color="#6366f1" label="Completion Rate" value={`${completionRate}%`} delay={0} />
            <ProgressRing
              progress={sessions.length > 0 ? Math.min((todaySessions.length / Math.max(sessions.length / 7, 1)) * 100, 100) : 0}
              size={110} strokeWidth={10} color="#f59e0b"
              label="Today's Load" value={`${todaySessions.length}`} delay={150} />
            <ProgressRing progress={classes.length > 0 ? 100 : 0} size={110} strokeWidth={10}
              color="#10b981" label="Active Classes" value={`${classes.length}`} delay={300} />
          </div>
        </motion.div>

        {/* Next session countdown */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }} className="card relative overflow-hidden">
          <div className="absolute -right-8 -top-8 h-24 w-24 rounded-full bg-gradient-to-br from-amber-500/10 to-orange-500/10 blur-xl" />
          <div className="relative">
            <div className="flex items-center gap-2 mb-4">
              <motion.div animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                <Timer className="h-4 w-4 text-white" />
              </motion.div>
              <h2 className="text-slate-800 font-bold">Next Session</h2>
            </div>
            {nextSession ? (
              <div className="space-y-3">
                <div className="text-slate-700 font-semibold">{nextSession.member_name || 'Session'}</div>
                <div className="text-slate-500 text-sm flex items-center gap-1.5">
                  <Dumbbell className="h-3.5 w-3.5" /> {nextSession.sessionType || 'Training'}
                </div>
                <div className="text-slate-500 text-sm flex items-center gap-1.5">
                  <Clock className="h-3.5 w-3.5" /> {nextSession.startTime || '—'} – {nextSession.endTime || '—'}
                </div>
                <div className="mt-2 pt-3 border-t border-slate-100">
                  <div className="text-xs text-slate-400 mb-1">Starts in</div>
                  <div className="text-2xl font-bold bg-gradient-to-r from-amber-600 to-orange-500 bg-clip-text text-transparent">
                    <CountdownTimer targetDate={nextSession.sessionDate} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-4">
                <motion.div animate={{ y: [0, -6, 0] }}
                  transition={{ repeat: Infinity, duration: 2.5, ease: 'easeInOut' }}
                  className="text-4xl mb-2">🧘</motion.div>
                <div className="text-slate-500 text-sm">No upcoming sessions</div>
                <div className="text-slate-400 text-xs mt-1">Enjoy your free time!</div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* ─── QUICK NAV — Connected to tabs ─────────────────── */}
      <div>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          className="flex items-center gap-2 mb-3">
          <Zap className="h-4 w-4 text-amber-500" />
          <h2 className="text-slate-800 text-lg font-bold">Quick Actions</h2>
        </motion.div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <QuickNavCard icon={Play} label="My Sessions"
            desc="View & manage sessions" gradient="from-indigo-500 to-violet-500"
            onClick={() => onTabChange?.('sessions')} delay={0.35} />
          <QuickNavCard icon={Calendar} label="My Classes"
            desc="Schedule & attendance" gradient="from-emerald-500 to-teal-500"
            onClick={() => onTabChange?.('classes')} delay={0.4} />
          <QuickNavCard icon={Users} label="Members"
            desc="Browse member list" gradient="from-amber-500 to-orange-500"
            onClick={() => onTabChange?.('members')} delay={0.45} />
        </div>
      </div>

      {/* ─── WEEKLY CALENDAR + SCHEDULE ────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }} className="card">
          <div className="flex items-center gap-2 mb-4">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center">
              <Calendar className="h-4 w-4 text-white" />
            </div>
            <h2 className="text-slate-800 font-bold">This Week</h2>
          </div>
          <div className="h-32">
            <WeeklyMiniCalendar sessions={sessions} />
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }} className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-sky-500 to-blue-500 flex items-center justify-center">
                <Clock className="h-4 w-4 text-white" />
              </div>
              <h2 className="text-slate-800 font-bold">Upcoming Schedule</h2>
            </div>
            <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              onClick={() => onTabChange?.('sessions')}
              className="text-xs text-indigo-500 hover:text-indigo-700 font-medium flex items-center gap-1 transition-colors">
              View all <ArrowRight className="h-3 w-3" />
            </motion.button>
          </div>
          <div className="space-y-0">
            <AnimatePresence>
              {upcoming.map((s, i) => (
                <motion.div
                  key={s._id || i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className="group relative flex items-stretch gap-4"
                >
                  <div className="flex flex-col items-center">
                    <motion.div animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 3, delay: i * 0.5 }}
                      className="h-3 w-3 rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 ring-4 ring-indigo-100 mt-5 shrink-0" />
                    {i < upcoming.length - 1 && (
                      <div className="flex-1 w-0.5 bg-gradient-to-b from-indigo-200 to-slate-200 my-1" />
                    )}
                  </div>
                  <motion.div whileHover={{ x: 4, borderColor: 'rgb(165,180,252)' }}
                    className="flex-1 rounded-xl border border-slate-200 bg-white p-4 mb-3 flex items-center justify-between hover:shadow-md transition-all duration-200 cursor-default">
                    <div>
                      <div className="text-slate-900 font-semibold">{s.member_name || 'Session'}</div>
                      <div className="text-slate-500 text-sm flex items-center gap-2 mt-0.5">
                        <Dumbbell className="h-3.5 w-3.5" />
                        {s.sessionType || 'Session'} • {s.durationMinutes || 0} min
                      </div>
                      <div className="text-slate-400 text-xs mt-1 flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {s.sessionDate ? new Date(s.sessionDate).toLocaleDateString() : ''}
                      </div>
                    </div>
                    <div className="text-right flex flex-col items-end gap-1.5">
                      <div className="text-slate-600 text-sm font-medium flex items-center gap-1">
                        <Clock className="h-3.5 w-3.5" />
                        {s.startTime || '—'} – {s.endTime || '—'}
                      </div>
                      <span className={`text-xs px-2.5 py-0.5 rounded-full border font-medium ${statusColors[s.status] || statusColors.scheduled}`}>
                        {s.status}
                      </span>
                    </div>
                  </motion.div>
                </motion.div>
              ))}
            </AnimatePresence>
            {upcoming.length === 0 && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="text-center py-10">
                <motion.div animate={{ y: [0, -8, 0], rotate: [0, 5, -5, 0] }}
                  transition={{ repeat: Infinity, duration: 4, ease: 'easeInOut' }}
                  className="text-5xl mb-3">🏖️</motion.div>
                <div className="text-slate-500 font-medium">All clear!</div>
                <div className="text-slate-400 text-sm mt-1">No upcoming sessions scheduled</div>
                <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={() => onTabChange?.('sessions')}
                  className="mt-4 text-sm text-indigo-500 hover:text-indigo-700 font-medium inline-flex items-center gap-1">
                  Go to Sessions <ArrowRight className="h-3.5 w-3.5" />
                </motion.button>
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>

      {/* ─── MOTIVATION TICKER ─────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
        className="relative overflow-hidden rounded-2xl border border-slate-200 bg-gradient-to-r from-indigo-50 via-white to-violet-50 p-5">
        <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br from-indigo-100 to-violet-100 blur-xl" />
        <div className="relative flex items-center gap-4">
          <motion.div animate={{ rotate: [0, 360] }}
            transition={{ repeat: Infinity, duration: 20, ease: 'linear' }}
            className="h-10 w-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center shrink-0">
            <Sparkles className="h-5 w-5 text-white" />
          </motion.div>
          <div className="min-w-0 flex-1">
            <div className="text-xs text-indigo-500 font-semibold mb-1">Coach Tip of the Hour</div>
            <AnimatePresence mode="wait">
              <motion.div key={tipIdx}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.4 }}
                className="text-slate-700 font-medium text-sm">
                {motivationalTips[tipIdx]}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </PageShell>
  );
}
