import React, { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence, useSpring, useTransform } from 'framer-motion';
import {
  Award, Heart, Clock, Edit2, Lock, Save, X, Sparkles, Activity, ArrowRight,
  Calendar, Flame, Target, TrendingUp, ChevronRight, Zap, FileText, LifeBuoy,
  Sun, Moon, Sunrise, Dumbbell,
} from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';

/* ─── Helpers ──────────────────────────────────────────────── */
function getGreeting() {
  const h = new Date().getHours();
  if (h < 12) return { text: 'Good Morning', emoji: '🌅', Icon: Sunrise };
  if (h < 17) return { text: 'Good Afternoon', emoji: '☀️', Icon: Sun };
  return { text: 'Good Evening', emoji: '🌙', Icon: Moon };
}

const motivationalQuotes = [
  { text: "The only bad workout is the one that didn't happen.", author: "Unknown" },
  { text: "Strength doesn't come from what you can do. It comes from overcoming what you thought you couldn't.", author: "Rikki Rogers" },
  { text: "Take care of your body. It's the only place you have to live.", author: "Jim Rohn" },
  { text: "The body achieves what the mind believes.", author: "Napoleon Hill" },
  { text: "Success isn't always about greatness. It's about consistency.", author: "Dwayne Johnson" },
  { text: "Your health is an investment, not an expense.", author: "Unknown" },
];

const achievements = [
  { emoji: '🔥', label: 'First Visit', desc: 'Checked in for the first time', threshold: 1 },
  { emoji: '⚡', label: 'Week Warrior', desc: '7 total visits', threshold: 7 },
  { emoji: '💎', label: 'Committed', desc: '30 total visits', threshold: 30 },
  { emoji: '🏆', label: 'Century Club', desc: '100 total visits', threshold: 100 },
  { emoji: '👑', label: 'Legend', desc: '365 total visits', threshold: 365 },
];

/* ─── XP Level System ──────────────────────────────────────── */
const xpLevels = [
  { level: 1, name: 'Newbie', minXp: 0, emoji: '🌱', color: 'from-slate-400 to-slate-500' },
  { level: 2, name: 'Regular', minXp: 50, emoji: '⚡', color: 'from-blue-400 to-blue-600' },
  { level: 3, name: 'Dedicated', minXp: 150, emoji: '💜', color: 'from-violet-400 to-violet-600' },
  { level: 4, name: 'Warrior', minXp: 350, emoji: '⚔️', color: 'from-amber-400 to-amber-600' },
  { level: 5, name: 'Champion', minXp: 700, emoji: '🏆', color: 'from-red-400 to-red-600' },
  { level: 6, name: 'Legend', minXp: 1500, emoji: '👑', color: 'from-amber-300 to-yellow-500' },
];

function getLevel(xp) {
  for (let i = xpLevels.length - 1; i >= 0; i--) {
    if (xp >= xpLevels[i].minXp) return xpLevels[i];
  }
  return xpLevels[0];
}

function getNextLevel(xp) {
  for (let i = 0; i < xpLevels.length; i++) {
    if (xp < xpLevels[i].minXp) return xpLevels[i];
  }
  return null;
}

/* ─── Floating Fitness Emojis (background delight) ─────────── */
const floatingEmojis = ['💪', '🔥', '⚡', '🎯', '✨', '🚀', '💎', '🏃'];
function FloatingEmojis() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      {floatingEmojis.map((emoji, i) => (
        <motion.div
          key={i}
          className="absolute text-lg opacity-[0.06] select-none"
          style={{ left: `${8 + (i * 13) % 85}%`, top: `${5 + (i * 15) % 80}%` }}
          animate={{
            y: [0, -25, 0],
            x: [0, i % 2 === 0 ? 12 : -12, 0],
            rotate: [0, i % 2 === 0 ? 15 : -15, 0],
            scale: [1, 1.15, 1],
          }}
          transition={{ repeat: Infinity, duration: 5 + i * 0.8, ease: 'easeInOut', delay: i * 0.5 }}
        >
          {emoji}
        </motion.div>
      ))}
    </div>
  );
}

/* ─── BMI Gauge (animated SVG arc) ─────────────────────────── */
function BmiGauge({ bmi }) {
  const numBmi = parseFloat(bmi);
  const isValid = !isNaN(numBmi) && numBmi > 0;
  // BMI range: 15–40 mapped to arc
  const minBmi = 15, maxBmi = 40;
  const clampedBmi = isValid ? Math.min(Math.max(numBmi, minBmi), maxBmi) : 22;
  const pct = ((clampedBmi - minBmi) / (maxBmi - minBmi)) * 100;

  const springPct = useSpring(0, { stiffness: 40, damping: 15 });
  useEffect(() => {
    const timer = setTimeout(() => springPct.set(pct), 500);
    return () => clearTimeout(timer);
  }, [pct, springPct]);

  // Arc params
  const size = 160, strokeW = 12;
  const radius = (size - strokeW) / 2;
  const startAngle = -210, endAngle = 30;
  const totalAngle = endAngle - startAngle; // 240°
  const circumference = (totalAngle / 360) * 2 * Math.PI * radius;
  const offset = useTransform(springPct, (v) => circumference - (v / 100) * circumference);

  // Color based on BMI
  let color = '#10b981'; // normal
  if (isValid) {
    if (numBmi < 18.5) color = '#f59e0b';
    else if (numBmi < 25) color = '#10b981';
    else if (numBmi < 30) color = '#f97316';
    else color = '#ef4444';
  }

  let category = 'Normal';
  if (isValid) {
    if (numBmi < 18.5) category = 'Underweight';
    else if (numBmi < 25) category = 'Normal';
    else if (numBmi < 30) category = 'Overweight';
    else category = 'Obese';
  }

  const polarToCartesian = (cx, cy, r, angleDeg) => {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };
  const start = polarToCartesian(size / 2, size / 2, radius, startAngle);
  const end = polarToCartesian(size / 2, size / 2, radius, endAngle);
  const largeArc = totalAngle > 180 ? 1 : 0;
  const bgPath = `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size * 0.72 }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
          style={{ marginTop: -size * 0.14 }}>
          <path d={bgPath} fill="none" stroke="#e2e8f0" strokeWidth={strokeW} strokeLinecap="round" />
          <motion.path d={bgPath} fill="none" stroke={color} strokeWidth={strokeW} strokeLinecap="round"
            style={{ strokeDasharray: circumference, strokeDashoffset: offset }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center" style={{ paddingTop: size * 0.1 }}>
          <div className="text-center">
            <div className="text-3xl font-bold text-slate-800">{isValid ? numBmi.toFixed(1) : '—'}</div>
            <div className="text-xs font-medium mt-0.5" style={{ color }}>{isValid ? category : 'N/A'}</div>
          </div>
        </div>
      </div>
      <div className="text-xs text-slate-400 -mt-1">Body Mass Index</div>
    </div>
  );
}

/* ─── Attendance Streak ────────────────────────────────────── */
function AttendanceStreak({ attendance }) {
  const streak = useMemo(() => {
    if (!attendance.length) return 0;
    const dates = [...new Set(attendance.map((a) => {
      const d = a.date ? new Date(a.date) : null;
      return d ? d.toISOString().split('T')[0] : null;
    }).filter(Boolean))].sort().reverse();

    let count = 0;
    const today = new Date();
    for (let i = 0; i < dates.length; i++) {
      const expected = new Date(today);
      expected.setDate(today.getDate() - i);
      if (dates[i] === expected.toISOString().split('T')[0]) {
        count++;
      } else break;
    }
    return count;
  }, [attendance]);

  return (
    <div className="text-center">
      <motion.div
        animate={streak > 0 ? { scale: [1, 1.15, 1], rotate: [0, 5, -5, 0] } : {}}
        transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}
        className="text-5xl mb-2"
      >
        {streak > 0 ? '🔥' : '❄️'}
      </motion.div>
      <div className="text-3xl font-bold text-slate-800">{streak}</div>
      <div className="text-xs text-slate-500 mt-0.5">day streak</div>
      {streak >= 7 && (
        <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
          className="mt-2 text-xs font-medium text-amber-600 bg-amber-50 rounded-full px-2 py-0.5 inline-block">
          🏅 On fire!
        </motion.div>
      )}
    </div>
  );
}

/* ─── Mini Heatmap (last 30 days) ──────────────────────────── */
function MiniHeatmap({ attendance }) {
  const days = useMemo(() => {
    const dateSet = new Set(attendance.map((a) => {
      const d = a.date ? new Date(a.date) : null;
      return d ? d.toISOString().split('T')[0] : null;
    }).filter(Boolean));

    const result = [];
    const today = new Date();
    for (let i = 29; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      const key = d.toISOString().split('T')[0];
      result.push({ date: key, active: dateSet.has(key), day: d.getDate(), isToday: i === 0 });
    }
    return result;
  }, [attendance]);

  return (
    <div>
      <div className="grid grid-cols-10 gap-1">
        {days.map((d, i) => (
          <motion.div
            key={d.date}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.02 }}
            whileHover={{ scale: 1.3 }}
            title={`${d.date}${d.active ? ' ✓' : ''}`}
            className={`aspect-square rounded-sm cursor-default flex items-center justify-center text-[7px] font-medium transition-colors ${
              d.isToday
                ? 'ring-1 ring-indigo-400 ' + (d.active ? 'bg-indigo-500 text-white' : 'bg-indigo-100 text-indigo-600')
                : d.active
                  ? 'bg-emerald-400 text-white'
                  : 'bg-slate-100 text-slate-300'
            }`}
          >
            {d.day}
          </motion.div>
        ))}
      </div>
      <div className="flex items-center justify-between mt-2 text-[10px] text-slate-400">
        <span>30 days ago</span>
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 rounded-sm bg-slate-100" />
          <span>Miss</span>
          <div className="h-2 w-2 rounded-sm bg-emerald-400 ml-1" />
          <span>Visit</span>
        </div>
        <span>Today</span>
      </div>
    </div>
  );
}

/* ─── Achievement Badge ────────────────────────────────────── */
function AchievementBadge({ emoji, label, desc, unlocked, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      whileHover={{ scale: 1.05, y: -2 }}
      className={`relative flex flex-col items-center gap-1 p-3 rounded-xl border transition-all duration-300 cursor-default ${
        unlocked
          ? 'bg-white border-amber-200 shadow-sm hover:shadow-md'
          : 'bg-slate-50 border-slate-100 opacity-40 grayscale'
      }`}
    >
      <motion.span
        className="text-2xl"
        animate={unlocked ? { scale: [1, 1.15, 1] } : {}}
        transition={{ repeat: Infinity, duration: 3, delay: delay * 2 }}
      >
        {emoji}
      </motion.span>
      <span className={`text-[10px] font-bold ${unlocked ? 'text-slate-700' : 'text-slate-400'}`}>{label}</span>
      <span className="text-[9px] text-slate-400 text-center leading-tight">{desc}</span>
      {unlocked && (
        <div className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-emerald-500 flex items-center justify-center">
          <span className="text-white text-[8px]">✓</span>
        </div>
      )}
    </motion.div>
  );
}

/* ─── Quick Action Card ────────────────────────────────────── */
function QuickAction({ icon: Icon, label, desc, gradient, onClick, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: 'spring', stiffness: 100 }}
      whileHover={{ y: -4, scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className="group relative overflow-hidden rounded-xl border border-slate-200 bg-white p-4 cursor-pointer hover:shadow-lg hover:border-slate-300 transition-all duration-300"
    >
      <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-[0.05] transition-opacity duration-500`} />
      <div className="relative flex items-center gap-3">
        <motion.div whileHover={{ rotate: 12, scale: 1.1 }}
          className={`h-10 w-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-sm shrink-0`}>
          <Icon className="h-4.5 w-4.5 text-white" />
        </motion.div>
        <div className="min-w-0 flex-1">
          <div className="text-slate-800 font-semibold text-sm">{label}</div>
          <div className="text-slate-400 text-xs">{desc}</div>
        </div>
        <motion.div animate={{ x: [0, 3, 0] }} transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}>
          <ChevronRight className="h-4 w-4 text-slate-300 group-hover:text-slate-500 transition-colors" />
        </motion.div>
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */
export default function MemberDashboard({ onTabChange }) {
  const [me, setMe] = useState(null);
  const [attendance, setAttendance] = useState([]);
  const [invoices, setInvoices] = useState([]);
  const [editing, setEditing] = useState(false);
  const [editForm, setEditForm] = useState({});
  const [showPwModal, setShowPwModal] = useState(false);
  const [pwForm, setPwForm] = useState({ currentPassword: '', newPassword: '' });
  const [busy, setBusy] = useState(false);
  const [quoteIdx, setQuoteIdx] = useState(0);
  const [questsDone, setQuestsDone] = useState(new Set());
  const toast = useToast();
  const greeting = getGreeting();

  function loadProfile() {
    api.get('/api/users/me').then((res) => {
      setMe(res.data);
      setEditForm({
        name: res.data?.name || '',
        phone: res.data?.phone || '',
        fitnessGoals: res.data?.fitness_goals || res.data?.fitnessGoals || '',
        heightCm: res.data?.height_cm || res.data?.heightCm || '',
        weightKg: res.data?.weight_kg || res.data?.weightKg || '',
      });
    }).catch((e) => console.error(e));
  }

  useEffect(() => {
    loadProfile();
    Promise.all([api.get('/api/me/attendance'), api.get('/api/me/invoices')])
      .then(([a, i]) => { setAttendance(a.data || []); setInvoices(i.data || []); })
      .catch(() => null);
  }, []);

  // Rotate quotes
  useEffect(() => {
    const id = setInterval(() => setQuoteIdx((i) => (i + 1) % motivationalQuotes.length), 10000);
    return () => clearInterval(id);
  }, []);

  const bmi = useMemo(() => {
    const h = me?.height_cm || me?.heightCm;
    const w = me?.weight_kg || me?.weightKg;
    return h && w ? (w / ((h / 100) ** 2)).toFixed(1) : '-';
  }, [me]);

  // Gamification: XP & Level
  const xp = attendance.length * 10;
  const currentLevel = getLevel(xp);
  const nextLevelInfo = getNextLevel(xp);
  const levelProgress = nextLevelInfo
    ? ((xp - currentLevel.minXp) / (nextLevelInfo.minXp - currentLevel.minXp)) * 100
    : 100;

  // Daily quests
  const todayStr = new Date().toISOString().split('T')[0];
  const checkedInToday = attendance.some((a) => {
    const d = a.date ? new Date(a.date).toISOString().split('T')[0] : '';
    return d === todayStr;
  });

  const dailyQuests = [
    { id: 'visit', emoji: '🏃', title: 'Visit the gym today', xpReward: 10, done: checkedInToday },
    { id: 'progress', emoji: '📈', title: 'Review your progress', xpReward: 5,
      done: questsDone.has('progress'),
      action: () => { onTabChange?.('progress'); setQuestsDone((s) => new Set(s).add('progress')); } },
    { id: 'workout', emoji: '💪', title: 'Check workout plans', xpReward: 5,
      done: questsDone.has('workout'),
      action: () => { onTabChange?.('workouts'); setQuestsDone((s) => new Set(s).add('workout')); } },
  ];

  const nextInvoice = invoices?.[0];
  const lastVisit = attendance?.[0];

  const saveProfile = async () => {
    setBusy(true);
    try {
      await api.patch('/api/me/profile', editForm);
      toast('Profile updated!', 'success');
      setEditing(false);
      loadProfile();
    } catch (e) {
      toast(e?.response?.data?.error || 'Failed to update profile', 'error');
    } finally { setBusy(false); }
  };

  const changePassword = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api.patch('/api/me/password', pwForm);
      toast('Password changed!', 'success');
      setShowPwModal(false);
      setPwForm({ currentPassword: '', newPassword: '' });
    } catch (e2) {
      toast(e2?.response?.data?.error || 'Failed to change password', 'error');
    } finally { setBusy(false); }
  };

  return (
    <PageShell title="My Dashboard" subtitle="Your membership, goals, and progress.">
      <FloatingEmojis />
      {/* ─── HERO WITH GREETING ─────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="card-hero relative overflow-hidden text-white shine"
        style={{ background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #c026d3 100%)' }}
      >
        <div className="absolute inset-0 opacity-20 animate-gradient-x" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent, rgba(255,255,255,0.1), transparent)', backgroundSize: '300% 100%' }} />
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-white/10 blur-2xl" />
        <div className="absolute -left-10 -bottom-10 h-48 w-48 rounded-full bg-white/5 blur-2xl" />
        <div className="relative flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-1.5 text-xs text-white/70 bg-white/15 rounded-full px-3 py-1 border border-white/20">
              <Sparkles className="h-3 w-3" /> Your Fitness Journey
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
              className="text-2xl sm:text-3xl font-bold mt-3 flex items-center gap-3">
              {greeting.text}, {me?.name?.split(' ')[0] || 'Champ'}!{' '}
              <motion.span animate={{ rotate: [0, 14, -14, 0] }}
                transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                className="text-3xl">{greeting.emoji}</motion.span>
            </motion.div>
            <div className="text-white/70 mt-2">
              {me?.plan_name ? (
                <>Your <span className="text-white font-semibold">{me.plan_name}</span> plan is active — keep pushing! 💪</>
              ) : 'Loading your plan...'}
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Total visits', value: attendance.length, icon: '🏃' },
              { label: 'Last visit', value: lastVisit?.date ? new Date(lastVisit.date).toLocaleDateString() : '—', icon: '📅' },
              { label: 'Next invoice', value: nextInvoice?.dueDate ? new Date(nextInvoice.dueDate).toLocaleDateString() : '—', icon: '💳' },
            ].map((item, i) => (
              <motion.div key={item.label}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                whileHover={{ scale: 1.05, y: -2 }}
                className="rounded-xl bg-white/[0.12] border border-white/[0.18] px-4 py-3 backdrop-blur-sm hover:bg-white/[0.18] transition-all cursor-default">
                <div className="text-xs text-white/60 flex items-center gap-1">
                  <span>{item.icon}</span> {item.label}
                </div>
                <div className="text-lg font-bold mt-0.5">{item.value}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ─── XP LEVEL + DAILY QUESTS ─────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* XP Level Card */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }} className="card lg:col-span-2 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/[0.03] via-violet-500/[0.03] to-purple-500/[0.03]" />
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <motion.div
                  animate={{ scale: [1, 1.1, 1], rotate: [0, 5, -5, 0] }}
                  transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                  className={`h-12 w-12 rounded-xl bg-gradient-to-br ${currentLevel.color} flex items-center justify-center shadow-lg`}>
                  <span className="text-xl">{currentLevel.emoji}</span>
                </motion.div>
                <div>
                  <div className="text-slate-800 font-bold text-lg">Level {currentLevel.level} — {currentLevel.name}</div>
                  <div className="text-xs text-slate-500">{xp} XP earned from {attendance.length} visits</div>
                </div>
              </div>
              {nextLevelInfo && (
                <div className="text-right hidden sm:block">
                  <div className="text-[10px] text-slate-400">Next level</div>
                  <div className="text-sm font-semibold text-slate-600">{nextLevelInfo.emoji} {nextLevelInfo.name}</div>
                </div>
              )}
            </div>
            {/* XP Progress bar */}
            <div className="h-3.5 bg-slate-100 rounded-full overflow-hidden relative">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${levelProgress}%` }}
                transition={{ delay: 0.5, duration: 1.5, type: 'spring', stiffness: 30, damping: 12 }}
                className={`h-full bg-gradient-to-r ${currentLevel.color} rounded-full relative`}
              >
                <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}
                  className="absolute right-0 top-1/2 -translate-y-1/2 h-5 w-5 rounded-full bg-white shadow-md border-2 border-current" style={{ borderColor: 'inherit' }} />
              </motion.div>
            </div>
            <div className="flex justify-between mt-1.5 text-[10px] text-slate-400">
              <span>{xp} XP</span>
              <span>{nextLevelInfo ? `${nextLevelInfo.minXp} XP to ${nextLevelInfo.name}` : '🏆 MAX LEVEL!'}</span>
            </div>
          </div>
        </motion.div>

        {/* Daily Quests */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.18 }} className="card">
          <div className="flex items-center gap-2 mb-3">
            <motion.div animate={{ rotate: [0, 10, -10, 0] }} transition={{ repeat: Infinity, duration: 2, ease: 'easeInOut' }}>
              <Flame className="h-5 w-5 text-orange-500" />
            </motion.div>
            <h3 className="text-slate-800 font-bold">Daily Quests</h3>
            <span className="text-[10px] text-slate-400 ml-auto">
              {dailyQuests.filter((q) => q.done).length}/{dailyQuests.length}
            </span>
          </div>
          <div className="space-y-2">
            {dailyQuests.map((q, i) => (
              <motion.div
                key={q.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.25 + i * 0.08 }}
                whileHover={q.action ? { x: 3, scale: 1.01 } : {}}
                onClick={q.action}
                className={`flex items-center gap-3 p-2.5 rounded-xl border transition-all duration-300 ${
                  q.done
                    ? 'bg-emerald-50 border-emerald-200'
                    : 'bg-white border-slate-100 hover:border-slate-200 hover:shadow-sm'
                } ${q.action ? 'cursor-pointer' : 'cursor-default'}`}
              >
                <motion.span className="text-lg" animate={q.done ? { scale: [1, 1.3, 1] } : {}}
                  transition={{ duration: 0.3 }}>{q.emoji}</motion.span>
                <div className="flex-1 min-w-0">
                  <div className={`text-xs font-medium ${q.done ? 'text-emerald-700 line-through' : 'text-slate-700'}`}>
                    {q.title}
                  </div>
                </div>
                <div className={`text-[10px] font-bold shrink-0 ${
                  q.done ? 'text-emerald-500' : 'text-amber-500'
                }`}>+{q.xpReward} XP</div>
                {q.done && <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }}
                  transition={{ type: 'spring' }} className="text-xs">✅</motion.span>}
              </motion.div>
            ))}
          </div>
          {dailyQuests.every((q) => q.done) && (
            <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
              className="mt-3 text-center text-xs font-bold text-emerald-600 bg-emerald-50 rounded-lg py-2">
              🎉 All quests complete! Amazing!
            </motion.div>
          )}
        </motion.div>
      </div>

      {/* ─── BMI GAUGE + STREAK + HEATMAP ROW ─────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }} className="card flex flex-col items-center justify-center py-6">
          <BmiGauge bmi={bmi} />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }} className="card flex flex-col items-center justify-center py-6">
          <AttendanceStreak attendance={attendance} />
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }} className="card sm:col-span-2 lg:col-span-1">
          <div className="flex items-center gap-2 mb-3">
            <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
              <Calendar className="h-3.5 w-3.5 text-white" />
            </div>
            <h3 className="text-slate-800 font-bold text-sm">Last 30 Days</h3>
          </div>
          <MiniHeatmap attendance={attendance} />
        </motion.div>
      </div>

      {/* ─── QUICK ACTIONS — Connected to tabs ─────────────── */}
      <div>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          className="flex items-center gap-2 mb-3">
          <Zap className="h-4 w-4 text-violet-500" />
          <h2 className="text-slate-800 text-lg font-bold">Quick Actions</h2>
        </motion.div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          <QuickAction icon={Calendar} label="My Attendance" desc="Check-in history"
            gradient="from-indigo-500 to-violet-500" onClick={() => onTabChange?.('attendance')} delay={0.32} />
          <QuickAction icon={Dumbbell} label="Classes" desc="Browse & enroll"
            gradient="from-emerald-500 to-teal-500" onClick={() => onTabChange?.('classes')} delay={0.36} />
          <QuickAction icon={Target} label="Workouts" desc="Your workout plans"
            gradient="from-amber-500 to-orange-500" onClick={() => onTabChange?.('workouts')} delay={0.4} />
          <QuickAction icon={FileText} label="Invoices" desc="Billing & payments"
            gradient="from-sky-500 to-blue-500" onClick={() => onTabChange?.('invoices')} delay={0.44} />
          <QuickAction icon={TrendingUp} label="Progress" desc="Track your gains"
            gradient="from-rose-500 to-pink-500" onClick={() => onTabChange?.('progress')} delay={0.48} />
          <QuickAction icon={LifeBuoy} label="Support" desc="Get help"
            gradient="from-violet-500 to-purple-500" onClick={() => onTabChange?.('tickets')} delay={0.52} />
        </div>
      </div>

      {/* ─── ACHIEVEMENTS ──────────────────────────────────── */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
        className="card">
        <div className="flex items-center gap-2 mb-4">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center">
            <Award className="h-4 w-4 text-white" />
          </div>
          <h2 className="text-slate-800 font-bold">Achievements</h2>
          <span className="text-xs text-slate-400 ml-auto">
            {achievements.filter((a) => attendance.length >= a.threshold).length}/{achievements.length} unlocked
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {achievements.map((a, i) => (
            <AchievementBadge key={a.label} {...a}
              unlocked={attendance.length >= a.threshold} delay={0.45 + i * 0.06} />
          ))}
        </div>
      </motion.div>

      {/* ─── PROFILE + QUOTE ───────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.45 }} className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-slate-800 text-xl font-bold">Profile</h2>
            <div className="flex gap-2">
              {!editing ? (
                <>
                  <button onClick={() => setEditing(true)} className="btn-ghost text-sm">
                    <Edit2 className="h-3.5 w-3.5" /> Edit
                  </button>
                  <button onClick={() => setShowPwModal(true)} className="btn-ghost text-sm">
                    <Lock className="h-3.5 w-3.5" /> Password
                  </button>
                </>
              ) : (
                <>
                  <button onClick={saveProfile} disabled={busy} className="btn-primary text-sm">
                    <Save className="h-3.5 w-3.5" /> {busy ? 'Saving…' : 'Save'}
                  </button>
                  <button onClick={() => setEditing(false)} className="btn-ghost text-sm">
                    <X className="h-3.5 w-3.5" /> Cancel
                  </button>
                </>
              )}
            </div>
          </div>
          {!me ? (
            <div className="text-slate-500">Loading…</div>
          ) : editing ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <label className="label">Name</label>
                <input className="input" value={editForm.name} onChange={(e) => setEditForm((s) => ({ ...s, name: e.target.value }))} />
              </div>
              <div>
                <label className="label">Phone</label>
                <input className="input" value={editForm.phone} onChange={(e) => setEditForm((s) => ({ ...s, phone: e.target.value }))} />
              </div>
              <div>
                <label className="label">Height (cm)</label>
                <input type="number" className="input" value={editForm.heightCm} onChange={(e) => setEditForm((s) => ({ ...s, heightCm: e.target.value }))} />
              </div>
              <div>
                <label className="label">Weight (kg)</label>
                <input type="number" className="input" value={editForm.weightKg} onChange={(e) => setEditForm((s) => ({ ...s, weightKg: e.target.value }))} />
              </div>
              <div className="md:col-span-2">
                <label className="label">Fitness Goals</label>
                <textarea className="input" rows={2} value={editForm.fitnessGoals} onChange={(e) => setEditForm((s) => ({ ...s, fitnessGoals: e.target.value }))} />
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              {[
                { label: 'Name', val: me.name },
                { label: 'Email', val: me.email },
                { label: 'Goal', val: me.fitness_goals || me.fitnessGoals || '—' },
                { label: 'Status', val: me.status },
                { label: 'Phone', val: me.phone || '—' },
                { label: 'Membership plan', val: me.plan_name || '—' },
              ].map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  whileHover={{ scale: 1.01, borderColor: 'rgb(165,180,252)' }}
                  className="rounded-xl border border-slate-200 bg-white p-4 hover:shadow-md transition-all duration-200"
                >
                  <div className="text-slate-500 text-xs">{item.label}</div>
                  <div className="text-slate-900 font-semibold mt-0.5">{item.val}</div>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>

        {/* Motivational quote */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="relative overflow-hidden rounded-2xl border border-slate-200 bg-gradient-to-br from-violet-50 via-white to-fuchsia-50 p-5 flex flex-col">
          <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br from-violet-100 to-fuchsia-100 blur-xl" />
          <div className="relative flex-1 flex flex-col">
            <div className="flex items-center gap-2 mb-4">
              <motion.div animate={{ rotate: [0, 360] }}
                transition={{ repeat: Infinity, duration: 20, ease: 'linear' }}
                className="h-9 w-9 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shrink-0">
                <Sparkles className="h-4 w-4 text-white" />
              </motion.div>
              <h3 className="text-slate-800 font-bold text-sm">Daily Motivation</h3>
            </div>
            <div className="flex-1 flex flex-col justify-center">
              <AnimatePresence mode="wait">
                <motion.div key={quoteIdx}
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.4 }}>
                  <div className="text-slate-700 font-medium text-sm leading-relaxed italic">
                    "{motivationalQuotes[quoteIdx].text}"
                  </div>
                  <div className="text-xs text-violet-500 font-semibold mt-3">
                    — {motivationalQuotes[quoteIdx].author}
                  </div>
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      </div>

      {/* ─── Password Change Modal ─────────────────────────── */}
      <AnimatePresence>
        {showPwModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              transition={{ type: 'spring', stiffness: 300, damping: 25 }}
              className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-md border border-slate-200"
            >
              <h3 className="text-slate-900 text-lg font-bold mb-4">Change Password</h3>
              <form onSubmit={changePassword} className="space-y-4">
                <div>
                  <label className="label">Current Password</label>
                  <input type="password" className="input"
                    value={pwForm.currentPassword}
                    onChange={(e) => setPwForm((s) => ({ ...s, currentPassword: e.target.value }))} required />
                </div>
                <div>
                  <label className="label">New Password</label>
                  <input type="password" className="input"
                    value={pwForm.newPassword}
                    onChange={(e) => setPwForm((s) => ({ ...s, newPassword: e.target.value }))} required minLength={6} />
                </div>
                <div className="flex justify-end gap-2">
                  <button type="button" onClick={() => setShowPwModal(false)} className="btn-ghost">Cancel</button>
                  <button type="submit" disabled={busy} className="btn-primary">{busy ? 'Changing…' : 'Change Password'}</button>
                </div>
              </form>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </PageShell>
  );
}
