import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Dumbbell,
  ArrowRight,
  ArrowLeft,
  Eye,
  EyeOff,
  Zap,
  Shield,
  Users,
  BarChart3,
  Lock,
  Mail,
  User,
  Sparkles,
  Activity,
  Heart,
  Timer,
  Trophy,
  Flame,
  CheckCircle2,
} from 'lucide-react';
import api from '../services/api';

/* ─── Particle canvas: floating dots with connection lines ── */
function ParticleField() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;
    let particles = [];
    const COLORS = [
      'rgba(99,102,241,0.4)',
      'rgba(14,165,233,0.35)',
      'rgba(16,185,129,0.35)',
      'rgba(244,63,94,0.3)',
      'rgba(168,85,247,0.3)',
    ];

    function resize() {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    function init() {
      resize();
      particles = Array.from({ length: 50 }, () => ({
        x: Math.random() * canvas.offsetWidth,
        y: Math.random() * canvas.offsetHeight,
        r: Math.random() * 3 + 1,
        dx: (Math.random() - 0.5) * 0.5,
        dy: (Math.random() - 0.5) * 0.5,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
      for (const p of particles) {
        p.x += p.dx;
        p.y += p.dy;
        if (p.x < 0 || p.x > canvas.offsetWidth) p.dx *= -1;
        if (p.y < 0 || p.y > canvas.offsetHeight) p.dy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
      }
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(99,102,241,${0.08 * (1 - dist / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      raf = requestAnimationFrame(draw);
    }

    init();
    draw();
    window.addEventListener('resize', resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" style={{ opacity: 0.7 }} />
  );
}

/* ─── Glowing cursor follower (smooth RAF-based) ─────────── */
function GlowCursor() {
  const ref = useRef(null);
  const mouse = useRef({ x: -400, y: -400 });
  const current = useRef({ x: -400, y: -400 });

  useEffect(() => {
    const handler = (e) => {
      mouse.current.x = e.clientX;
      mouse.current.y = e.clientY;
    };
    window.addEventListener('mousemove', handler, { passive: true });
    return () => window.removeEventListener('mousemove', handler);
  }, []);

  useEffect(() => {
    let raf;
    const lerp = (a, b, t) => a + (b - a) * t;
    const animate = () => {
      current.current.x = lerp(current.current.x, mouse.current.x, 0.15);
      current.current.y = lerp(current.current.y, mouse.current.y, 0.15);
      if (ref.current) {
        ref.current.style.transform = `translate3d(${current.current.x - 200}px, ${current.current.y - 200}px, 0)`;
      }
      raf = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div
      ref={ref}
      className="fixed top-0 left-0 pointer-events-none z-0 hidden lg:block"
      style={{
        width: 400,
        height: 400,
        background: 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.06) 40%, transparent 70%)',
        borderRadius: '50%',
        willChange: 'transform',
      }}
    />
  );
}

/* ─── Password strength meter ────────────────────────────── */
function PasswordStrength({ password }) {
  const score = (() => {
    let s = 0;
    if (password.length >= 6) s++;
    if (password.length >= 10) s++;
    if (/[A-Z]/.test(password)) s++;
    if (/[0-9]/.test(password)) s++;
    if (/[^A-Za-z0-9]/.test(password)) s++;
    return s;
  })();
  const label = ['', 'Weak', 'Fair', 'Good', 'Strong', 'Excellent'][score] || '';
  const color = ['bg-slate-200', 'bg-red-400', 'bg-orange-400', 'bg-yellow-400', 'bg-emerald-400', 'bg-emerald-500'][
    score
  ];

  if (!password) return null;
  return (
    <div className="mt-2">
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full transition-all duration-300 ${i <= score ? color : 'bg-white/10'}`}
          />
        ))}
      </div>
      <div className="text-xs mt-1 text-white/40">{label}</div>
    </div>
  );
}

/* ─── Animated feature cards data ────────────────────────── */
const features = [
  {
    icon: Zap,
    title: 'Smart Billing',
    desc: 'Auto invoices & payments tracking',
    color: 'from-amber-500 to-orange-600',
  },
  {
    icon: BarChart3,
    title: 'Live Analytics',
    desc: 'Real-time performance insights',
    color: 'from-indigo-500 to-purple-600',
  },
  { icon: Users, title: 'Member Hub', desc: 'Complete member management', color: 'from-sky-500 to-cyan-600' },
  {
    icon: Shield,
    title: 'Role-Based Access',
    desc: 'Admin, trainer & member portals',
    color: 'from-emerald-500 to-teal-600',
  },
];

const floatingIcons = [Activity, Heart, Timer, Trophy, Flame, Dumbbell, Zap, Sparkles];

/* ─── Input component for dark auth form ─────────────────── */
function AuthInput({ icon: Icon, iconChar, ...props }) {
  return (
    <div className="relative group">
      {Icon && (
        <Icon className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-white/30 group-focus-within:text-indigo-400 transition-colors" />
      )}
      {iconChar && (
        <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30 text-sm group-focus-within:text-indigo-400 transition-colors font-medium">
          {iconChar}
        </span>
      )}
      <input
        {...props}
        className={`w-full bg-white/[0.06] border border-white/[0.1] rounded-xl py-3 text-white placeholder:text-white/20 focus:outline-none focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/20 transition-all ${
          Icon || iconChar ? 'pl-11' : 'pl-4'
        } ${props.className || 'pr-4'}`}
      />
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────── */
export default function LoginPage({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [regForm, setRegForm] = useState({ name: '', username: '', email: '', password: '' });
  const [showRegPwd, setShowRegPwd] = useState(false);

  const [activeFeature, setActiveFeature] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setActiveFeature((p) => (p + 1) % features.length), 3000);
    return () => clearInterval(timer);
  }, []);

  const switchMode = useCallback((newMode) => {
    setError('');
    setSuccess('');
    setMode(newMode);
  }, []);

  const submitLogin = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      const isEmail = identifier.includes('@');
      const payload = isEmail ? { email: identifier, password } : { username: identifier, password };
      const res = await api.post('/api/auth/login', payload);
      setSuccess('Welcome back! Redirecting…');
      setTimeout(() => onLogin(res.data), 600);
    } catch (err) {
      setError(err?.response?.data?.error || 'Login failed. Please check your credentials.');
    } finally {
      setBusy(false);
    }
  };

  const submitRegister = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      const res = await api.post('/api/auth/register', regForm);
      setSuccess('Account created! Logging you in…');
      setTimeout(() => onLogin(res.data), 600);
    } catch (err) {
      setError(err?.response?.data?.error || err?.response?.data?.errors?.[0]?.message || 'Registration failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden flex items-center justify-center">
      {/* Dark animated gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950" />
      <div className="absolute inset-0 opacity-30">
        <div
          className="absolute w-[600px] h-[600px] sm:w-[800px] sm:h-[800px] rounded-full top-[-300px] left-[-200px]"
          style={{
            background: 'radial-gradient(circle, rgba(99,102,241,0.4) 0%, transparent 70%)',
            animation: 'auth-pulse-glow 8s ease-in-out infinite',
          }}
        />
        <div
          className="absolute w-[400px] h-[400px] sm:w-[600px] sm:h-[600px] rounded-full bottom-[-200px] right-[-100px]"
          style={{
            background: 'radial-gradient(circle, rgba(14,165,233,0.35) 0%, transparent 70%)',
            animation: 'auth-pulse-glow 8s ease-in-out infinite 2s',
          }}
        />
        <div
          className="absolute w-[300px] h-[300px] sm:w-[500px] sm:h-[500px] rounded-full top-[40%] left-[60%]"
          style={{
            background: 'radial-gradient(circle, rgba(168,85,247,0.3) 0%, transparent 70%)',
            animation: 'auth-pulse-glow 8s ease-in-out infinite 4s',
          }}
        />
      </div>

      <ParticleField />
      <GlowCursor />

      {/* Floating icons (desktop only) */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden hidden lg:block">
        {floatingIcons.map((Icon, i) => (
          <motion.div
            key={i}
            className="absolute text-white/[0.06]"
            animate={{
              y: [0, -20, 0, 20, 0],
              x: [0, 10, 0, -10, 0],
              rotate: [0, 5, 0, -5, 0],
            }}
            transition={{
              duration: 8 + i * 2,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: i * 0.5,
            }}
            style={{
              left: `${10 + ((i * 12) % 80)}%`,
              top: `${15 + ((i * 17) % 60)}%`,
            }}
          >
            <Icon size={40 + (i % 3) * 16} strokeWidth={1} />
          </motion.div>
        ))}
      </div>

      {/* Main content */}
      <div className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-0 items-center">
          {/* ─── Left Panel: Branding ────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
            className="hidden lg:flex flex-col justify-center pr-16"
          >
            <div className="flex items-center gap-4 mb-10">
              <motion.div
                className="h-14 w-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30"
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <Dumbbell className="h-7 w-7 text-white" />
              </motion.div>
              <div>
                <h1 className="text-3xl font-black text-white tracking-tight">FitFlex</h1>
                <p className="text-white/50 text-sm font-medium">Performance Studio</p>
              </div>
            </div>

            <motion.h2
              className="text-4xl xl:text-5xl font-black text-white leading-[1.15] mb-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              Elevate every{' '}
              <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                visit
              </span>{' '}
              into{' '}
              <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                progress
              </span>
            </motion.h2>

            <motion.p
              className="text-white/50 text-lg mb-10 max-w-md leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35, duration: 0.6 }}
            >
              The complete gym management platform — membership, billing, training, and analytics unified.
            </motion.p>

            <div className="space-y-3">
              {features.map((f, i) => {
                const Icon = f.icon;
                const isActive = i === activeFeature;
                return (
                  <motion.div
                    key={f.title}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + i * 0.1, duration: 0.5 }}
                    onMouseEnter={() => setActiveFeature(i)}
                    className={`flex items-center gap-4 p-4 rounded-2xl cursor-pointer transition-all duration-500 ${
                      isActive
                        ? 'bg-white/[0.08] border border-white/[0.12] shadow-lg shadow-black/10'
                        : 'bg-transparent border border-transparent hover:bg-white/[0.04]'
                    }`}
                  >
                    <div
                      className={`h-11 w-11 rounded-xl bg-gradient-to-br ${f.color} flex items-center justify-center shrink-0 transition-transform duration-500 ${
                        isActive ? 'scale-110 shadow-lg' : ''
                      }`}
                    >
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-white font-semibold text-sm">{f.title}</div>
                      <div
                        className={`text-sm transition-colors duration-300 ${isActive ? 'text-white/60' : 'text-white/30'}`}
                      >
                        {f.desc}
                      </div>
                    </div>
                    {isActive && (
                      <motion.div
                        layoutId="feature-dot"
                        className="h-2 w-2 rounded-full bg-indigo-400 shrink-0"
                        transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                      />
                    )}
                  </motion.div>
                );
              })}
            </div>

            <motion.div
              className="mt-10 flex items-center gap-6 text-white/30 text-xs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.6 }}
            >
              <span className="flex items-center gap-1.5">
                <Shield className="h-3.5 w-3.5" /> Secure
              </span>
              <span className="flex items-center gap-1.5">
                <Zap className="h-3.5 w-3.5" /> Fast
              </span>
              <span className="flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" /> Real-time
              </span>
            </motion.div>
          </motion.div>

          {/* ─── Right Panel: Auth Card ──────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.1, ease: 'easeOut' }}
            className="flex justify-center lg:justify-end"
          >
            <div className="w-full max-w-md">
              {/* Mobile logo */}
              <div className="lg:hidden flex items-center justify-center gap-3 mb-8">
                <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
                  <Dumbbell className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-black text-white tracking-tight">FitFlex</h1>
                  <p className="text-white/50 text-xs font-medium">Performance Studio</p>
                </div>
              </div>

              {/* Card with glow */}
              <div className="relative">
                <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/20 via-purple-500/20 to-pink-500/20 rounded-3xl blur-xl" />

                <div className="relative bg-white/[0.07] backdrop-blur-xl border border-white/[0.12] rounded-3xl p-5 sm:p-8 shadow-2xl shadow-black/20">
                  <AnimatePresence mode="wait">
                    {mode === 'login' ? (
                      <motion.div
                        key="login"
                        initial={{ opacity: 0, x: 30 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -30 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="mb-6">
                          <h2 className="text-2xl font-bold text-white">Welcome back</h2>
                          <p className="text-white/40 text-sm mt-1">Sign in to your portal</p>
                        </div>

                        <form onSubmit={submitLogin} className="space-y-4">
                          <div>
                            <label className="text-white/60 text-sm font-medium mb-1.5 block">Email or Username</label>
                            <AuthInput
                              icon={Mail}
                              value={identifier}
                              onChange={(e) => setIdentifier(e.target.value)}
                              placeholder="admin@gym.local"
                              autoComplete="username"
                              required
                            />
                          </div>

                          <div>
                            <label className="text-white/60 text-sm font-medium mb-1.5 block">Password</label>
                            <div className="relative group">
                              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-white/30 group-focus-within:text-indigo-400 transition-colors" />
                              <input
                                type={showPwd ? 'text' : 'password'}
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full bg-white/[0.06] border border-white/[0.1] rounded-xl pl-11 pr-12 py-3 text-white placeholder:text-white/20 focus:outline-none focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                                autoComplete="current-password"
                                required
                              />
                              <button
                                type="button"
                                onClick={() => setShowPwd(!showPwd)}
                                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
                              >
                                {showPwd ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                              </button>
                            </div>
                          </div>

                          <AnimatePresence>
                            {error && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-red-300 text-sm overflow-hidden"
                              >
                                {error}
                              </motion.div>
                            )}
                            {success && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3 text-emerald-300 text-sm flex items-center gap-2 overflow-hidden"
                              >
                                <CheckCircle2 className="h-4 w-4 shrink-0" /> {success}
                              </motion.div>
                            )}
                          </AnimatePresence>

                          <motion.button
                            type="submit"
                            disabled={busy}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="w-full py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/25 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-60"
                          >
                            {busy ? (
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                                className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full"
                              />
                            ) : (
                              <>
                                Sign in <ArrowRight className="h-4 w-4" />
                              </>
                            )}
                          </motion.button>
                        </form>

                        <div className="mt-6 text-center">
                          <button
                            onClick={() => switchMode('register')}
                            className="text-white/40 hover:text-white/70 text-sm transition-colors"
                          >
                            Don&apos;t have an account?{' '}
                            <span className="text-indigo-400 font-medium hover:text-indigo-300">Create one</span>
                          </button>
                        </div>

                        {/* Demo quick-access */}
                        <div className="mt-6 pt-6 border-t border-white/[0.08]">
                          <p className="text-white/30 text-xs text-center mb-3">Quick demo access</p>
                          <div className="grid grid-cols-3 gap-2">
                            {[
                              {
                                label: 'Admin',
                                user: 'admin',
                                color:
                                  'from-violet-500/20 to-purple-500/20 border-violet-500/20 text-violet-300 hover:border-violet-400/40',
                              },
                              {
                                label: 'Trainer',
                                user: 'trainer',
                                color:
                                  'from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-300 hover:border-amber-400/40',
                              },
                              {
                                label: 'Member',
                                user: 'member',
                                color:
                                  'from-sky-500/20 to-cyan-500/20 border-sky-500/20 text-sky-300 hover:border-sky-400/40',
                              },
                            ].map((demo) => (
                              <motion.button
                                key={demo.user}
                                type="button"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => {
                                  setIdentifier(demo.user);
                                  setPassword(demo.user + '123');
                                }}
                                className={`bg-gradient-to-br ${demo.color} border rounded-xl px-3 py-2.5 text-xs font-medium text-center transition-all`}
                              >
                                {demo.label}
                              </motion.button>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="register"
                        initial={{ opacity: 0, x: 30 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -30 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="flex items-center gap-3 mb-6">
                          <motion.button
                            type="button"
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.9 }}
                            onClick={() => switchMode('login')}
                            className="h-10 w-10 rounded-xl bg-white/[0.08] border border-white/[0.1] text-white/60 flex items-center justify-center hover:bg-white/[0.12] hover:text-white transition-all shrink-0"
                          >
                            <ArrowLeft className="h-5 w-5" />
                          </motion.button>
                          <div>
                            <h2 className="text-2xl font-bold text-white">Create account</h2>
                            <p className="text-white/40 text-sm">Join FitFlex today</p>
                          </div>
                        </div>

                        <form onSubmit={submitRegister} className="space-y-4">
                          <div>
                            <label className="text-white/60 text-sm font-medium mb-1.5 block">Full Name</label>
                            <AuthInput
                              icon={User}
                              value={regForm.name}
                              onChange={(e) => setRegForm((s) => ({ ...s, name: e.target.value }))}
                              placeholder="Jane Smith"
                              required
                            />
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <div>
                              <label className="text-white/60 text-sm font-medium mb-1.5 block">Username</label>
                              <AuthInput
                                iconChar="@"
                                value={regForm.username}
                                onChange={(e) => setRegForm((s) => ({ ...s, username: e.target.value }))}
                                placeholder="janesmith"
                                required
                              />
                            </div>
                            <div>
                              <label className="text-white/60 text-sm font-medium mb-1.5 block">Email</label>
                              <AuthInput
                                icon={Mail}
                                value={regForm.email}
                                onChange={(e) => setRegForm((s) => ({ ...s, email: e.target.value }))}
                                placeholder="jane@email.com"
                                type="email"
                                required
                              />
                            </div>
                          </div>

                          <div>
                            <label className="text-white/60 text-sm font-medium mb-1.5 block">Password</label>
                            <div className="relative group">
                              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-white/30 group-focus-within:text-indigo-400 transition-colors" />
                              <input
                                type={showRegPwd ? 'text' : 'password'}
                                value={regForm.password}
                                onChange={(e) => setRegForm((s) => ({ ...s, password: e.target.value }))}
                                className="w-full bg-white/[0.06] border border-white/[0.1] rounded-xl pl-11 pr-12 py-3 text-white placeholder:text-white/20 focus:outline-none focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                                placeholder="Min. 6 characters"
                                autoComplete="new-password"
                                required
                              />
                              <button
                                type="button"
                                onClick={() => setShowRegPwd(!showRegPwd)}
                                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
                              >
                                {showRegPwd ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                              </button>
                            </div>
                            <PasswordStrength password={regForm.password} />
                          </div>

                          <AnimatePresence>
                            {error && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-red-300 text-sm overflow-hidden"
                              >
                                {error}
                              </motion.div>
                            )}
                            {success && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3 text-emerald-300 text-sm flex items-center gap-2 overflow-hidden"
                              >
                                <CheckCircle2 className="h-4 w-4 shrink-0" /> {success}
                              </motion.div>
                            )}
                          </AnimatePresence>

                          <motion.button
                            type="submit"
                            disabled={busy}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="w-full py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/25 transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-60"
                          >
                            {busy ? (
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                                className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full"
                              />
                            ) : (
                              <>
                                Create account <Sparkles className="h-4 w-4" />
                              </>
                            )}
                          </motion.button>
                        </form>

                        <div className="mt-6 text-center">
                          <button
                            onClick={() => switchMode('login')}
                            className="text-white/40 hover:text-white/70 text-sm transition-colors"
                          >
                            Already have an account?{' '}
                            <span className="text-indigo-400 font-medium hover:text-indigo-300">Sign in</span>
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      <style>{`
        @keyframes auth-pulse-glow {
          0%, 100% { transform: scale(1); opacity: 0.3; }
          50% { transform: scale(1.15); opacity: 0.6; }
        }
      `}</style>
    </div>
  );
}
