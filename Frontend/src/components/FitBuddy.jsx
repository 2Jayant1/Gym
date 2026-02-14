import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/* ─── Tab Reactions Config ─────────────────────────────────── */
const tabReactions = {
  dashboard:  { msg: "Let's crush it today! 💪", eyes: 'normal' },
  attendance: { msg: "Consistency is king! Keep showing up! 🔥", eyes: 'star' },
  classes:    { msg: "Time to sweat! Ready? 🏋️", eyes: 'normal' },
  workouts:   { msg: "Gains incoming! Let's go! 💪", eyes: 'star' },
  invoices:   { msg: "Ka-ching! Money time! 💰", eyes: 'money' },
  progress:   { msg: "Look at those gains! 📈", eyes: 'star' },
  tickets:    { msg: "I'm here to help! 🤗", eyes: 'heart' },
  sessions:   { msg: "Coach mode: ON! 🎯", eyes: 'normal' },
  members:    { msg: "Let's check on the team! 👥", eyes: 'normal' },
  billing:    { msg: "Show me the money! 💵", eyes: 'money' },
  support:    { msg: "Let's solve this together! 🛟", eyes: 'heart' },
  analytics:  { msg: "Data nerd mode! 📊", eyes: 'star' },
  db:         { msg: "Ooh, the matrix! 🖥️", eyes: 'star' },
  plans:      { msg: "Planning for success! 🎯", eyes: 'normal' },
};

const roleGreetings = {
  admin:   { greeting: "Hey Boss!", sub: "I'll watch the numbers for you!" },
  trainer: { greeting: "Hey Coach!", sub: "Let's fire up those sessions!" },
  member:  { greeting: "Hey Champ!", sub: "I'll be your fitness buddy!" },
};

/* ═══════════════════════════════════════════════════════════════
   FIT BUDDY — Animated Mascot
   ═══════════════════════════════════════════════════════════════ */
export default function FitBuddy({ activeTab, role }) {
  const hasSeenIntro = useRef(sessionStorage.getItem('fitbuddy_intro'));
  const [phase, setPhase] = useState(hasSeenIntro.current ? 'idle' : 'intro');
  const [eyeMode, setEyeMode] = useState('normal');
  const [bubbleMsg, setBubbleMsg] = useState('');
  const [showBubble, setShowBubble] = useState(false);
  const [blinking, setBlink] = useState(false);
  const [clicked, setClicked] = useState(false);

  const buddyRef = useRef(null);
  const leftPupilRef = useRef(null);
  const rightPupilRef = useRef(null);
  const leftShineRef = useRef(null);
  const rightShineRef = useRef(null);
  const mouseRef = useRef({ x: typeof window !== 'undefined' ? window.innerWidth / 2 : 400, y: typeof window !== 'undefined' ? window.innerHeight / 2 : 400 });
  const pupilCurrent = useRef({ x: 0, y: 0 });
  const rafRef = useRef(null);
  const prevTabRef = useRef(activeTab);
  const bubbleTimerRef = useRef(null);
  const eyeTimerRef = useRef(null);

  /* ─── Global Mouse Tracking ──────────────────────────────── */
  useEffect(() => {
    const handle = (e) => { mouseRef.current = { x: e.clientX, y: e.clientY }; };
    window.addEventListener('mousemove', handle, { passive: true });
    return () => window.removeEventListener('mousemove', handle);
  }, []);

  /* ─── RAF Eye Pupil Update ───────────────────────────────── */
  useEffect(() => {
    const update = () => {
      if (phase === 'idle' && eyeMode === 'normal' && buddyRef.current) {
        const rect = buddyRef.current.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height * 0.38;
        const dx = mouseRef.current.x - cx;
        const dy = mouseRef.current.y - cy;
        const angle = Math.atan2(dy, dx);
        const maxD = 4.5;
        const dist = Math.min(maxD, Math.hypot(dx, dy) * 0.004);
        const tx = Math.cos(angle) * dist;
        const ty = Math.sin(angle) * dist;
        pupilCurrent.current.x += (tx - pupilCurrent.current.x) * 0.12;
        pupilCurrent.current.y += (ty - pupilCurrent.current.y) * 0.12;
        const px = pupilCurrent.current.x;
        const py = pupilCurrent.current.y;
        leftPupilRef.current?.setAttribute('cx', String(35 + px));
        leftPupilRef.current?.setAttribute('cy', String(38 + py));
        rightPupilRef.current?.setAttribute('cx', String(65 + px));
        rightPupilRef.current?.setAttribute('cy', String(38 + py));
        leftShineRef.current?.setAttribute('cx', String(32.5 + px));
        leftShineRef.current?.setAttribute('cy', String(35 + py));
        rightShineRef.current?.setAttribute('cx', String(62.5 + px));
        rightShineRef.current?.setAttribute('cy', String(35 + py));
      }
      rafRef.current = requestAnimationFrame(update);
    };
    rafRef.current = requestAnimationFrame(update);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [phase, eyeMode]);

  /* ─── Blink Timer ────────────────────────────────────────── */
  useEffect(() => {
    let id;
    const schedule = () => {
      id = setTimeout(() => {
        setBlink(true);
        setTimeout(() => { setBlink(false); schedule(); }, 150);
      }, 2500 + Math.random() * 3500);
    };
    schedule();
    return () => clearTimeout(id);
  }, []);

  /* ─── Intro Timer ────────────────────────────────────────── */
  useEffect(() => {
    if (phase !== 'intro') return;
    const t = setTimeout(() => {
      setPhase('idle');
      sessionStorage.setItem('fitbuddy_intro', '1');
    }, 3800);
    return () => clearTimeout(t);
  }, [phase]);

  /* ─── Tab-change Reactions ───────────────────────────────── */
  useEffect(() => {
    if (phase !== 'idle') return;
    if (activeTab === prevTabRef.current) return;
    prevTabRef.current = activeTab;
    const reaction = tabReactions[activeTab];
    if (!reaction) return;
    setEyeMode(reaction.eyes);
    setBubbleMsg(reaction.msg);
    setShowBubble(true);
    clearTimeout(bubbleTimerRef.current);
    clearTimeout(eyeTimerRef.current);
    bubbleTimerRef.current = setTimeout(() => setShowBubble(false), 4000);
    eyeTimerRef.current = setTimeout(() => setEyeMode('normal'), 5000);
    return () => { clearTimeout(bubbleTimerRef.current); clearTimeout(eyeTimerRef.current); };
  }, [activeTab, phase]);

  /* ─── Click Handler ──────────────────────────────────────── */
  const handleClick = useCallback(() => {
    setClicked(true);
    const reaction = tabReactions[activeTab] || tabReactions.dashboard;
    setEyeMode(reaction.eyes);
    setBubbleMsg(reaction.msg);
    setShowBubble(true);
    clearTimeout(bubbleTimerRef.current);
    clearTimeout(eyeTimerRef.current);
    bubbleTimerRef.current = setTimeout(() => { setShowBubble(false); setClicked(false); }, 3000);
    eyeTimerRef.current = setTimeout(() => setEyeMode('normal'), 4000);
  }, [activeTab]);

  /* ─── Dismiss Intro ──────────────────────────────────────── */
  const dismissIntro = useCallback(() => {
    setPhase('idle');
    sessionStorage.setItem('fitbuddy_intro', '1');
  }, []);

  const { greeting, sub } = roleGreetings[role] || roleGreetings.member;

  /* ─── Face SVG (shared renderer) ─────────────────────────── */
  const renderFace = (size, withTracking) => (
    <div
      className="rounded-full bg-gradient-to-br from-indigo-500 via-violet-500 to-purple-600 shadow-2xl relative overflow-hidden"
      style={{ width: size, height: size }}
    >
      {/* Body shine */}
      <div className="absolute top-[8%] left-[12%] w-[35%] h-[22%] rounded-full bg-white/20 blur-[3px]" />
      <svg viewBox="0 0 100 100" className="w-full h-full relative">
        {/* Headband */}
        <path d="M 15 23 Q 50 13 85 23" stroke="#f97316" strokeWidth="5" fill="none" strokeLinecap="round" />
        <text x="50" y="22" textAnchor="middle" fill="white" fontSize="7" fontWeight="bold">★</text>

        {/* Eye whites */}
        <motion.ellipse cx="35" cy="38" rx="12" fill="white"
          animate={{ ry: blinking ? 1 : 13 }} transition={{ duration: 0.08 }} />
        <motion.ellipse cx="65" cy="38" rx="12" fill="white"
          animate={{ ry: blinking ? 1 : 13 }} transition={{ duration: 0.08 }} />

        {/* Pupils */}
        {!blinking && (eyeMode === 'money' ? (
          <>
            <text x="35" y="44" textAnchor="middle" fill="#16a34a" fontSize="18" fontWeight="bold" style={{ fontFamily: 'system-ui' }}>$</text>
            <text x="65" y="44" textAnchor="middle" fill="#16a34a" fontSize="18" fontWeight="bold" style={{ fontFamily: 'system-ui' }}>$</text>
          </>
        ) : eyeMode === 'star' ? (
          <>
            <text x="35" y="44" textAnchor="middle" fill="#f59e0b" fontSize="15">★</text>
            <text x="65" y="44" textAnchor="middle" fill="#f59e0b" fontSize="15">★</text>
          </>
        ) : eyeMode === 'heart' ? (
          <>
            <text x="35" y="44" textAnchor="middle" fill="#ef4444" fontSize="14">♥</text>
            <text x="65" y="44" textAnchor="middle" fill="#ef4444" fontSize="14">♥</text>
          </>
        ) : (
          <>
            <circle ref={withTracking ? leftPupilRef : undefined} cx="35" cy="38" r="5.5" fill="#1e293b" />
            <circle ref={withTracking ? rightPupilRef : undefined} cx="65" cy="38" r="5.5" fill="#1e293b" />
            <circle ref={withTracking ? leftShineRef : undefined} cx="32.5" cy="35" r="2" fill="white" opacity="0.8" />
            <circle ref={withTracking ? rightShineRef : undefined} cx="62.5" cy="35" r="2" fill="white" opacity="0.8" />
          </>
        ))}

        {/* Mouth */}
        {eyeMode === 'money' ? (
          <path d="M 33 63 Q 50 78 67 63" stroke="white" strokeWidth="3" fill="none" strokeLinecap="round" />
        ) : eyeMode === 'heart' ? (
          <path d="M 40 62 Q 50 68 60 62" stroke="white" strokeWidth="2.5" fill="none" strokeLinecap="round" />
        ) : (
          <path d="M 38 62 Q 50 72 62 62" stroke="white" strokeWidth="2.5" fill="none" strokeLinecap="round" />
        )}

        {/* Blush */}
        <circle cx="20" cy="50" r="5" fill="#f472b6" opacity="0.25" />
        <circle cx="80" cy="50" r="5" fill="#f472b6" opacity="0.25" />
      </svg>
    </div>
  );

  return (
    <>
      {/* ─── INTRO OVERLAY ──────────────────────────────── */}
      <AnimatePresence>
        {phase === 'intro' && (
          <motion.div
            key="intro"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            onClick={dismissIntro}
            className="fixed inset-0 z-[200] flex items-center justify-center bg-black/30 backdrop-blur-sm cursor-pointer"
          >
            <motion.div
              initial={{ scale: 0, y: 80, rotate: -15 }}
              animate={{ scale: 1, y: 0, rotate: 0 }}
              exit={{ scale: 0.3, y: 60, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 180, damping: 14 }}
              className="flex flex-col items-center gap-5"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Floating mascot */}
              <motion.div
                animate={{ y: [0, -10, 0], rotate: [0, 3, -3, 0] }}
                transition={{ repeat: Infinity, duration: 2.5, ease: 'easeInOut' }}
              >
                {renderFace(120, false)}
              </motion.div>

              {/* Speech bubble */}
              <motion.div
                initial={{ opacity: 0, y: 15, scale: 0.8 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ delay: 0.5, type: 'spring', stiffness: 200 }}
                className="bg-white rounded-2xl px-7 py-4 shadow-xl border border-slate-200 relative max-w-xs"
              >
                <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-white border-l border-t border-slate-200 rotate-45" />
                <div className="text-slate-800 font-bold text-lg text-center">
                  {greeting} I'm FlexBot! 💪
                </div>
                <div className="text-slate-500 text-sm text-center mt-1">{sub}</div>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1.5 }}
                  className="text-[10px] text-slate-400 text-center mt-3"
                >
                  tap anywhere to continue
                </motion.div>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── CORNER MASCOT ──────────────────────────────── */}
      <AnimatePresence>
        {phase === 'idle' && (
          <motion.div
            key="corner"
            ref={buddyRef}
            initial={{ scale: 0, opacity: 0, y: 30 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.2 }}
            onClick={handleClick}
            className="fixed bottom-5 right-5 z-[100] cursor-pointer select-none"
          >
            {/* Speech Bubble */}
            <AnimatePresence>
              {showBubble && bubbleMsg && (
                <motion.div
                  key="bubble"
                  initial={{ opacity: 0, y: 8, scale: 0.85 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.85 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 22 }}
                  className="absolute bottom-full right-0 mb-3 bg-white rounded-xl px-3.5 py-2 shadow-lg border border-slate-200 whitespace-nowrap pointer-events-none"
                >
                  <div className="absolute -bottom-1.5 right-5 w-3 h-3 bg-white border-r border-b border-slate-200 rotate-45" />
                  <div className="text-slate-700 font-medium text-xs">{bubbleMsg}</div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Mascot body with idle float */}
            <motion.div
              animate={clicked
                ? { scale: [1, 1.25, 0.9, 1.1, 1], rotate: [0, 12, -12, 5, 0] }
                : { y: [0, -4, 0] }
              }
              transition={clicked
                ? { duration: 0.5 }
                : { repeat: Infinity, duration: 3, ease: 'easeInOut' }
              }
              whileHover={{ scale: 1.12 }}
              className="relative"
            >
              {/* Glow ring */}
              <div className="absolute -inset-1 rounded-full bg-gradient-to-br from-indigo-400 to-purple-400 opacity-30 blur-sm animate-pulse" />
              <div className="relative ring-2 ring-white rounded-full">
                {renderFace(56, true)}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
