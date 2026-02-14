import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

/* ─── Animated counter (spring-physics counting) ─────────── */
function AnimatedCounter({ value }) {
  const [display, setDisplay] = useState('0');
  const prevRef = useRef(0);

  useEffect(() => {
    const numericVal = parseFloat(String(value).replace(/[^0-9.-]/g, ''));
    if (isNaN(numericVal)) {
      setDisplay(value);
      return;
    }
    const prefix = String(value).match(/^[^0-9.-]*/)?.[0] || '';
    const suffix = String(value).match(/[^0-9.-]*$/)?.[0] || '';
    const isFloat = String(value).includes('.');
    const from = prevRef.current;
    const to = numericVal;
    prevRef.current = to;
    const duration = 1200;
    const start = performance.now();

    const tick = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = from + (to - from) * eased;
      setDisplay(`${prefix}${isFloat ? current.toFixed(1) : Math.round(current)}${suffix}`);
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value]);

  return <>{display}</>;
}

/* ─── 3D Tilt Card ────────────────────────────────────────── */
export default function StatCard({ title, value, Icon, colorClass = 'text-slate-700' }) {
  const cardRef = useRef(null);
  const mouseX = useMotionValue(0.5);
  const mouseY = useMotionValue(0.5);

  const rotateX = useSpring(useTransform(mouseY, [0, 1], [6, -6]), { stiffness: 200, damping: 20 });
  const rotateY = useSpring(useTransform(mouseX, [0, 1], [-6, 6]), { stiffness: 200, damping: 20 });
  const glareX = useTransform(mouseX, [0, 1], [0, 100]);
  const glareY = useTransform(mouseY, [0, 1], [0, 100]);

  const handleMouseMove = useCallback(
    (e) => {
      const rect = cardRef.current?.getBoundingClientRect();
      if (!rect) return;
      mouseX.set((e.clientX - rect.left) / rect.width);
      mouseY.set((e.clientY - rect.top) / rect.height);
    },
    [mouseX, mouseY],
  );

  const handleMouseLeave = useCallback(() => {
    mouseX.set(0.5);
    mouseY.set(0.5);
  }, [mouseX, mouseY]);

  return (
    <motion.div
      ref={cardRef}
      initial={{ opacity: 0, y: 18, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ rotateX, rotateY, transformPerspective: 800 }}
      className="card relative overflow-hidden group cursor-default"
    >
      {/* Mouse-follow gradient glow */}
      <motion.div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{
          background: useTransform(
            [glareX, glareY],
            ([x, y]) => `radial-gradient(circle at ${x}% ${y}%, rgba(99,102,241,0.08) 0%, transparent 60%)`,
          ),
        }}
      />
      {/* Shimmer sweep */}
      <div
        className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 ease-in-out pointer-events-none"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent)', width: '60%' }}
      />
      {/* Decorative corner accent */}
      <div className="absolute -right-6 -top-6 h-20 w-20 rounded-full bg-gradient-to-br from-slate-100 to-slate-50 group-hover:scale-125 transition-transform duration-500" />
      {/* Accent bar */}
      <div className="absolute left-0 top-0 bottom-0 w-1 rounded-full bg-gradient-to-b from-indigo-400 via-sky-400 to-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      <div className="relative flex items-center justify-between">
        <div className="min-w-0 flex-1">
          <p className="text-slate-500 text-xs sm:text-sm font-medium">{title}</p>
          <p className="text-2xl sm:text-3xl font-extrabold text-slate-900 mt-1 truncate tabular-nums">
            <AnimatedCounter value={value} />
          </p>
        </div>
        {Icon ? (
          <motion.div
            whileHover={{ rotate: 12, scale: 1.15 }}
            transition={{ type: 'spring', stiffness: 300 }}
            className="relative"
          >
            <Icon
              className={`h-9 w-9 sm:h-12 sm:w-12 shrink-0 ml-3 ${colorClass} group-hover:scale-110 transition-transform duration-300`}
            />
            {/* Glow behind icon */}
            <div
              className={`absolute inset-0 blur-lg opacity-0 group-hover:opacity-30 transition-opacity duration-500 ${colorClass}`}
              style={{ background: 'currentColor', borderRadius: '50%' }}
            />
          </motion.div>
        ) : null}
      </div>
    </motion.div>
  );
}
