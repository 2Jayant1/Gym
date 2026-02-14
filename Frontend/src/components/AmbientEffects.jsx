import React, { useRef, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';

/* ─── Mouse-reactive particle canvas for portal pages ─────── */
function PortalParticles() {
  const canvasRef = useRef(null);
  const mouseRef = useRef({ x: -9999, y: -9999 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf;
    let particles = [];
    const COLORS = [
      'rgba(99,102,241,0.22)',
      'rgba(14,165,233,0.18)',
      'rgba(16,185,129,0.18)',
      'rgba(168,85,247,0.15)',
      'rgba(244,63,94,0.12)',
    ];

    const onMouseMove = (e) => { mouseRef.current = { x: e.clientX, y: e.clientY }; };
    window.addEventListener('mousemove', onMouseMove, { passive: true });

    function resize() {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }

    function init() {
      resize();
      const count = Math.min(50, Math.floor((canvas.offsetWidth * canvas.offsetHeight) / 28000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * canvas.offsetWidth,
        y: Math.random() * canvas.offsetHeight,
        baseR: Math.random() * 2.5 + 0.8,
        r: Math.random() * 2.5 + 0.8,
        dx: (Math.random() - 0.5) * 0.35,
        dy: (Math.random() - 0.5) * 0.35,
        color: COLORS[Math.floor(Math.random() * COLORS.length)],
        pulse: Math.random() * Math.PI * 2,
      }));
    }

    function draw() {
      const w = canvas.offsetWidth;
      const h = canvas.offsetHeight;
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;
      ctx.clearRect(0, 0, w, h);

      for (const p of particles) {
        p.pulse += 0.015;
        // Mouse interaction — particles gently repel/attract near cursor
        const dmx = p.x - mx;
        const dmy = p.y - my;
        const distM = Math.sqrt(dmx * dmx + dmy * dmy);
        if (distM < 180) {
          const force = (180 - distM) / 180 * 0.4;
          p.x += dmx / distM * force;
          p.y += dmy / distM * force;
          p.r = p.baseR + (180 - distM) / 180 * 2; // grow near cursor
        } else {
          p.r += (p.baseR - p.r) * 0.05;
        }

        p.x += p.dx;
        p.y += p.dy;
        if (p.x < 0 || p.x > w) p.dx *= -1;
        if (p.y < 0 || p.y > h) p.dy *= -1;

        const pulseScale = 1 + Math.sin(p.pulse) * 0.2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * pulseScale, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
      }

      // Connection lines
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 150) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(99,102,241,${0.06 * (1 - dist / 150)})`;
            ctx.lineWidth = 0.6;
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
      window.removeEventListener('mousemove', onMouseMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full pointer-events-none z-0"
      style={{ opacity: 0.6 }}
    />
  );
}

/* ─── Smooth cursor glow (light theme - subtle) ──────────── */
function PortalGlowCursor() {
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
      current.current.x = lerp(current.current.x, mouse.current.x, 0.12);
      current.current.y = lerp(current.current.y, mouse.current.y, 0.12);
      if (ref.current) {
        ref.current.style.transform = `translate3d(${current.current.x - 250}px, ${current.current.y - 250}px, 0)`;
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
        width: 500,
        height: 500,
        background:
          'radial-gradient(circle, rgba(99,102,241,0.06) 0%, rgba(14,165,233,0.03) 35%, transparent 70%)',
        borderRadius: '50%',
        willChange: 'transform',
      }}
    />
  );
}

/* ─── Click ripple effect ─────────────────────────────────── */
function ClickRipple() {
  const [ripples, setRipples] = React.useState([]);

  useEffect(() => {
    const handler = (e) => {
      const id = Date.now();
      setRipples((prev) => [...prev.slice(-4), { id, x: e.clientX, y: e.clientY }]);
      setTimeout(() => setRipples((prev) => prev.filter((r) => r.id !== id)), 800);
    };
    window.addEventListener('click', handler);
    return () => window.removeEventListener('click', handler);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {ripples.map((r) => (
        <motion.div
          key={r.id}
          initial={{ scale: 0, opacity: 0.25 }}
          animate={{ scale: 4, opacity: 0 }}
          transition={{ duration: 0.7, ease: 'easeOut' }}
          className="absolute rounded-full"
          style={{
            left: r.x - 20,
            top: r.y - 20,
            width: 40,
            height: 40,
            border: '2px solid rgba(99,102,241,0.2)',
          }}
        />
      ))}
    </div>
  );
}

/* ─── Floating gradient orbs ──────────────────────────────── */
function FloatingOrbs() {
  const orbs = useMemo(
    () => [
      { size: 300, x: '5%', y: '-8%', color: 'rgba(99,102,241,0.07)', delay: 0, dur: 18 },
      { size: 250, x: '85%', y: '15%', color: 'rgba(14,165,233,0.06)', delay: -4, dur: 22 },
      { size: 350, x: '40%', y: '80%', color: 'rgba(16,185,129,0.05)', delay: -8, dur: 20 },
      { size: 200, x: '70%', y: '60%', color: 'rgba(168,85,247,0.05)', delay: -12, dur: 24 },
      { size: 180, x: '20%', y: '40%', color: 'rgba(244,63,94,0.04)', delay: -6, dur: 26 },
    ],
    [],
  );

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {orbs.map((orb, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            width: orb.size,
            height: orb.size,
            left: orb.x,
            top: orb.y,
            background: `radial-gradient(circle, ${orb.color} 0%, transparent 70%)`,
          }}
          animate={{
            y: [0, 25, 0, -20, 0],
            x: [0, 15, 0, -15, 0],
            scale: [1, 1.06, 1, 0.96, 1],
          }}
          transition={{
            duration: orb.dur,
            repeat: Infinity,
            ease: 'easeInOut',
            delay: orb.delay,
          }}
        />
      ))}
    </div>
  );
}

/* ─── Combined ambient effects wrapper ────────────────────── */
export default function AmbientEffects() {
  return (
    <>
      <FloatingOrbs />
      <PortalParticles />
      <PortalGlowCursor />
      <ClickRipple />
    </>
  );
}
