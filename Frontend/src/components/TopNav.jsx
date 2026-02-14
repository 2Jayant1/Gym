import React, { useEffect, useState, useRef, useCallback, useLayoutEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Dumbbell, Bell, LogOut, Sparkles, ChevronDown } from 'lucide-react';
import api from '../services/api';

/* ─── Animated sliding pill indicator ─────────────────────── */
function PillIndicator({ tabs, activeTab, onTabChange }) {
  const containerRef = useRef(null);
  const [indicator, setIndicator] = useState({ left: 0, width: 0 });
  const tabRefs = useRef({});

  const updateIndicator = useCallback(() => {
    const el = tabRefs.current[activeTab];
    const container = containerRef.current;
    if (el && container) {
      const cr = container.getBoundingClientRect();
      const er = el.getBoundingClientRect();
      setIndicator({ left: er.left - cr.left, width: er.width });
    }
  }, [activeTab]);

  useLayoutEffect(() => { updateIndicator(); }, [updateIndicator]);

  useEffect(() => {
    window.addEventListener('resize', updateIndicator);
    return () => window.removeEventListener('resize', updateIndicator);
  }, [updateIndicator]);

  return (
    <div ref={containerRef} className="hidden md:flex items-center relative rounded-full bg-slate-100/80 backdrop-blur-sm border border-slate-200/60 px-1 py-1">
      <motion.div
        className="absolute top-1 bottom-1 rounded-full z-0"
        style={{ background: 'linear-gradient(135deg, #1e293b 0%, #475569 100%)' }}
        initial={false}
        animate={{ left: indicator.left, width: indicator.width }}
        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      />
      {tabs.map((tab) => (
        <button
          key={tab.key}
          ref={(el) => { tabRefs.current[tab.key] = el; }}
          onClick={() => onTabChange(tab.key)}
          className={`relative z-10 px-4 py-1.5 rounded-full text-sm font-medium transition-colors duration-200 ${
            activeTab === tab.key ? 'text-white' : 'text-slate-500 hover:text-slate-800'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

/* ─── Animated notification bell ──────────────────────────── */
function AnimatedBell({ unread, onClick }) {
  return (
    <motion.button
      onClick={onClick}
      className="relative p-2.5 rounded-xl hover:bg-slate-100 transition-colors group"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <motion.div
        animate={unread > 0 ? { rotate: [0, -15, 15, -10, 10, 0] } : {}}
        transition={{ duration: 0.5, repeat: unread > 0 ? Infinity : 0, repeatDelay: 4 }}
      >
        <Bell className="h-5 w-5 text-slate-600 group-hover:text-slate-800 transition-colors" />
      </motion.div>
      <AnimatePresence>
        {unread > 0 && (
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="absolute -top-0.5 -right-0.5 h-5 min-w-[20px] flex items-center justify-center rounded-full bg-gradient-to-r from-rose-500 to-pink-500 text-white text-[10px] font-bold px-1 shadow-lg shadow-rose-500/30"
          >
            {unread > 9 ? '9+' : unread}
          </motion.span>
        )}
      </AnimatePresence>
      {unread > 0 && (
        <span className="absolute -top-0.5 -right-0.5 h-5 w-5 rounded-full bg-rose-400 animate-ping opacity-30 pointer-events-none" />
      )}
    </motion.button>
  );
}

/* ─── User avatar with status ─────────────────────────────── */
function UserAvatar({ name, role }) {
  const initials = (name || 'U').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
  const gradients = {
    admin: 'from-violet-500 to-purple-600',
    trainer: 'from-amber-500 to-orange-600',
    member: 'from-sky-500 to-cyan-600',
  };
  return (
    <div className="relative">
      <motion.div
        whileHover={{ scale: 1.08 }}
        className={`h-9 w-9 rounded-xl bg-gradient-to-br ${gradients[role] || gradients.member} flex items-center justify-center text-white text-xs font-bold shadow-md cursor-default`}
      >
        {initials}
      </motion.div>
      <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full bg-emerald-400 border-2 border-white" />
    </div>
  );
}

/* ─── Role badge ──────────────────────────────────────────── */
function RoleBadge({ role }) {
  const styles = {
    admin: 'from-violet-500/15 to-purple-500/15 text-violet-700 border-violet-200/60',
    trainer: 'from-amber-500/15 to-orange-500/15 text-amber-700 border-amber-200/60',
    member: 'from-sky-500/15 to-cyan-500/15 text-sky-700 border-sky-200/60',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-gradient-to-r ${styles[role] || styles.member} border`}>
      <Sparkles className="h-2.5 w-2.5" />
      {role}
    </span>
  );
}

export default function TopNav({ role, name, tabs, activeTab, onTabChange, onLogout }) {
  const [unread, setUnread] = useState(0);
  const [notifs, setNotifs] = useState([]);
  const [showNotifs, setShowNotifs] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const dropdownRef = useRef(null);
  const profileRef = useRef(null);

  const fetchUnread = useCallback(() => {
    api.get('/api/me/notifications/unread-count').then((r) => setUnread(r.data?.count || 0)).catch(() => null);
  }, []);

  useEffect(() => {
    fetchUnread();
    const interval = setInterval(fetchUnread, 30000);
    return () => clearInterval(interval);
  }, [fetchUnread]);

  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) setShowNotifs(false);
      if (profileRef.current && !profileRef.current.contains(e.target)) setShowProfile(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const toggleNotifs = () => {
    if (!showNotifs) {
      api.get('/api/me/notifications').then((r) => setNotifs(r.data || [])).catch(() => null);
    }
    setShowNotifs(!showNotifs);
    setShowProfile(false);
  };

  const markAllRead = () => {
    api.patch('/api/me/notifications/read-all').then(() => { setUnread(0); fetchUnread(); }).catch(() => null);
  };

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="sticky top-0 z-40 relative"
    >
      {/* Animated gradient top bar */}
      <div className="absolute inset-x-0 top-0 h-[3px] animate-gradient-x" style={{ background: 'linear-gradient(90deg, #6366f1, #38bdf8, #10b981, #a855f7, #f43e5c, #6366f1)', backgroundSize: '300% 100%' }} />

      {/* Glass background */}
      <div className="glass-effect border-b border-slate-200/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <motion.div className="flex items-center space-x-2.5" whileHover={{ scale: 1.02 }} transition={{ type: 'spring', stiffness: 300 }}>
              <div className="relative">
                <motion.div
                  className="h-10 w-10 rounded-xl bg-gradient-to-br from-slate-800 via-slate-900 to-slate-800 text-white flex items-center justify-center shadow-lg"
                  whileHover={{ rotate: [0, -5, 5, 0] }}
                  transition={{ duration: 0.4 }}
                >
                  <Dumbbell className="h-5 w-5" />
                </motion.div>
                <div className="absolute inset-0 rounded-xl bg-indigo-500/20 blur-md animate-soft-pulse -z-10" />
              </div>
              <div className="hidden xs:block">
                <div className="text-slate-900 text-lg font-extrabold leading-tight tracking-tight">FitFlex</div>
                <div className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest leading-tight">Studio</div>
              </div>
            </motion.div>

            {/* Center: Animated tab pills */}
            <PillIndicator tabs={tabs} activeTab={activeTab} onTabChange={onTabChange} />

            {/* Right side */}
            <div className="flex items-center space-x-2">
              {/* Notification bell */}
              <div className="relative" ref={dropdownRef}>
                <AnimatedBell unread={unread} onClick={toggleNotifs} />
                <AnimatePresence>
                  {showNotifs && (
                    <motion.div
                      initial={{ opacity: 0, y: 8, scale: 0.96 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 8, scale: 0.96 }}
                      transition={{ duration: 0.2 }}
                      className="absolute right-0 mt-2 w-80 max-h-96 bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl shadow-slate-900/10 border border-slate-200/80 overflow-hidden z-50"
                    >
                      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-slate-50/50">
                        <span className="text-slate-800 font-bold text-sm">Notifications</span>
                        {unread > 0 && (
                          <button onClick={markAllRead} className="text-indigo-600 hover:text-indigo-700 text-xs font-semibold hover:underline">Mark all read</button>
                        )}
                      </div>
                      <div className="overflow-y-auto max-h-72">
                        {notifs.length === 0 && (
                          <div className="p-8 text-slate-400 text-sm text-center">
                            <Bell className="h-8 w-8 mx-auto mb-2 opacity-30" />
                            All caught up!
                          </div>
                        )}
                        {notifs.map((n, i) => (
                          <motion.div
                            key={n._id}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.03 }}
                            className={`px-4 py-3 border-b border-slate-50 hover:bg-indigo-50/30 transition-colors cursor-default ${!n.read ? 'bg-indigo-50/40' : ''}`}
                          >
                            <div className="flex items-start gap-3">
                              <div className={`h-2 w-2 rounded-full mt-1.5 shrink-0 ${!n.read ? 'bg-indigo-500' : 'bg-slate-200'}`} />
                              <div>
                                <div className="text-slate-800 text-sm font-medium">{n.title}</div>
                                <div className="text-slate-500 text-xs mt-0.5">{n.message}</div>
                                <div className="text-slate-300 text-[10px] mt-1">{n.createdAt ? new Date(n.createdAt).toLocaleString() : ''}</div>
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Profile section */}
              <div className="relative" ref={profileRef}>
                <motion.button
                  onClick={() => { setShowProfile(!showProfile); setShowNotifs(false); }}
                  className="flex items-center gap-2.5 pl-2 pr-3 py-1.5 rounded-xl hover:bg-slate-100/80 transition-colors"
                  whileTap={{ scale: 0.97 }}
                >
                  <UserAvatar name={name} role={role} />
                  <div className="hidden sm:block text-left">
                    <div className="text-slate-800 text-sm font-semibold leading-tight">{name}</div>
                    <RoleBadge role={role} />
                  </div>
                  <ChevronDown className={`h-3.5 w-3.5 text-slate-400 transition-transform duration-200 hidden sm:block ${showProfile ? 'rotate-180' : ''}`} />
                </motion.button>
                <AnimatePresence>
                  {showProfile && (
                    <motion.div
                      initial={{ opacity: 0, y: 8, scale: 0.96 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 8, scale: 0.96 }}
                      transition={{ duration: 0.2 }}
                      className="absolute right-0 mt-2 w-56 bg-white/95 backdrop-blur-xl rounded-2xl shadow-2xl shadow-slate-900/10 border border-slate-200/80 overflow-hidden z-50"
                    >
                      <div className="p-4 border-b border-slate-100 bg-gradient-to-br from-slate-50 to-white">
                        <div className="flex items-center gap-3">
                          <UserAvatar name={name} role={role} />
                          <div>
                            <div className="text-slate-800 font-bold text-sm">{name}</div>
                            <RoleBadge role={role} />
                          </div>
                        </div>
                      </div>
                      <div className="p-1.5">
                        <motion.button
                          onClick={onLogout}
                          whileHover={{ x: 2 }}
                          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm text-rose-600 hover:bg-rose-50 transition-colors font-medium"
                        >
                          <LogOut className="h-4 w-4" />
                          Sign out
                        </motion.button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* Mobile tabs */}
          <div className="md:hidden pb-3 flex gap-1.5 overflow-x-auto scrollbar-hide -mx-4 px-4 sm:-mx-6 sm:px-6">
            {tabs.map((tab) => (
              <motion.button
                key={tab.key}
                onClick={() => onTabChange(tab.key)}
                whileTap={{ scale: 0.95 }}
                className={`whitespace-nowrap text-xs sm:text-sm px-3.5 py-1.5 rounded-full font-medium transition-all duration-200 ${
                  activeTab === tab.key ? 'text-white shadow-sm' : 'text-slate-500 hover:text-slate-800 hover:bg-white/80'
                }`}
                style={activeTab === tab.key ? { background: 'linear-gradient(135deg, #1e293b, #334155)' } : {}}
              >
                {tab.label}
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </motion.nav>
  );
}
