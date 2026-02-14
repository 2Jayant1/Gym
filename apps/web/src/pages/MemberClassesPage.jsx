import React, { useEffect, useMemo, useState } from 'react';
import { CalendarDays, RefreshCcw, Search, UserPlus, XCircle } from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';
import { motion } from 'framer-motion';

export default function MemberClassesPage() {
  const [classes, setClasses] = useState([]);
  const [registrations, setRegistrations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(null); // scheduleId being acted on
  const toast = useToast();

  function load() {
    setLoading(true);
    setError(null);
    Promise.all([api.get('/api/me/classes'), api.get('/api/me/registrations')])
      .then(([c, r]) => {
        // Backend returns schedules; group them by class
        const schedulesRaw = c.data || [];
        const classMap = {};
        for (const s of schedulesRaw) {
          const classId = s.fitnessClass?._id || s.fitnessClass || 'unknown';
          if (!classMap[classId]) {
            classMap[classId] = {
              _id: classId,
              name: s.className || s.fitnessClass?.name || 'Class',
              category: s.classCategory || s.fitnessClass?.category || '',
              difficultyLevel: s.classLevel || s.fitnessClass?.difficultyLevel || '',
              durationMinutes: s.classDuration || s.fitnessClass?.durationMinutes || 0,
              description: s.fitnessClass?.description || '',
              maxParticipants: s.maxParticipants || s.fitnessClass?.maxParticipants || '',
              trainer_name: s.trainerName || '',
              schedules: [],
            };
          }
          classMap[classId].schedules.push(s);
        }
        setClasses(Object.values(classMap));
        setRegistrations(r.data || []);
      })
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load classes'))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  const regScheduleIds = useMemo(
    () => new Set(registrations.filter((r) => r.status !== 'cancelled').map((r) => r.classSchedule?._id || r.classSchedule)),
    [registrations]
  );

  const handleRegister = async (scheduleId) => {
    setBusy(scheduleId);
    try {
      await api.post('/api/me/classes/register', { classScheduleId: scheduleId });
      toast('Registered for class!', 'success');
      load();
    } catch (e) {
      toast(e?.response?.data?.error || 'Registration failed', 'error');
    } finally {
      setBusy(null);
    }
  };

  const handleCancel = async (regId) => {
    setBusy(regId);
    try {
      await api.patch(`/api/me/registrations/${regId}/cancel`);
      toast('Registration cancelled', 'info');
      load();
    } catch (e) {
      toast(e?.response?.data?.error || 'Cancellation failed', 'error');
    } finally {
      setBusy(null);
    }
  };

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return classes;
    return classes.filter((c) =>
      [c.name, c.category, c.difficultyLevel, c.trainer_name]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(s))
    );
  }, [classes, q]);

  return (
    <PageShell
      title="Classes"
      subtitle="Browse and register for fitness classes."
      right={
        <>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search classes…" className="input pl-9" />
          </div>
          <button onClick={load} className="btn-ghost" title="Refresh">
            <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
            Refresh
          </button>
        </>
      }
    >
      {/* My Registrations */}
      {registrations.length > 0 && (
        <div className="card">
          <h2 className="text-slate-800 font-bold mb-3">My Registrations</h2>
          <div className="grid gap-3">
            {registrations
              .filter((r) => r.status !== 'cancelled')
              .map((r) => {
                const sched = r.classSchedule || {};
                const cls = sched.fitnessClass || {};
                return (
                  <div key={r._id} className="flex items-center justify-between rounded-xl border border-slate-200 bg-white p-4">
                    <div>
                      <div className="text-slate-900 font-semibold">{cls.name || 'Class'}</div>
                      <div className="text-slate-500 text-sm">
                        {sched.classDate ? new Date(sched.classDate).toLocaleDateString() : '—'} •{' '}
                        {sched.startTime || '—'} – {sched.endTime || '—'} • {sched.room || ''}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                        {r.status}
                      </span>
                      <button
                        onClick={() => handleCancel(r._id)}
                        disabled={busy === r._id}
                        className="btn-ghost text-rose-600 border-rose-200/60 hover:text-rose-700 text-xs"
                      >
                        <XCircle className="h-3.5 w-3.5" />
                        Cancel
                      </button>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Available Classes */}
      {error && <div className="card text-red-600 border border-red-200/60">{error}</div>}
      {loading && <div className="card text-slate-500">Loading classes…</div>}

      {!loading && !error && (
        <div className="card p-0 overflow-hidden">
          <div className="p-4 border-b border-slate-200 flex items-center justify-between">
            <div className="text-slate-700 font-bold flex items-center gap-2">
              <CalendarDays className="h-4 w-4 text-slate-600" /> Available Classes
            </div>
            <div className="text-slate-500 text-xs">{filtered.length} classes</div>
          </div>
          <div className="grid gap-3 p-4">
            {filtered.map((c, i) => (
              <motion.div key={c._id || c.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }} whileHover={{ y: -2 }} className="rounded-2xl border border-slate-200 bg-white p-4 hover:bg-slate-50 transition-colors">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-slate-900 font-bold text-lg">{c.name}</div>
                    <div className="text-slate-500 text-sm">
                      {c.category || 'Class'} • {c.difficultyLevel || 'All levels'} • {c.durationMinutes || '—'} min
                      {c.trainer_name ? ` • ${c.trainer_name}` : ''}
                    </div>
                    {c.description && <div className="text-slate-500 text-sm mt-1">{c.description}</div>}
                    <div className="text-slate-400 text-xs mt-2">
                      Max participants: {c.maxParticipants || '—'}
                    </div>
                  </div>
                </div>

                {/* Schedules */}
                {c.schedules && c.schedules.length > 0 && (
                  <div className="mt-3 space-y-2">
                    {c.schedules.slice(0, 5).map((s) => {
                      const isRegistered = regScheduleIds.has(s._id);
                      return (
                        <div key={s._id} className="flex items-center justify-between bg-slate-50 rounded-lg px-3 py-2 text-sm">
                          <div className="text-slate-600">
                            {s.classDate ? new Date(s.classDate).toLocaleDateString() : '—'} •{' '}
                            {s.startTime || '—'} – {s.endTime || '—'} • {s.room || ''}
                          </div>
                          {isRegistered ? (
                            <span className="text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                              Registered
                            </span>
                          ) : (
                            <button
                              onClick={() => handleRegister(s._id)}
                              disabled={busy === s._id}
                              className="text-xs px-3 py-1 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors flex items-center gap-1"
                            >
                              <UserPlus className="h-3 w-3" />
                              {busy === s._id ? 'Registering…' : 'Register'}
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </motion.div>
            ))}
            {filtered.length === 0 && <div className="text-slate-500">No classes available.</div>}
          </div>
        </div>
      )}
    </PageShell>
  );
}
