import React, { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, LogIn, LogOut, RefreshCcw, Search } from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';
import { motion } from 'framer-motion';

export default function MemberAttendancePage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');
  const [checkedIn, setCheckedIn] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const toast = useToast();

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/me/attendance')
      .then((res) => {
        const data = res.data || [];
        setRows(data);
        // If the latest record has no checkOut, user is currently checked in
        setCheckedIn(data.length > 0 && !data[0].checkOut);
      })
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load attendance');
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  const handleCheckIn = async () => {
    setActionBusy(true);
    try {
      await api.post('/api/me/check-in');
      toast('Checked in successfully!', 'success');
      load();
    } catch (e) {
      toast(e?.response?.data?.error || 'Check-in failed', 'error');
    } finally {
      setActionBusy(false);
    }
  };

  const handleCheckOut = async () => {
    setActionBusy(true);
    try {
      await api.post('/api/me/check-out');
      toast('Checked out successfully!', 'success');
      load();
    } catch (e) {
      toast(e?.response?.data?.error || 'Check-out failed', 'error');
    } finally {
      setActionBusy(false);
    }
  };

  // Stats
  const totalVisits = rows.length;
  const totalMinutes = rows.reduce((sum, r) => sum + (r.durationMinutes || 0), 0);
  const avgMinutes = totalVisits > 0 ? Math.round(totalMinutes / totalVisits) : 0;

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return rows;
    return rows.filter((r) =>
      [r.date, r.checkIn, r.checkOut].some((v) =>
        String(v ?? '')
          .toLowerCase()
          .includes(s)
      )
    );
  }, [rows, q]);

  return (
    <PageShell
      title="Attendance"
      subtitle="Your check-ins for the last 90 days."
      right={
        <>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search…"
              className="input pl-9"
            />
          </div>
          <button onClick={load} className="btn-ghost" title="Refresh">
            <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
            Refresh
          </button>
        </>
      }
    >
      {/* Check-in/out action + stats */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card flex flex-col items-center justify-center py-6">
          {checkedIn ? (
            <button onClick={handleCheckOut} disabled={actionBusy} className="btn-primary bg-rose-600 hover:bg-rose-700 flex items-center gap-2">
              <LogOut className="h-4 w-4" />
              {actionBusy ? 'Processing…' : 'Check Out'}
            </button>
          ) : (
            <button onClick={handleCheckIn} disabled={actionBusy} className="btn-primary bg-emerald-600 hover:bg-emerald-700 flex items-center gap-2">
              <LogIn className="h-4 w-4" />
              {actionBusy ? 'Processing…' : 'Check In'}
            </button>
          )}
          <span className="text-xs text-slate-500 mt-2">{checkedIn ? 'You are currently checked in' : 'Tap to check in'}</span>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-slate-800">{totalVisits}</div>
          <div className="text-xs text-slate-500">Total Visits</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-slate-800">{Math.round(totalMinutes / 60)}h</div>
          <div className="text-xs text-slate-500">Total Hours</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-slate-800">{avgMinutes}m</div>
          <div className="text-xs text-slate-500">Avg. Duration</div>
        </div>
      </motion.div>
      <div className="card p-0 overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="text-slate-700 font-bold flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-slate-600" /> Attendance
          </div>
          <div className="text-slate-500 text-xs">
            {loading ? 'Loading…' : `${filtered.length} shown`}
          </div>
        </div>

        {error ? <div className="p-4 text-red-600 text-sm">{error}</div> : null}
        {loading ? <div className="p-4 text-slate-500">Loading…</div> : null}

        {!loading && !error ? (
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Date</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Check-in</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Check-out</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Duration</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <motion.tr
                    key={r._id || r.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03 }}
                    className="border-t border-slate-100 hover:bg-slate-50 transition-colors"
                  >
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.date ? new Date(r.date).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.checkIn ? new Date(r.checkIn).toLocaleTimeString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.checkOut ? new Date(r.checkOut).toLocaleTimeString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.durationMinutes ? `${r.durationMinutes} min` : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      <span className="inline-flex px-2 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                        {r.checkOut ? 'checked out' : 'checked in'}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 ? <div className="p-4 text-slate-500">No attendance records.</div> : null}
          </div>
        ) : null}
      </div>
    </PageShell>
  );
}
