import React, { useEffect, useMemo, useState } from 'react';
import { Calendar, Plus, RefreshCcw, Search, Trash2 } from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';
import { motion } from 'framer-motion';

function formatDate(d) {
  if (!d) return '—';
  return String(d);
}

const emptySession = { userId: '', sessionDate: '', startTime: '', endTime: '', durationMinutes: 45, sessionType: 'personal', notes: '', cost: '' };

export default function TrainerSessionsPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptySession);
  const [members, setMembers] = useState([]);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/trainer/sessions')
      .then((res) => setRows(res.data || []))
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load sessions');
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
    api.get('/api/users').then((res) => setMembers((res.data || []).filter((u) => u.role === 'member'))).catch(() => null);
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api.post('/api/trainer/sessions', {
        ...form,
        durationMinutes: Number(form.durationMinutes) || 45,
        cost: form.cost ? Number(form.cost) : undefined,
      });
      toast('Session created!', 'success');
      setForm(emptySession);
      setShowForm(false);
      load();
    } catch (e2) {
      toast(e2?.response?.data?.error || 'Failed to create session', 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await api.delete(`/api/trainer/sessions/${id}`);
      toast('Session deleted', 'info');
      load();
    } catch (e2) {
      toast(e2?.response?.data?.error || 'Failed to delete', 'error');
    }
  };

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return rows;
    return rows.filter((r) =>
      [
        r.member_name,
        r.member_email,
        r.sessionType,
        r.notes,
        r.status,
        r.sessionDate,
        r.startTime,
        r.endTime,
      ]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(s))
    );
  }, [rows, q]);

  return (
    <PageShell
      title="My Sessions"
      subtitle="Only sessions assigned to you (trainer-scoped endpoint)."
      right={
        <>
          <button onClick={() => setShowForm(!showForm)} className="btn-ghost text-sm">
            <Plus className="h-4 w-4" />
            New Session
          </button>
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
      {showForm && (
        <div className="card">
          <div className="text-slate-800 font-semibold mb-3">Create Session</div>
          <form onSubmit={handleCreate} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="label">Member</label>
              <select className="input" value={form.userId} onChange={(e) => setForm((s) => ({ ...s, userId: e.target.value }))} required>
                <option value="">Select member</option>
                {members.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.email})</option>)}
              </select>
            </div>
            <div><label className="label">Date</label><input type="date" className="input" value={form.sessionDate} onChange={(e) => setForm((s) => ({ ...s, sessionDate: e.target.value }))} required /></div>
            <div><label className="label">Start Time</label><input type="time" className="input" value={form.startTime} onChange={(e) => setForm((s) => ({ ...s, startTime: e.target.value }))} required /></div>
            <div><label className="label">End Time</label><input type="time" className="input" value={form.endTime} onChange={(e) => setForm((s) => ({ ...s, endTime: e.target.value }))} required /></div>
            <div>
              <label className="label">Type</label>
              <select className="input" value={form.sessionType} onChange={(e) => setForm((s) => ({ ...s, sessionType: e.target.value }))}>
                <option value="personal">Personal</option>
                <option value="group">Group</option>
                <option value="assessment">Assessment</option>
              </select>
            </div>
            <div><label className="label">Duration (min)</label><input type="number" className="input" value={form.durationMinutes} onChange={(e) => setForm((s) => ({ ...s, durationMinutes: e.target.value }))} /></div>
            <div><label className="label">Cost ($)</label><input type="number" step="0.01" className="input" value={form.cost} onChange={(e) => setForm((s) => ({ ...s, cost: e.target.value }))} /></div>
            <div className="md:col-span-2"><label className="label">Notes</label><input className="input" value={form.notes} onChange={(e) => setForm((s) => ({ ...s, notes: e.target.value }))} /></div>
            <div className="md:col-span-2 lg:col-span-3 flex justify-end gap-2">
              <button type="button" onClick={() => setShowForm(false)} className="btn-ghost">Cancel</button>
              <button type="submit" disabled={busy} className="btn-primary">{busy ? 'Creating…' : 'Create Session'}</button>
            </div>
          </form>
        </div>
      )}
      <div className="card p-0 overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="text-slate-700 font-bold flex items-center gap-2">
            <Calendar className="h-4 w-4 text-slate-600" />
            Sessions
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
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Time</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Member</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Type</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Status</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3 w-16"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r, i) => (
                  <motion.tr
                    key={r.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03 }}
                    className="border-t border-slate-100 hover:bg-slate-50 transition-colors"
                  >
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.sessionDate ? new Date(r.sessionDate).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.startTime || '—'} – {r.endTime || '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      <div className="font-semibold text-slate-900">{r.member_name || '—'}</div>
                      <div className="text-slate-500 text-xs">{r.member_email || ''}</div>
                    </td>
                    <td className="px-4 py-3 text-slate-700">{r.sessionType || '—'}</td>
                    <td className="px-4 py-3 text-slate-700">
                      <span className="inline-flex px-2 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                        {r.status || 'scheduled'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <button onClick={() => handleDelete(r.id || r._id)} className="text-rose-500 hover:text-rose-700 transition-colors">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 ? <div className="p-4 text-slate-500">No sessions found.</div> : null}
          </div>
        ) : null}
      </div>
    </PageShell>
  );
}
