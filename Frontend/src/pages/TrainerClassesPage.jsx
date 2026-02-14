import React, { useEffect, useMemo, useState } from 'react';
import { Dumbbell, Plus, RefreshCcw, Search, Trash2 } from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';
import { motion } from 'framer-motion';

const emptyClass = { name: '', description: '', durationMinutes: 60, difficultyLevel: 'beginner', category: 'cardio', maxParticipants: 20 };

export default function TrainerClassesPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyClass);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/trainer/classes')
      .then((res) => setRows(res.data || []))
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load classes');
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      await api.post('/api/trainer/classes', {
        ...form,
        durationMinutes: Number(form.durationMinutes) || 60,
        maxParticipants: Number(form.maxParticipants) || 20,
      });
      toast('Class created!', 'success');
      setForm(emptyClass);
      setShowForm(false);
      load();
    } catch (e2) {
      toast(e2?.response?.data?.error || 'Failed to create class', 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (id) => {
    try {
      await api.delete(`/api/trainer/classes/${id}`);
      toast('Class deleted', 'info');
      load();
    } catch (e2) {
      toast(e2?.response?.data?.error || 'Failed to delete', 'error');
    }
  };

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return rows;
    return rows.filter((r) =>
      [r.name, r.category, r.difficultyLevel, r.description, r.durationMinutes]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(s))
    );
  }, [rows, q]);

  return (
    <PageShell
      title="My Classes"
      subtitle="Classes you teach (trainer-scoped endpoint)."
      right={
        <>
          <button onClick={() => setShowForm(!showForm)} className="btn-ghost text-sm">
            <Plus className="h-4 w-4" />
            New Class
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
          <div className="text-slate-800 font-semibold mb-3">Create Class</div>
          <form onSubmit={handleCreate} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div><label className="label">Name</label><input className="input" value={form.name} onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))} required /></div>
            <div>
              <label className="label">Category</label>
              <select className="input" value={form.category} onChange={(e) => setForm((s) => ({ ...s, category: e.target.value }))}>
                <option value="cardio">Cardio</option>
                <option value="strength">Strength</option>
                <option value="yoga">Yoga</option>
                <option value="hiit">HIIT</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div>
              <label className="label">Difficulty</label>
              <select className="input" value={form.difficultyLevel} onChange={(e) => setForm((s) => ({ ...s, difficultyLevel: e.target.value }))}>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
            <div><label className="label">Duration (min)</label><input type="number" className="input" value={form.durationMinutes} onChange={(e) => setForm((s) => ({ ...s, durationMinutes: e.target.value }))} /></div>
            <div><label className="label">Max Participants</label><input type="number" className="input" value={form.maxParticipants} onChange={(e) => setForm((s) => ({ ...s, maxParticipants: e.target.value }))} /></div>
            <div className="md:col-span-2"><label className="label">Description</label><textarea className="input" rows={2} value={form.description} onChange={(e) => setForm((s) => ({ ...s, description: e.target.value }))} /></div>
            <div className="md:col-span-2 lg:col-span-3 flex justify-end gap-2">
              <button type="button" onClick={() => setShowForm(false)} className="btn-ghost">Cancel</button>
              <button type="submit" disabled={busy} className="btn-primary">{busy ? 'Creating…' : 'Create Class'}</button>
            </div>
          </form>
        </div>
      )}
      <div className="card p-0 overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="text-slate-700 font-bold flex items-center gap-2">
            <Dumbbell className="h-4 w-4 text-slate-600" />
            Classes
          </div>
          <div className="text-slate-500 text-xs">
            {loading ? 'Loading…' : `${filtered.length} shown`}
          </div>
        </div>

        {error ? <div className="p-4 text-red-600 text-sm">{error}</div> : null}
        {loading ? <div className="p-4 text-slate-500">Loading…</div> : null}

        {!loading && !error ? (
          <div className="grid gap-3 p-4">
            {filtered.map((c, i) => (
              <motion.div
                key={c.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
                whileHover={{ y: -2 }}
                className="rounded-2xl border border-slate-200 bg-white p-4 hover:bg-slate-50 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-slate-900 font-bold text-lg">{c.name}</div>
                    <div className="text-slate-500 text-sm">
                      {c.category || 'Class'} • {c.difficultyLevel || 'All levels'} • {c.durationMinutes || '—'} min
                    </div>
                    {c.description ? (
                      <div className="text-slate-500 text-sm mt-2">{c.description}</div>
                    ) : null}
                  </div>
                  <div className="text-slate-400 text-xs flex items-center gap-2">
                    ID: {c.id}
                    <button onClick={() => handleDelete(c.id || c._id)} className="text-rose-500 hover:text-rose-700 transition-colors ml-2">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
            {filtered.length === 0 ? <div className="text-slate-500">No classes found.</div> : null}
          </div>
        ) : null}
      </div>
    </PageShell>
  );
}
