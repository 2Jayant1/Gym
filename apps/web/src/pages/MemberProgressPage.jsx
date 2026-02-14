import React, { useEffect, useMemo, useState } from 'react';
import { LineChart, RefreshCcw, Search, Plus } from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';
import { motion } from 'framer-motion';

const emptyEntry = {
  weightKg: '',
  bodyFatPercentage: '',
  muscleMassKg: '',
  chestCm: '',
  waistCm: '',
  hipsCm: '',
  notes: '',
};

export default function MemberProgressPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [entry, setEntry] = useState(emptyEntry);
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/me/progress')
      .then((res) => setRows(res.data || []))
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load progress');
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return rows;
    return rows.filter((r) =>
      [r.recordDate, r.weightKg, r.bodyFatPercentage, r.muscleMassKg, r.notes]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(s)),
    );
  }, [rows, q]);

  return (
    <PageShell
      title="Progress"
      subtitle="Weekly progress snapshots (latest 52)."
      right={
        <>
          <button onClick={() => setShowForm(!showForm)} className="btn-ghost text-sm">
            <Plus className="h-4 w-4" />
            Add Entry
          </button>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search…" className="input pl-9" />
          </div>
          <button onClick={load} className="btn-ghost" title="Refresh">
            <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
            Refresh
          </button>
        </>
      }
    >
      {/* Add Entry Form */}
      {showForm && (
        <div className="card">
          <div className="text-slate-800 font-semibold mb-3">Add Progress Entry</div>
          <form
            className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
            onSubmit={async (e) => {
              e.preventDefault();
              setBusy(true);
              try {
                const payload = {};
                for (const [k, v] of Object.entries(entry)) {
                  if (v !== '' && v !== null && v !== undefined) payload[k] = k === 'notes' ? v : Number(v);
                }
                await api.post('/api/me/progress', payload);
                toast('Progress entry added!', 'success');
                setEntry(emptyEntry);
                setShowForm(false);
                load();
              } catch (e2) {
                toast(e2?.response?.data?.error || 'Failed to add entry', 'error');
              } finally {
                setBusy(false);
              }
            }}
          >
            <div>
              <label className="label">Weight (kg)</label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={entry.weightKg}
                onChange={(e) => setEntry((s) => ({ ...s, weightKg: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Body Fat %</label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={entry.bodyFatPercentage}
                onChange={(e) => setEntry((s) => ({ ...s, bodyFatPercentage: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Muscle (kg)</label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={entry.muscleMassKg}
                onChange={(e) => setEntry((s) => ({ ...s, muscleMassKg: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Chest (cm)</label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={entry.chestCm}
                onChange={(e) => setEntry((s) => ({ ...s, chestCm: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Waist (cm)</label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={entry.waistCm}
                onChange={(e) => setEntry((s) => ({ ...s, waistCm: e.target.value }))}
              />
            </div>
            <div>
              <label className="label">Hips (cm)</label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={entry.hipsCm}
                onChange={(e) => setEntry((s) => ({ ...s, hipsCm: e.target.value }))}
              />
            </div>
            <div className="col-span-2">
              <label className="label">Notes</label>
              <input
                className="input"
                value={entry.notes}
                onChange={(e) => setEntry((s) => ({ ...s, notes: e.target.value }))}
              />
            </div>
            <div className="col-span-2 md:col-span-3 lg:col-span-4 flex justify-end gap-2">
              <button type="button" onClick={() => setShowForm(false)} className="btn-ghost">
                Cancel
              </button>
              <button type="submit" disabled={busy} className="btn-primary">
                {busy ? 'Saving…' : 'Save Entry'}
              </button>
            </div>
          </form>
        </div>
      )}
      <div className="card p-0 overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="text-slate-700 font-bold flex items-center gap-2">
            <LineChart className="h-4 w-4 text-slate-600" /> Progress
          </div>
          <div className="text-slate-500 text-xs">{loading ? 'Loading…' : `${filtered.length} shown`}</div>
        </div>

        {error ? <div className="p-4 text-red-600 text-sm">{error}</div> : null}
        {loading ? <div className="p-4 text-slate-500">Loading…</div> : null}

        {!loading && !error ? (
          <div className="overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Date</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Weight (kg)</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Body Fat %</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Muscle (kg)</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Notes</th>
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
                      {r.recordDate ? new Date(r.recordDate).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">{r.weightKg ?? '—'}</td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">{r.bodyFatPercentage ?? '—'}</td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">{r.muscleMassKg ?? '—'}</td>
                    <td className="px-4 py-3 text-slate-600">{r.notes || '—'}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 ? <div className="p-4 text-slate-500">No progress records.</div> : null}
          </div>
        ) : null}
      </div>
    </PageShell>
  );
}
