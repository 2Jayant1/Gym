import React, { useEffect, useState } from 'react';
import { CheckCircle2, Edit3, RefreshCcw, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import PageShell from '../components/PageShell';
import api from '../services/api';

const emptyPlan = {
  name: '',
  description: '',
  monthlyFee: 0,
  durationDays: 30,
  maxVisitsPerWeek: '',
  includesTrainer: false,
  includesClasses: true,
  signupFee: 0,
  cancellationFee: 0,
  status: 'active',
};

export default function AdminPlansPage() {
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [form, setForm] = useState(emptyPlan);
  const [editing, setEditing] = useState(null);

  const load = () => {
    setLoading(true);
    setError('');
    api
      .get('/api/admin/crud/membership_plans')
      .then((res) => setPlans(res.data || []))
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load plans'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const payload = {
        ...form,
        monthlyFee: Number(form.monthlyFee || 0),
        durationDays: Number(form.durationDays || 0),
        maxVisitsPerWeek: form.maxVisitsPerWeek ? Number(form.maxVisitsPerWeek) : null,
        signupFee: Number(form.signupFee || 0),
        cancellationFee: Number(form.cancellationFee || 0),
      };
      if (editing) {
        await api.patch(`/api/admin/crud/membership_plans/${editing}`, payload);
      } else {
        await api.post('/api/admin/crud/membership_plans', payload);
      }
      setForm(emptyPlan);
      setEditing(null);
      load();
    } catch (e2) {
      setError(e2?.response?.data?.error || e2.message || 'Failed to save plan');
    }
  };

  return (
    <PageShell
      title="Plans"
      subtitle="Create and manage membership plans across the organization."
      right={
        <button onClick={load} className="btn-ghost">
          <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
          Refresh
        </button>
      }
    >
      {error ? <div className="card border border-red-200/60 text-red-600">{error}</div> : null}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card lg:col-span-1">
          <div className="text-slate-800 font-semibold mb-4 flex items-center gap-2">
            <Edit3 className="h-4 w-4" />
            {editing ? 'Edit plan' : 'Create plan'}
          </div>
          <form onSubmit={submit} className="space-y-3">
            <div>
              <label className="label">Name</label>
              <input className="input" value={form.name} onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))} required />
            </div>
            <div>
              <label className="label">Description</label>
              <input className="input" value={form.description} onChange={(e) => setForm((s) => ({ ...s, description: e.target.value }))} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Monthly fee</label>
                <input type="number" className="input" value={form.monthlyFee} onChange={(e) => setForm((s) => ({ ...s, monthlyFee: e.target.value }))} />
              </div>
              <div>
                <label className="label">Duration (days)</label>
                <input type="number" className="input" value={form.durationDays} onChange={(e) => setForm((s) => ({ ...s, durationDays: e.target.value }))} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Signup fee</label>
                <input type="number" className="input" value={form.signupFee} onChange={(e) => setForm((s) => ({ ...s, signupFee: e.target.value }))} />
              </div>
              <div>
                <label className="label">Cancellation fee</label>
                <input type="number" className="input" value={form.cancellationFee} onChange={(e) => setForm((s) => ({ ...s, cancellationFee: e.target.value }))} />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Max visits / week</label>
                <input className="input" value={form.maxVisitsPerWeek} onChange={(e) => setForm((s) => ({ ...s, maxVisitsPerWeek: e.target.value }))} />
              </div>
              <div>
                <label className="label">Status</label>
                <select className="input" value={form.status} onChange={(e) => setForm((s) => ({ ...s, status: e.target.value }))}>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                </select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="flex items-center gap-2 text-slate-600 text-sm">
                <input type="checkbox" checked={form.includesTrainer} onChange={(e) => setForm((s) => ({ ...s, includesTrainer: e.target.checked }))} />
                Includes trainer
              </label>
              <label className="flex items-center gap-2 text-slate-600 text-sm">
                <input type="checkbox" checked={form.includesClasses} onChange={(e) => setForm((s) => ({ ...s, includesClasses: e.target.checked }))} />
                Includes classes
              </label>
            </div>
            <div className="flex items-center justify-between">
              <button type="submit" className="btn-primary">{editing ? 'Save changes' : 'Create plan'}</button>
              {editing ? (
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => {
                    setEditing(null);
                    setForm(emptyPlan);
                  }}
                >
                  Cancel
                </button>
              ) : null}
            </div>
          </form>
        </div>

        <div className="lg:col-span-2 space-y-4">
          {plans.map((p, i) => (
            <motion.div key={p._id || p.id} className="card flex items-start justify-between gap-4" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }} whileHover={{ y: -2 }}>
              <div>
                <div className="text-slate-900 font-semibold text-lg flex items-center gap-2">
                  {p.name}
                  {p.status === 'active' ? (
                    <span className="inline-flex items-center gap-1 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-full px-2 py-1">
                      <CheckCircle2 className="h-3 w-3" /> Active
                    </span>
                  ) : (
                    <span className="text-xs text-slate-600 bg-slate-100 border border-slate-200 rounded-full px-2 py-1">Inactive</span>
                  )}
                </div>
                <div className="text-slate-500 text-sm mt-1">{p.description || '—'}</div>
                <div className="text-slate-600 text-sm mt-3">
                  ${p.monthlyFee} / {p.durationDays} days · Signup ${p.signupFee} · Cancel ${p.cancellationFee}
                </div>
                <div className="text-slate-500 text-xs mt-2">
                  {p.includesTrainer ? 'Trainer included' : 'Trainer optional'} · {p.includesClasses ? 'Classes included' : 'Classes optional'}
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <button
                  className="btn-ghost"
                  onClick={() => {
                    setEditing(p._id || p.id);
                    setForm({
                      name: p.name || '',
                      description: p.description || '',
                      monthlyFee: p.monthlyFee || 0,
                      durationDays: p.durationDays || 30,
                      maxVisitsPerWeek: p.maxVisitsPerWeek ?? '',
                      includesTrainer: !!p.includesTrainer,
                      includesClasses: !!p.includesClasses,
                      signupFee: p.signupFee || 0,
                      cancellationFee: p.cancellationFee || 0,
                      status: p.status || 'active',
                    });
                  }}
                >
                  <Edit3 className="h-4 w-4" /> Edit
                </button>
                <button
                  className="btn-ghost text-rose-600 border-rose-200/60 hover:text-rose-700"
                  onClick={async () => {
                    setError('');
                    try {
                      await api.delete(`/api/admin/crud/membership_plans/${p._id || p.id}`);
                      load();
                    } catch (e2) {
                      setError(e2?.response?.data?.error || e2.message || 'Failed to delete plan');
                    }
                  }}
                >
                  <Trash2 className="h-4 w-4" /> Delete
                </button>
              </div>
            </motion.div>
          ))}
          {!loading && plans.length === 0 ? <div className="card text-slate-500">No plans created yet.</div> : null}
        </div>
      </div>
    </PageShell>
  );
}
