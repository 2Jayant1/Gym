import React, { useEffect, useState } from 'react';
import { RefreshCcw, ShieldCheck } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import PageShell from '../components/PageShell';
import api from '../services/api';

const statuses = ['open', 'in_progress', 'resolved', 'closed'];
const priorities = ['low', 'medium', 'high', 'urgent'];

export default function AdminSupportPage() {
  const [tickets, setTickets] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = () => {
    setLoading(true);
    setError('');
    Promise.all([
      api.get('/api/admin/crud/support_tickets'),
      api.get('/api/admin/crud/support_categories'),
    ])
      .then(([t, c]) => {
        setTickets(t.data || []);
        setCategories(c.data || []);
      })
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load support data'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const categoryName = (id) => categories.find((c) => c._id === id)?.name || '—';

  return (
    <PageShell
      title="Support Center"
      subtitle="Review, prioritize, and resolve member tickets."
      right={
        <button onClick={load} className="btn-ghost">
          <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
          Refresh
        </button>
      }
    >
      {error ? <div className="card border border-red-200/60 text-red-600">{error}</div> : null}

      <div className="space-y-4">
        {tickets.map((t, i) => (
          <motion.div key={t._id || t.id} className="card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }} whileHover={{ y: -2 }}>
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="text-slate-900 font-semibold text-base sm:text-lg">{t.title}</div>
                <div className="text-slate-500 text-sm mt-1">{t.message}</div>
                <div className="text-slate-400 text-xs mt-2">
                  Category: {categoryName(t.category)} · Priority: {t.priority}
                </div>
              </div>
              <div className="grid grid-cols-2 sm:flex sm:flex-col gap-2 sm:min-w-[180px]">
                <label className="label">Status</label>
                <select
                  className="input"
                  value={t.status}
                  onChange={async (e) => {
                    try {
                      await api.patch(`/api/admin/crud/support_tickets/${t._id || t.id}`, {
                        status: e.target.value,
                      });
                      load();
                    } catch (e2) {
                      setError(e2?.response?.data?.error || e2.message || 'Failed to update ticket');
                    }
                  }}
                >
                  {statuses.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
                <label className="label">Priority</label>
                <select
                  className="input"
                  value={t.priority}
                  onChange={async (e) => {
                    try {
                      await api.patch(`/api/admin/crud/support_tickets/${t._id || t.id}`, {
                        priority: e.target.value,
                      });
                      load();
                    } catch (e2) {
                      setError(e2?.response?.data?.error || e2.message || 'Failed to update ticket');
                    }
                  }}
                >
                  {priorities.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="mt-3 text-slate-500 text-xs flex items-center gap-2">
              <ShieldCheck className="h-4 w-4" /> Ticket ID: {t._id || t.id}
            </div>
          </motion.div>
        ))}
        {!loading && tickets.length === 0 ? <div className="card text-slate-500">No tickets yet.</div> : null}
      </div>
    </PageShell>
  );
}
