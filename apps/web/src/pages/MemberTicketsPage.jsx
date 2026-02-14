import React, { useEffect, useMemo, useState } from 'react';
import { HelpCircle, RefreshCcw, Search, ChevronDown, ChevronUp, Send } from 'lucide-react';
import PageShell from '../components/PageShell';
import { useToast } from '../components/ToastContext';
import api from '../services/api';
import { motion } from 'framer-motion';

export default function MemberTicketsPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');
  const [categories, setCategories] = useState([]);
  const [expandedTicket, setExpandedTicket] = useState(null);
  const [replies, setReplies] = useState([]);
  const [replyText, setReplyText] = useState('');
  const [replyBusy, setReplyBusy] = useState(false);
  const toast = useToast();
  const [form, setForm] = useState({
    title: '',
    message: '',
    categoryId: '',
    priority: 'medium',
  });

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/me/tickets')
      .then((res) => setRows(res.data || []))
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load tickets');
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
    api
      .get('/api/me/support-categories')
      .then((res) => setCategories(res.data || []))
      .catch(() => null);
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return rows;
    return rows.filter((r) =>
      [r.title, r.message, r.status, r.priority, r.category_name, r.createdAt]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(s)),
    );
  }, [rows, q]);

  return (
    <PageShell
      title="Support"
      subtitle="Your support tickets."
      right={
        <>
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
      <div className="card">
        <div className="text-slate-800 font-semibold mb-3">Create a ticket</div>
        <form
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
          onSubmit={async (e) => {
            e.preventDefault();
            setError(null);
            try {
              await api.post('/api/me/tickets', {
                title: form.title,
                message: form.message,
                categoryId: form.categoryId || undefined,
                priority: form.priority,
              });
              setForm({ title: '', message: '', categoryId: '', priority: 'medium' });
              load();
            } catch (e2) {
              setError(e2?.response?.data?.error || e2.message || 'Failed to create ticket');
            }
          }}
        >
          <div>
            <label className="label">Title</label>
            <input
              className="input"
              value={form.title}
              onChange={(e) => setForm((s) => ({ ...s, title: e.target.value }))}
              required
            />
          </div>
          <div>
            <label className="label">Category</label>
            <select
              className="input"
              value={form.categoryId}
              onChange={(e) => setForm((s) => ({ ...s, categoryId: e.target.value }))}
            >
              <option value="">Select category</option>
              {categories.map((c) => (
                <option key={c._id} value={c._id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          <div className="md:col-span-2">
            <label className="label">Message</label>
            <textarea
              className="input"
              rows={4}
              value={form.message}
              onChange={(e) => setForm((s) => ({ ...s, message: e.target.value }))}
              required
            />
          </div>
          <div>
            <label className="label">Priority</label>
            <select
              className="input"
              value={form.priority}
              onChange={(e) => setForm((s) => ({ ...s, priority: e.target.value }))}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>
          </div>
          <div className="md:col-span-2 flex justify-end">
            <button type="submit" className="btn-primary">
              Submit ticket
            </button>
          </div>
        </form>
      </div>

      <div className="card p-0 overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="text-slate-700 font-bold flex items-center gap-2">
            <HelpCircle className="h-4 w-4 text-slate-600" /> Tickets
          </div>
          <div className="text-slate-500 text-xs">{loading ? 'Loading…' : `${filtered.length} shown`}</div>
        </div>

        {error ? <div className="p-4 text-red-600 text-sm">{error}</div> : null}
        {loading ? <div className="p-4 text-slate-500">Loading…</div> : null}

        {!loading && !error ? (
          <div className="grid gap-3 p-4">
            {filtered.map((t, i) => {
              const tid = t._id || t.id;
              const isExpanded = expandedTicket === tid;
              return (
                <motion.div
                  key={tid}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                  whileHover={{ y: -2 }}
                  className="rounded-2xl border border-slate-200 bg-white overflow-hidden"
                >
                  <button
                    onClick={() => {
                      if (isExpanded) {
                        setExpandedTicket(null);
                      } else {
                        setExpandedTicket(tid);
                        api
                          .get(`/api/me/tickets/${tid}/replies`)
                          .then((r) => setReplies(r.data || []))
                          .catch(() => setReplies([]));
                      }
                    }}
                    className="w-full p-4 hover:bg-slate-50 transition-colors text-left"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="text-slate-900 font-bold">{t.title || `Ticket #${tid}`}</div>
                        <div className="text-slate-500 text-sm mt-1">{t.message || '—'}</div>
                        <div className="text-slate-400 text-xs mt-3">
                          {t.category_name ? `${t.category_name} • ` : ''}
                          {t.priority ? `Priority: ${t.priority} • ` : ''}
                          {t.createdAt ? `Created: ${new Date(t.createdAt).toLocaleDateString()}` : ''}
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <span
                          className={`inline-flex px-2 py-1 rounded-lg text-xs border ${
                            t.status === 'resolved'
                              ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                              : t.status === 'in_progress'
                                ? 'bg-amber-50 border-amber-200 text-amber-700'
                                : 'bg-slate-50 border-slate-200 text-slate-700'
                          }`}
                        >
                          {t.status || 'open'}
                        </span>
                        {isExpanded ? (
                          <ChevronUp className="h-4 w-4 text-slate-400" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-slate-400" />
                        )}
                      </div>
                    </div>
                  </button>

                  {isExpanded && (
                    <div className="border-t border-slate-100 p-4 bg-slate-50/50">
                      <div className="text-slate-700 font-semibold text-sm mb-2">Replies</div>
                      {replies.length === 0 && <div className="text-slate-500 text-sm">No replies yet.</div>}
                      <div className="space-y-2 mb-3">
                        {replies.map((r) => (
                          <div key={r._id} className="rounded-lg bg-white border border-slate-200 p-3">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-slate-800">{r.user?.name || 'Staff'}</span>
                              <span className="text-xs text-slate-400">
                                {r.createdAt ? new Date(r.createdAt).toLocaleString() : ''}
                              </span>
                            </div>
                            <div className="text-sm text-slate-600 mt-1">{r.message}</div>
                          </div>
                        ))}
                      </div>
                      <form
                        className="flex gap-2"
                        onSubmit={async (e) => {
                          e.preventDefault();
                          if (!replyText.trim()) return;
                          setReplyBusy(true);
                          try {
                            await api.post(`/api/me/tickets/${tid}/replies`, { message: replyText });
                            setReplyText('');
                            toast('Reply sent!', 'success');
                            const r = await api.get(`/api/me/tickets/${tid}/replies`);
                            setReplies(r.data || []);
                          } catch (e2) {
                            toast(e2?.response?.data?.error || 'Failed to send reply', 'error');
                          } finally {
                            setReplyBusy(false);
                          }
                        }}
                      >
                        <input
                          className="input flex-1"
                          placeholder="Type a reply…"
                          value={replyText}
                          onChange={(e) => setReplyText(e.target.value)}
                        />
                        <button type="submit" disabled={replyBusy} className="btn-primary">
                          <Send className="h-4 w-4" />
                        </button>
                      </form>
                    </div>
                  )}
                </motion.div>
              );
            })}
            {filtered.length === 0 ? <div className="text-slate-500">No tickets found.</div> : null}
          </div>
        ) : null}
      </div>
    </PageShell>
  );
}
