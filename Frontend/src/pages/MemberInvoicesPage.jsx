import React, { useEffect, useMemo, useState } from 'react';
import { CreditCard, RefreshCcw, Search } from 'lucide-react';
import PageShell from '../components/PageShell';
import api from '../services/api';
import { motion } from 'framer-motion';

function currency(n) {
  const v = Number(n);
  if (Number.isNaN(v)) return '—';
  return `$${v.toFixed(2)}`;
}

export default function MemberInvoicesPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [q, setQ] = useState('');

  function load() {
    setLoading(true);
    setError(null);
    api
      .get('/api/me/invoices')
      .then((res) => setRows(res.data || []))
      .catch((e) => {
        console.error(e);
        setRows([]);
        setError(e?.response?.data?.error || e.message || 'Failed to load invoices');
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
      [r.invoiceNumber, r.status, r.issueDate, r.dueDate, r.totalAmount]
        .filter(Boolean)
        .some((v) => String(v).toLowerCase().includes(s))
    );
  }, [rows, q]);

  return (
    <PageShell
      title="Invoices"
      subtitle="Your membership invoices (latest 50)."
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
      <div className="card p-0 overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="text-slate-700 font-bold flex items-center gap-2">
            <CreditCard className="h-4 w-4 text-slate-600" /> Invoices
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
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Invoice</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Issue</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Due</th>
                  <th className="text-left text-slate-600 font-semibold px-4 py-3">Total</th>
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
                      <div className="font-semibold text-slate-900">
                        {r.invoiceNumber || `INV-${r._id || r.id}`}
                      </div>
                      <div className="text-slate-400 text-xs">ID: {r._id || r.id}</div>
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.issueDate ? new Date(r.issueDate).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {r.dueDate ? new Date(r.dueDate).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      {currency(r.totalAmount)}
                    </td>
                    <td className="px-4 py-3 text-slate-700 whitespace-nowrap">
                      <span className="inline-flex px-2 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs">
                        {r.status || 'unpaid'}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 ? <div className="p-4 text-slate-500">No invoices found.</div> : null}
          </div>
        ) : null}
      </div>
    </PageShell>
  );
}
