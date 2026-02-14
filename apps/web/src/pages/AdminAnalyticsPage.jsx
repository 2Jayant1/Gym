import React, { useEffect, useState } from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import PageShell from '../components/PageShell';
import StatCard from '../components/StatCard';
import api from '../services/api';

export default function AdminAnalyticsPage() {
  const [overview, setOverview] = useState({});
  const [details, setDetails] = useState({
    usersByRole: [],
    revenueByStatus: [],
    paymentsByMethod: [],
    monthlyRevenue: [],
    planDistribution: [],
  });
  const [error, setError] = useState('');

  useEffect(() => {
    api
      .get('/api/admin/analytics/overview')
      .then((res) => setOverview(res.data || {}))
      .catch(() => null);
    api
      .get('/api/admin/analytics/details')
      .then((res) => setDetails(res.data || {}))
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load analytics'));
  }, []);

  return (
    <PageShell title="Analytics" subtitle="Operational insights and revenue tracking.">
      {error ? <div className="card border border-red-200/60 text-red-600">{error}</div> : null}

      <motion.div
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <StatCard title="Total Users" value={overview.totalUsers || 0} Icon={BarChart3} />
        <StatCard
          title="Active Users"
          value={overview.activeUsers || 0}
          Icon={TrendingUp}
          colorClass="text-emerald-600"
        />
        <StatCard
          title="Total Visits"
          value={overview.totalVisits || 0}
          Icon={TrendingUp}
          colorClass="text-amber-600"
        />
        <StatCard
          title="Revenue"
          value={`$${(overview.totalRevenue || 0).toFixed(2)}`}
          Icon={BarChart3}
          colorClass="text-indigo-600"
        />
      </motion.div>

      <motion.div
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <div className="card">
          <div className="text-slate-800 font-semibold mb-3">Users by role</div>
          <div className="space-y-2 text-sm">
            {details.usersByRole?.map((r) => (
              <div key={r.role} className="flex items-center justify-between">
                <div className="text-slate-600">{r.role}</div>
                <div className="text-slate-900 font-semibold">{r.count}</div>
              </div>
            ))}
            {details.usersByRole?.length === 0 ? <div className="text-slate-500">No data yet.</div> : null}
          </div>
        </div>

        <div className="card">
          <div className="text-slate-800 font-semibold mb-3">Plan distribution</div>
          <div className="space-y-2 text-sm">
            {details.planDistribution?.map((p) => (
              <div key={p.name} className="flex items-center justify-between">
                <div className="text-slate-600">{p.name || 'Unassigned'}</div>
                <div className="text-slate-900 font-semibold">{p.count}</div>
              </div>
            ))}
            {details.planDistribution?.length === 0 ? <div className="text-slate-500">No data yet.</div> : null}
          </div>
        </div>

        <div className="card">
          <div className="text-slate-800 font-semibold mb-3">Revenue by invoice status</div>
          <div className="space-y-2 text-sm">
            {details.revenueByStatus?.map((r) => (
              <div key={r.status} className="flex items-center justify-between">
                <div className="text-slate-600">{r.status}</div>
                <div className="text-slate-900 font-semibold">${(r.total || 0).toFixed(2)}</div>
              </div>
            ))}
            {details.revenueByStatus?.length === 0 ? <div className="text-slate-500">No data yet.</div> : null}
          </div>
        </div>

        <div className="card">
          <div className="text-slate-800 font-semibold mb-3">Payments by method</div>
          <div className="space-y-2 text-sm">
            {details.paymentsByMethod?.map((p) => (
              <div key={p.method} className="flex items-center justify-between">
                <div className="text-slate-600">{p.method}</div>
                <div className="text-slate-900 font-semibold">${(p.total || 0).toFixed(2)}</div>
              </div>
            ))}
            {details.paymentsByMethod?.length === 0 ? <div className="text-slate-500">No data yet.</div> : null}
          </div>
        </div>
      </motion.div>

      <motion.div
        className="card"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <div className="text-slate-800 font-semibold mb-3">Monthly revenue</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          {details.monthlyRevenue?.map((m) => (
            <div key={m.month} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
              <div className="text-slate-500">{m.month}</div>
              <div className="text-slate-900 font-semibold">${(m.total || 0).toFixed(2)}</div>
            </div>
          ))}
          {details.monthlyRevenue?.length === 0 ? <div className="text-slate-500">No data yet.</div> : null}
        </div>
      </motion.div>
    </PageShell>
  );
}
