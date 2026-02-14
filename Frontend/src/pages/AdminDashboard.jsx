import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Users, Activity, DollarSign, TrendingUp, ArrowRight, Database, CreditCard, LifeBuoy, Zap } from 'lucide-react';
import StatCard from '../components/StatCard';
import PageShell from '../components/PageShell';
import api from '../services/api';

const quickActions = [
  { label: 'Members', desc: 'View & manage', icon: Users, gradient: 'from-blue-500 to-cyan-500', tab: 'members' },
  { label: 'Billing', desc: 'Invoices & payments', icon: CreditCard, gradient: 'from-emerald-500 to-teal-500', tab: 'billing' },
  { label: 'Support', desc: 'Tickets & issues', icon: LifeBuoy, gradient: 'from-violet-500 to-purple-500', tab: 'support' },
  { label: 'Database', desc: 'Explore tables', icon: Database, gradient: 'from-amber-500 to-orange-500', tab: 'db' },
];

export default function AdminDashboard({ onTabChange }) {
  const [stats, setStats] = useState({});
  const [overview, setOverview] = useState({});

  useEffect(() => {
    let mounted = true;
    api
      .get('/api/stats')
      .then((res) => { if (mounted) setStats(res.data); })
      .catch((e) => console.error(e));
    api
      .get('/api/admin/analytics/overview')
      .then((res) => { if (mounted) setOverview(res.data || {}); })
      .catch(() => null);
    return () => { mounted = false; };
  }, []);

  return (
    <PageShell
      title="Admin Dashboard"
      subtitle="Command center for memberships, revenue, and operations."
      right={
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="hidden sm:flex items-center gap-2 text-sm"
        >
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
          </span>
          <span className="text-slate-500">Live data from MongoDB</span>
        </motion.div>
      }
    >
      {/* Hero card with animated gradient */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="card-hero relative overflow-hidden text-white shine"
        style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)' }}
      >
        {/* Animated mesh gradient overlay */}
        <div className="absolute inset-0 opacity-30 animate-gradient-x" style={{ background: 'linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent, rgba(14,165,233,0.3), transparent)', backgroundSize: '300% 100%' }} />
        {/* Decorative circles */}
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-indigo-500/10 blur-2xl" />
        <div className="absolute -left-10 -bottom-10 h-48 w-48 rounded-full bg-cyan-500/10 blur-2xl" />

        <div className="relative flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div>
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-1.5 text-xs text-white/60 bg-white/10 rounded-full px-3 py-1 border border-white/10"
            >
              <Zap className="h-3 w-3" /> Command Center
            </motion.div>
            <div className="text-2xl sm:text-3xl font-bold mt-3">Welcome back, Admin.</div>
            <div className="text-white/60 mt-2 max-w-xl">
              Track member growth, revenue health, and studio activity from a single view.
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Active members', value: stats.activeUsers || 0 },
              { label: 'Revenue', value: `$${(overview.totalRevenue || 0).toFixed(0)}` },
              { label: 'Visits', value: overview.totalVisits || 0 },
            ].map((item, i) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + i * 0.1 }}
                className="rounded-xl bg-white/[0.08] border border-white/[0.12] px-4 py-3 backdrop-blur-sm hover:bg-white/[0.12] transition-colors"
              >
                <div className="text-xs text-white/50">{item.label}</div>
                <div className="text-xl font-bold mt-0.5">{item.value}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Members" value={stats.totalUsers || 0} Icon={Users} colorClass="text-slate-700" />
        <StatCard title="Active Members" value={stats.activeUsers || 0} Icon={Activity} colorClass="text-emerald-600" />
        <StatCard title="Membership Plans" value={stats.totalPlans || 0} Icon={DollarSign} colorClass="text-indigo-600" />
        <StatCard title="Total Visits" value={overview.totalVisits || 0} Icon={TrendingUp} colorClass="text-amber-600" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <StatCard title="Total Revenue" value={`$${(overview.totalRevenue || 0).toFixed(2)}`} Icon={DollarSign} colorClass="text-emerald-600" />
        <div className="card">
          <h2 className="text-slate-800 text-lg font-bold mb-4">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-3">
            {quickActions.map((action, i) => (
              <motion.div
                key={action.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * i }}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onTabChange?.(action.tab)}
                className="group relative overflow-hidden rounded-xl border border-slate-200 bg-white p-4 cursor-pointer hover:shadow-lg hover:border-slate-300 transition-all duration-300"
              >
                <div className={`h-9 w-9 rounded-lg bg-gradient-to-br ${action.gradient} flex items-center justify-center mb-3 shadow-sm`}>
                  <action.icon className="h-4.5 w-4.5 text-white" />
                </div>
                <div className="text-slate-800 font-semibold text-sm">{action.label}</div>
                <div className="text-slate-400 text-xs mt-0.5">{action.desc}</div>
                <ArrowRight className="absolute bottom-4 right-4 h-4 w-4 text-slate-300 group-hover:text-slate-500 group-hover:translate-x-0.5 transition-all" />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
