import React, { useEffect, useMemo, useState } from 'react';
import { Plus, RefreshCcw, Search, Trash2, ShieldCheck } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useToast } from '../components/ToastContext';
import api from '../services/api';

const emptyForm = {
  name: '',
  username: '',
  email: '',
  password: '',
  role: 'member',
  membershipPlan: '',
  status: 'active',
};

export default function MembersPage({ role }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState('');
  const [plans, setPlans] = useState([]);
  const [form, setForm] = useState(emptyForm);
  const [error, setError] = useState('');
  const isAdmin = role === 'admin';
  const toast = useToast();

  const loadUsers = () => {
    setLoading(true);
    setError('');
    api
      .get('/api/users')
      .then((res) => setUsers(res.data))
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load users'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadUsers();
    if (isAdmin) {
      api
        .get('/api/membership-plans')
        .then((res) => setPlans(res.data || []))
        .catch(() => null);
    }
  }, [isAdmin]);

  const filtered = useMemo(() => {
    const query = q.trim().toLowerCase();
    if (!query) return users;
    return users.filter((u) => {
      const hay = `${u.name} ${u.email} ${u.username} ${u.plan_name} ${u.status}`.toLowerCase();
      return hay.includes(query);
    });
  }, [users, q]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 sm:gap-4">
        <div className="min-w-0">
          <h1 className="text-slate-900 text-2xl sm:text-3xl font-extrabold">Members</h1>
          <p className="text-slate-500 text-sm sm:text-base">Search, filter, and review member profiles.</p>
        </div>

        <div className="w-full sm:w-96 relative shrink-0">
          <Search className="h-4 w-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            className="input pl-9"
            placeholder="Search name, email, plan…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>

      {error ? <div className="card text-red-600 border border-red-200/60">{error}</div> : null}

      {isAdmin ? (
        <motion.div className="card" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}>
          <div className="flex items-center justify-between mb-4">
            <div className="text-slate-700 font-semibold flex items-center gap-2">
              <Plus className="h-4 w-4" />
              Add customer
            </div>
            <button onClick={loadUsers} className="btn-ghost">
              <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
              Refresh
            </button>
          </div>
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              setError('');
              try {
                await api.post('/api/admin/crud/users', {
                  name: form.name,
                  username: form.username,
                  email: form.email,
                  password: form.password,
                  role: form.role,
                  membershipPlan: form.membershipPlan || null,
                  status: form.status,
                });
                setForm(emptyForm);
                loadUsers();
              } catch (err) {
                setError(err?.response?.data?.error || err.message || 'Failed to create user');
              }
            }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
          >
            <div>
              <label className="label">Name</label>
              <input
                className="input"
                value={form.name}
                onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))}
                required
              />
            </div>
            <div>
              <label className="label">Username</label>
              <input
                className="input"
                value={form.username}
                onChange={(e) => setForm((s) => ({ ...s, username: e.target.value }))}
                required
              />
            </div>
            <div>
              <label className="label">Email</label>
              <input
                type="email"
                className="input"
                value={form.email}
                onChange={(e) => setForm((s) => ({ ...s, email: e.target.value }))}
                required
              />
            </div>
            <div>
              <label className="label">Password</label>
              <input
                type="password"
                className="input"
                value={form.password}
                onChange={(e) => setForm((s) => ({ ...s, password: e.target.value }))}
                required
              />
            </div>
            <div>
              <label className="label">Role</label>
              <select
                className="input"
                value={form.role}
                onChange={(e) => setForm((s) => ({ ...s, role: e.target.value }))}
              >
                <option value="member">Member</option>
                <option value="trainer">Trainer</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <div>
              <label className="label">Plan</label>
              <select
                className="input"
                value={form.membershipPlan}
                onChange={(e) => setForm((s) => ({ ...s, membershipPlan: e.target.value }))}
              >
                <option value="">Select plan</option>
                {plans.map((p) => (
                  <option key={p._id} value={p._id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Status</label>
              <select
                className="input"
                value={form.status}
                onChange={(e) => setForm((s) => ({ ...s, status: e.target.value }))}
              >
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="suspended">Suspended</option>
              </select>
            </div>
            <div className="md:col-span-2 lg:col-span-3 flex justify-end">
              <button className="btn-primary" type="submit">
                Create member
              </button>
            </div>
          </form>
        </motion.div>
      ) : null}

      {loading ? (
        <div className="card text-slate-500">Loading members…</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((user, i) => (
            <motion.div
              key={user.id}
              className="card hover-lift"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              whileHover={{ y: -2 }}
            >
              <div className="flex items-center justify-between">
                <div className="text-slate-900 font-semibold">{user.name}</div>
                <span
                  className={
                    user.status === 'active'
                      ? 'text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : 'text-xs px-2 py-1 rounded-full bg-rose-50 text-rose-700 border border-rose-200'
                  }
                >
                  {user.status}
                </span>
              </div>
              <div className="text-slate-500 text-sm mt-1">{user.email}</div>
              <div className="mt-3 flex flex-wrap gap-2">
                <span className="text-xs px-2 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                  {user.plan_name || '—'}
                </span>
                <span className="text-xs px-2 py-1 rounded-full bg-white text-slate-600 border border-slate-200">
                  @{user.username}
                </span>
                <span
                  className={`text-xs px-2 py-1 rounded-full border ${
                    user.role === 'admin'
                      ? 'bg-violet-50 text-violet-700 border-violet-200'
                      : user.role === 'trainer'
                        ? 'bg-amber-50 text-amber-700 border-amber-200'
                        : 'bg-sky-50 text-sky-700 border-sky-200'
                  }`}
                >
                  {user.role}
                </span>
              </div>
              <div className="mt-4 text-slate-500 text-xs">
                Goal: <span className="text-slate-700">{user.fitness_goals || '—'}</span>
              </div>
              {isAdmin ? (
                <div className="mt-4 flex justify-end gap-2">
                  {user.role !== 'admin' && (
                    <button
                      className="btn-ghost text-indigo-600 border-indigo-200/60 hover:text-indigo-700 text-xs"
                      onClick={async () => {
                        const newRole = user.role === 'trainer' ? 'member' : 'trainer';
                        try {
                          await api.patch(`/api/admin/crud/users/${user.id}`, { role: newRole });
                          toast(`${user.name} is now a ${newRole}`, 'success');
                          loadUsers();
                        } catch (e) {
                          toast(e?.response?.data?.error || 'Failed to update role', 'error');
                        }
                      }}
                    >
                      <ShieldCheck className="h-4 w-4" />
                      {user.role === 'trainer' ? 'Demote' : 'Make Trainer'}
                    </button>
                  )}
                  <button
                    className="btn-ghost text-rose-600 border-rose-200/60 hover:text-rose-700"
                    onClick={async () => {
                      setError('');
                      try {
                        await api.delete(`/api/admin/crud/users/${user.id}`);
                        loadUsers();
                      } catch (e) {
                        setError(e?.response?.data?.error || e.message || 'Failed to delete user');
                      }
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </button>
                </div>
              ) : null}
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
