import React, { useEffect, useMemo, useState } from 'react';
import { CreditCard, FilePlus2, RefreshCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import PageShell from '../components/PageShell';
import api from '../services/api';

function currency(n) {
  if (n === null || n === undefined) return '—';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
}

export default function AdminBillingPage() {
  const [users, setUsers] = useState([]);
  const [plans, setPlans] = useState([]);
  const [invoices, setInvoices] = useState([]);
  const [payments, setPayments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [invoiceForm, setInvoiceForm] = useState({
    userId: '',
    membershipPlanId: '',
    amount: 0,
    taxAmount: 0,
    discountAmount: 0,
    issueDate: new Date().toISOString().slice(0, 10),
    dueDate: new Date().toISOString().slice(0, 10),
    notes: '',
  });

  const [paymentForm, setPaymentForm] = useState({
    invoiceId: '',
    userId: '',
    amount: 0,
    paymentMethod: 'cash',
    transactionId: '',
    status: 'completed',
    notes: '',
  });

  function loadAll() {
    setLoading(true);
    setError('');
    Promise.all([
      api.get('/api/users'),
      api.get('/api/membership-plans'),
      api.get('/api/admin/crud/invoices'),
      api.get('/api/admin/crud/payments'),
    ])
      .then(([u, p, i, pay]) => {
        setUsers(u.data || []);
        setPlans(p.data || []);
        setInvoices(i.data || []);
        setPayments(pay.data || []);
      })
      .catch((e) => setError(e?.response?.data?.error || e.message || 'Failed to load data'))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    loadAll();
  }, []);

  const invoiceTotal = useMemo(() => {
    const amount = Number(invoiceForm.amount || 0);
    const tax = Number(invoiceForm.taxAmount || 0);
    const discount = Number(invoiceForm.discountAmount || 0);
    return Math.max(0, amount + tax - discount);
  }, [invoiceForm]);

  async function createInvoice(e) {
    e.preventDefault();
    setError('');
    try {
      const payload = {
        user: invoiceForm.userId,
        membershipPlan: invoiceForm.membershipPlanId,
        amount: Number(invoiceForm.amount || 0),
        taxAmount: Number(invoiceForm.taxAmount || 0),
        discountAmount: Number(invoiceForm.discountAmount || 0),
        totalAmount: invoiceTotal,
        issueDate: new Date(invoiceForm.issueDate),
        dueDate: new Date(invoiceForm.dueDate),
        notes: invoiceForm.notes,
      };
      await api.post('/api/admin/crud/invoices', payload);
      setInvoiceForm({
        userId: '',
        membershipPlanId: '',
        amount: 0,
        taxAmount: 0,
        discountAmount: 0,
        issueDate: new Date().toISOString().slice(0, 10),
        dueDate: new Date().toISOString().slice(0, 10),
        notes: '',
      });
      loadAll();
    } catch (e2) {
      setError(e2?.response?.data?.error || e2.message || 'Failed to create invoice');
    }
  }

  async function createPayment(e) {
    e.preventDefault();
    setError('');
    try {
      const payload = {
        invoice: paymentForm.invoiceId,
        user: paymentForm.userId,
        amount: Number(paymentForm.amount || 0),
        paymentMethod: paymentForm.paymentMethod,
        transactionId: paymentForm.transactionId || undefined,
        status: paymentForm.status,
        notes: paymentForm.notes,
      };
      await api.post('/api/admin/crud/payments', payload);
      setPaymentForm({
        invoiceId: '',
        userId: '',
        amount: 0,
        paymentMethod: 'cash',
        transactionId: '',
        status: 'completed',
        notes: '',
      });
      loadAll();
    } catch (e2) {
      setError(e2?.response?.data?.error || e2.message || 'Failed to create payment');
    }
  }

  return (
    <PageShell
      title="Billing & Payments"
      subtitle="Create invoices, record payments, and audit billing activity."
      right={
        <button onClick={loadAll} className="btn-ghost" title="Refresh data">
          <RefreshCcw className={loading ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
          Refresh
        </button>
      }
    >
      {error ? (
        <div className="card border border-red-200/60 text-red-600 text-sm">{error}</div>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div className="card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} whileHover={{ y: -2 }}>
          <div className="flex items-center gap-2 text-slate-700 font-semibold mb-4">
            <FilePlus2 className="h-4 w-4" />
            Create invoice
          </div>
          <form onSubmit={createInvoice} className="space-y-3">
            <div>
              <label className="label">Member</label>
              <select
                className="input"
                value={invoiceForm.userId}
                onChange={(e) => setInvoiceForm((s) => ({ ...s, userId: e.target.value }))}
                required
              >
                <option value="">Select member</option>
                {users.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name} ({u.email})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Membership plan</label>
              <select
                className="input"
                value={invoiceForm.membershipPlanId}
                onChange={(e) => setInvoiceForm((s) => ({ ...s, membershipPlanId: e.target.value }))}
                required
              >
                <option value="">Select plan</option>
                {plans.map((p) => (
                  <option key={p._id} value={p._id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="label">Amount</label>
                <input
                  type="number"
                  className="input"
                  value={invoiceForm.amount}
                  onChange={(e) => setInvoiceForm((s) => ({ ...s, amount: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Tax</label>
                <input
                  type="number"
                  className="input"
                  value={invoiceForm.taxAmount}
                  onChange={(e) => setInvoiceForm((s) => ({ ...s, taxAmount: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Discount</label>
                <input
                  type="number"
                  className="input"
                  value={invoiceForm.discountAmount}
                  onChange={(e) => setInvoiceForm((s) => ({ ...s, discountAmount: e.target.value }))}
                />
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="label">Issue date</label>
                <input
                  type="date"
                  className="input"
                  value={invoiceForm.issueDate}
                  onChange={(e) => setInvoiceForm((s) => ({ ...s, issueDate: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Due date</label>
                <input
                  type="date"
                  className="input"
                  value={invoiceForm.dueDate}
                  onChange={(e) => setInvoiceForm((s) => ({ ...s, dueDate: e.target.value }))}
                />
              </div>
            </div>
            <div>
              <label className="label">Notes</label>
              <input
                className="input"
                value={invoiceForm.notes}
                onChange={(e) => setInvoiceForm((s) => ({ ...s, notes: e.target.value }))}
              />
            </div>

            <div className="flex items-center justify-between text-sm text-slate-600">
              <div>Total: {currency(invoiceTotal)}</div>
              <button className="btn-primary" type="submit">
                Create invoice
              </button>
            </div>
          </form>
        </motion.div>

        <motion.div className="card" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.1 }} whileHover={{ y: -2 }}>
          <div className="flex items-center gap-2 text-slate-700 font-semibold mb-4">
            <CreditCard className="h-4 w-4" />
            Record payment
          </div>
          <form onSubmit={createPayment} className="space-y-3">
            <div>
              <label className="label">Invoice</label>
              <select
                className="input"
                value={paymentForm.invoiceId}
                onChange={(e) => setPaymentForm((s) => ({ ...s, invoiceId: e.target.value }))}
                required
              >
                <option value="">Select invoice</option>
                {invoices.map((inv) => (
                  <option key={inv._id} value={inv._id}>
                    {inv.invoiceNumber || inv._id} — {currency(inv.totalAmount)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Member</label>
              <select
                className="input"
                value={paymentForm.userId}
                onChange={(e) => setPaymentForm((s) => ({ ...s, userId: e.target.value }))}
                required
              >
                <option value="">Select member</option>
                {users.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="label">Amount</label>
                <input
                  type="number"
                  className="input"
                  value={paymentForm.amount}
                  onChange={(e) => setPaymentForm((s) => ({ ...s, amount: e.target.value }))}
                />
              </div>
              <div>
                <label className="label">Method</label>
                <select
                  className="input"
                  value={paymentForm.paymentMethod}
                  onChange={(e) => setPaymentForm((s) => ({ ...s, paymentMethod: e.target.value }))}
                >
                  <option value="cash">Cash</option>
                  <option value="card">Card</option>
                  <option value="transfer">Transfer</option>
                  <option value="online">Online</option>
                </select>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="label">Status</label>
                <select
                  className="input"
                  value={paymentForm.status}
                  onChange={(e) => setPaymentForm((s) => ({ ...s, status: e.target.value }))}
                >
                  <option value="completed">Completed</option>
                  <option value="pending">Pending</option>
                  <option value="failed">Failed</option>
                  <option value="refunded">Refunded</option>
                </select>
              </div>
              <div>
                <label className="label">Transaction ID</label>
                <input
                  className="input"
                  value={paymentForm.transactionId}
                  onChange={(e) => setPaymentForm((s) => ({ ...s, transactionId: e.target.value }))}
                />
              </div>
            </div>
            <div>
              <label className="label">Notes</label>
              <input
                className="input"
                value={paymentForm.notes}
                onChange={(e) => setPaymentForm((s) => ({ ...s, notes: e.target.value }))}
              />
            </div>

            <div className="flex items-center justify-end">
              <button className="btn-primary" type="submit">
                Record payment
              </button>
            </div>
          </form>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <div className="text-slate-700 font-semibold mb-3">Recent invoices</div>
          <div className="space-y-2 text-sm">
            {invoices.slice(0, 8).map((inv, i) => (
              <motion.div key={inv._id} className="flex items-center justify-between border-b border-slate-200/60 pb-2" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
                <div>
                  <div className="text-slate-800 font-medium">
                    {inv.invoiceNumber || inv._id}
                  </div>
                  <div className="text-slate-500">Status: {inv.status}</div>
                </div>
                <div className="text-slate-700 font-semibold">{currency(inv.totalAmount)}</div>
              </motion.div>
            ))}
            {invoices.length === 0 ? <div className="text-slate-500">No invoices yet.</div> : null}
          </div>
        </div>
        <div className="card">
          <div className="text-slate-700 font-semibold mb-3">Recent payments</div>
          <div className="space-y-2 text-sm">
            {payments.slice(0, 8).map((pay, i) => (
              <motion.div key={pay._id} className="flex items-center justify-between border-b border-slate-200/60 pb-2" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}>
                <div>
                  <div className="text-slate-800 font-medium">{pay.transactionId || pay._id}</div>
                  <div className="text-slate-500">{pay.paymentMethod} · {pay.status}</div>
                </div>
                <div className="text-slate-700 font-semibold">{currency(pay.amount)}</div>
              </motion.div>
            ))}
            {payments.length === 0 ? <div className="text-slate-500">No payments yet.</div> : null}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
