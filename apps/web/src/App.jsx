import React, { useState, useEffect } from 'react';
import TopNav from './components/TopNav';
import { ToastProvider } from './components/ToastContext';
import LoginPage from './pages/LoginPage';
import AdminDashboard from './pages/AdminDashboard';
import AdminDatabaseExplorer from './pages/AdminDatabaseExplorer';
import AdminPlansPage from './pages/AdminPlansPage';
import AdminAnalyticsPage from './pages/AdminAnalyticsPage';
import AdminSupportPage from './pages/AdminSupportPage';
import AdminBillingPage from './pages/AdminBillingPage';
import TrainerDashboard from './pages/TrainerDashboard';
import MemberDashboard from './pages/MemberDashboard';
import MembersPage from './pages/MembersPage';
import TrainerSessionsPage from './pages/TrainerSessionsPage';
import TrainerClassesPage from './pages/TrainerClassesPage';
import MemberAttendancePage from './pages/MemberAttendancePage';
import MemberInvoicesPage from './pages/MemberInvoicesPage';
import MemberProgressPage from './pages/MemberProgressPage';
import MemberTicketsPage from './pages/MemberTicketsPage';
import MemberClassesPage from './pages/MemberClassesPage';
import MemberWorkoutPlansPage from './pages/MemberWorkoutPlansPage';
import AIInsightsPage from './pages/AIInsightsPage';
import AmbientEffects from './components/AmbientEffects';
import FitBuddy from './components/FitBuddy';
import { clearAuth, loadAuth, saveAuth } from './auth/authStorage';
import api, { setAuthToken, setOnUnauthorized } from './services/api';
import { setMlOnUnauthorized } from './services/mlApi';

const roleConfig = {
  admin: {
    tabs: [
      { key: 'dashboard', label: 'Dashboard' },
      { key: 'db', label: 'Database' },
      { key: 'members', label: 'Members' },
      { key: 'billing', label: 'Billing' },
      { key: 'support', label: 'Support' },
      { key: 'plans', label: 'Plans' },
      { key: 'analytics', label: 'Analytics' },
      { key: 'ai', label: 'AI Insights' },
    ],
  },
  trainer: {
    tabs: [
      { key: 'dashboard', label: 'Today' },
      { key: 'sessions', label: 'Sessions' },
      { key: 'classes', label: 'Classes' },
      { key: 'members', label: 'Members' },
      { key: 'ai', label: 'AI Insights' },
    ],
  },
  member: {
    tabs: [
      { key: 'dashboard', label: 'Overview' },
      { key: 'attendance', label: 'Attendance' },
      { key: 'classes', label: 'Classes' },
      { key: 'workouts', label: 'Workouts' },
      { key: 'invoices', label: 'Invoices' },
      { key: 'progress', label: 'Progress' },
      { key: 'tickets', label: 'Support' },
      { key: 'ai', label: 'AI Insights' },
    ],
  },
};

function PortalContent({ role, tab, onTabChange }) {
  if (role === 'admin') {
    if (tab === 'dashboard') return <AdminDashboard onTabChange={onTabChange} />;
    if (tab === 'db') return <AdminDatabaseExplorer />;
    if (tab === 'members') return <MembersPage role="admin" />;
    if (tab === 'billing') return <AdminBillingPage />;
    if (tab === 'support') return <AdminSupportPage />;
    if (tab === 'plans') return <AdminPlansPage />;
    if (tab === 'analytics') return <AdminAnalyticsPage />;
    if (tab === 'ai') return <AIInsightsPage />;
  }

  if (role === 'trainer') {
    if (tab === 'dashboard') return <TrainerDashboard onTabChange={onTabChange} />;
    if (tab === 'sessions') return <TrainerSessionsPage />;
    if (tab === 'classes') return <TrainerClassesPage />;
    if (tab === 'members') return <MembersPage role="trainer" />;
    if (tab === 'ai') return <AIInsightsPage />;
  }

  if (tab === 'dashboard') return <MemberDashboard onTabChange={onTabChange} />;
  if (tab === 'attendance') return <MemberAttendancePage />;
  if (tab === 'classes') return <MemberClassesPage />;
  if (tab === 'workouts') return <MemberWorkoutPlansPage />;
  if (tab === 'invoices') return <MemberInvoicesPage />;
  if (tab === 'progress') return <MemberProgressPage />;
  if (tab === 'tickets') return <MemberTicketsPage />;
  if (tab === 'ai') return <AIInsightsPage />;

  return <MemberDashboard onTabChange={onTabChange} />;
}

export default function App() {
  const [auth, setAuth] = useState(() => loadAuth());
  const user = auth?.user || null;
  const role = user?.role || 'member';
  const tabs = roleConfig[role]?.tabs || roleConfig.member.tabs;

  const defaultTab = tabs[0]?.key || 'dashboard';
  const [activeTab, setActiveTab] = useState(defaultTab);

  useEffect(() => {
    const allowed = new Set(tabs.map((t) => t.key));
    if (!allowed.has(activeTab)) setActiveTab(defaultTab);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [role]);

  useEffect(() => {
    setAuthToken(auth?.token);
  }, [auth?.token]);

  // Wire 401 auto-logout
  useEffect(() => {
    const logout = () => {
      clearAuth();
      setAuth(null);
      setAuthToken(null);
    };

    setOnUnauthorized(logout);
    setMlOnUnauthorized(logout);
  }, []);

  useEffect(() => {
    if (!auth?.token) return;
    api.get('/api/me').catch(() => {
      clearAuth();
      setAuth(null);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLogout = () => {
    clearAuth();
    setAuth(null);
    setAuthToken(null);
  };

  if (!auth?.token) {
    return (
      <ToastProvider>
        <LoginPage
          onLogin={(data) => {
            saveAuth(data);
            setAuth(data);
            setAuthToken(data?.token);
            const r = data?.user?.role || 'member';
            const t = roleConfig[r]?.tabs?.[0]?.key || 'dashboard';
            setActiveTab(t);
          }}
        />
      </ToastProvider>
    );
  }

  return (
    <ToastProvider>
      <div className="min-h-screen app-bg relative">
        <AmbientEffects />
        <TopNav
          role={role}
          name={user?.name || user?.username || 'User'}
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onLogout={handleLogout}
        />

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 sm:py-8 relative z-10">
          <PortalContent role={role} tab={activeTab} onTabChange={setActiveTab} />
        </main>
        <FitBuddy activeTab={activeTab} role={role} />
      </div>
    </ToastProvider>
  );
}
