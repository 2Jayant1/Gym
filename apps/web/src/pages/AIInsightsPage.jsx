import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Dumbbell,
  Flame,
  Activity,
  TrendingUp,
  Sparkles,
  Loader2,
  ChevronDown,
  AlertTriangle,
  Target,
  Zap,
  Heart,
  MessageSquare,
  BarChart3,
  Users,
  ShieldAlert,
  Lightbulb,
  BookOpen,
  Cpu,
} from 'lucide-react';
import PageShell from '../components/PageShell';
import AIChatPanel from '../components/AIChatPanel';
import { useToast } from '../components/ToastContext';
import {
  recommendWorkout,
  predictCalories,
  predictAdherence,
  predictProgress,
  mlHealth,
  getInsights,
  getClusters,
  getAnomalies,
} from '../services/mlApi';

/* ─── Tab button ─────────────────────────────────────────────── */
function Tab({ active, onClick, Icon, label, badge }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200
        ${
          active
            ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg shadow-indigo-200'
            : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
        }`}
    >
      <Icon size={16} />
      {label}
      {badge !== null && (
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded-full ${active ? 'bg-white/20' : 'bg-slate-200 text-slate-600'}`}
        >
          {badge}
        </span>
      )}
    </button>
  );
}

/* ─── Insight card ───────────────────────────────────────────── */
function InsightCard({ insight, index }) {
  const priorityColors = {
    critical: 'border-red-300 bg-red-50',
    high: 'border-orange-300 bg-orange-50',
    medium: 'border-blue-300 bg-blue-50',
    low: 'border-slate-200 bg-slate-50',
  };
  const priorityIcons = {
    critical: <ShieldAlert size={16} className="text-red-500" />,
    high: <AlertTriangle size={16} className="text-orange-500" />,
    medium: <Lightbulb size={16} className="text-blue-500" />,
    low: <BookOpen size={16} className="text-slate-400" />,
  };
  const categoryIcons = {
    workout: <Dumbbell size={14} />,
    attendance: <Activity size={14} />,
    performance: <TrendingUp size={14} />,
    anomaly: <ShieldAlert size={14} />,
    cluster: <Users size={14} />,
    model: <Cpu size={14} />,
    general: <BarChart3 size={14} />,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className={`rounded-xl border p-4 ${priorityColors[insight.priority] || priorityColors.low}`}
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5">{priorityIcons[insight.priority] || priorityIcons.low}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="font-semibold text-sm text-slate-800">{insight.title}</h4>
            <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-white/60 text-slate-500 border border-slate-200">
              {categoryIcons[insight.category] || categoryIcons.general}
              {insight.category}
            </span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">{insight.text}</p>
        </div>
      </div>
    </motion.div>
  );
}

/* ─── Cluster card ───────────────────────────────────────────── */
function ClusterCard({ cluster, index }) {
  const colors = [
    'from-indigo-500 to-purple-600',
    'from-emerald-500 to-teal-600',
    'from-orange-500 to-red-500',
    'from-blue-500 to-cyan-600',
    'from-pink-500 to-rose-600',
  ];
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.08 }}
      className="card overflow-hidden"
    >
      <div className={`bg-gradient-to-r ${colors[index % colors.length]} px-4 py-3 text-white`}>
        <div className="flex items-center justify-between">
          <h4 className="font-bold text-sm">{cluster.label}</h4>
          <span className="text-xs bg-white/20 px-2 py-0.5 rounded-full">
            {cluster.size} members ({cluster.pct}%)
          </span>
        </div>
      </div>
      <div className="p-4">
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(cluster.centroid || {})
            .slice(0, 6)
            .map(([k, v]) => (
              <div key={k} className="text-xs">
                <span className="text-slate-400">{k.replace(/_/g, ' ')}</span>
                <span className="block font-semibold text-slate-700">{typeof v === 'number' ? v.toFixed(1) : v}</span>
              </div>
            ))}
        </div>
      </div>
    </motion.div>
  );
}

/* ─── Anomaly summary ────────────────────────────────────────── */
function AnomalySummary({ data }) {
  if (!data) return null;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-50 border border-red-200">
          <ShieldAlert size={18} className="text-red-500" />
          <div>
            <div className="text-lg font-bold text-red-700">{data.total_anomalies}</div>
            <div className="text-[10px] text-red-500">Total Anomalies</div>
          </div>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-50 border border-slate-200">
          <Users size={18} className="text-slate-500" />
          <div>
            <div className="text-lg font-bold text-slate-700">{data.total_members}</div>
            <div className="text-[10px] text-slate-500">Records Analyzed</div>
          </div>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-50 border border-amber-200">
          <AlertTriangle size={18} className="text-amber-500" />
          <div>
            <div className="text-lg font-bold text-amber-700">{Number(data.anomaly_rate).toFixed(1)}%</div>
            <div className="text-[10px] text-amber-500">Anomaly Rate</div>
          </div>
        </div>
      </div>

      {Object.entries(data.datasets || {}).map(([dsName, ds]) => (
        <div key={dsName} className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-sm text-slate-700 capitalize">{dsName.replace(/_/g, ' ')}</h4>
            <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full">
              {ds.anomaly_count} anomalies / {ds.total} records
            </span>
          </div>
          {ds.top_anomalies && ds.top_anomalies.length > 0 && (
            <div className="space-y-2">
              {ds.top_anomalies.slice(0, 3).map((a, i) => (
                <div
                  key={i}
                  className="text-xs text-slate-600 bg-slate-50 rounded-lg px-3 py-2 border border-slate-100"
                >
                  <span className="font-medium text-red-600">Anomaly #{i + 1}:</span>{' '}
                  {a.description || `Deviating features: ${(a.top_features || []).join(', ')}`}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ─── Collapsible card (for predictions) ─────────────────────── */
function AiCard({ title, Icon, color, children, badge }) {
  const [open, setOpen] = useState(false);
  return (
    <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="card overflow-hidden">
      <button
        onClick={() => setOpen((p) => !p)}
        className={`w-full flex items-center justify-between px-5 py-4 ${color} text-white rounded-t-xl`}
      >
        <span className="flex items-center gap-2 font-bold text-lg">
          <Icon size={20} /> {title}
          {badge && <span className="ml-2 text-xs bg-white/20 rounded-full px-2 py-0.5">{badge}</span>}
        </span>
        <ChevronDown size={18} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="p-5 space-y-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

/* ─── Form input ─────────────────────────────────────────────── */
function Field({ label, value, onChange, type = 'number', min, max, step, placeholder }) {
  return (
    <label className="block">
      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(type === 'number' ? +e.target.value : e.target.value)}
        min={min}
        max={max}
        step={step || 'any'}
        placeholder={placeholder}
        className="mt-1 block w-full rounded-lg border border-slate-200 bg-white/60 px-3 py-2 text-sm
                   focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition"
      />
    </label>
  );
}

/* ─── Result badge ───────────────────────────────────────────── */
function ResultBadge({ label, value, color = 'bg-indigo-100 text-indigo-700' }) {
  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold ${color}`}>
      <Sparkles size={14} /> {label}: {value}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN PAGE
   ═══════════════════════════════════════════════════════════════ */
export default function AIInsightsPage() {
  const toast = useToast();
  const [tab, setTab] = useState('chat');
  const [mlOnline, setMlOnline] = useState(null);
  const [loading, setLoading] = useState({});

  // ── Data state ──
  const [insights, setInsights] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [anomalyData, setAnomalyData] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [clustersLoading, setClustersLoading] = useState(false);
  const [anomaliesLoading, setAnomaliesLoading] = useState(false);

  // ── Health check ──
  useEffect(() => {
    mlHealth()
      .then((r) => setMlOnline(r.data))
      .catch(() => setMlOnline(false));
  }, []);

  // ── Load insights ──
  useEffect(() => {
    if (tab === 'insights') {
      setInsightsLoading(true);
      getInsights()
        .then((r) => setInsights(r.data.insights || []))
        .catch(() => toast.error('Failed to load insights'))
        .finally(() => setInsightsLoading(false));
    }
  }, [tab]);

  // ── Load clusters ──
  useEffect(() => {
    if (tab === 'clusters') {
      setClustersLoading(true);
      getClusters()
        .then((r) => setClusters(r.data.clusters || []))
        .catch(() => toast.error('Clustering data not available — run training first'))
        .finally(() => setClustersLoading(false));
    }
  }, [tab]);

  // ── Load anomalies ──
  useEffect(() => {
    if (tab === 'anomalies') {
      setAnomaliesLoading(true);
      getAnomalies()
        .then((r) => setAnomalyData(r.data))
        .catch(() => toast.error('Anomaly data not available — run training first'))
        .finally(() => setAnomaliesLoading(false));
    }
  }, [tab]);

  // ──────────────────────────────────────────────────────────────
  //  PREDICTION FORMS (kept from v1)
  // ──────────────────────────────────────────────────────────────
  const [wrForm, setWrForm] = useState({
    age: 28,
    weight_kg: 75,
    height_m: 1.75,
    heart_rate_avg: 130,
    session_duration_hr: 1,
    experience_level: 2,
  });
  const [wrResult, setWrResult] = useState(null);
  const runWorkout = useCallback(async () => {
    setLoading((p) => ({ ...p, wr: true }));
    try {
      const { data } = await recommendWorkout(wrForm);
      setWrResult(data);
      toast.success(`Recommended: ${data.recommended_workout}`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Workout recommendation failed');
    } finally {
      setLoading((p) => ({ ...p, wr: false }));
    }
  }, [wrForm, toast]);

  const [calForm, setCalForm] = useState({
    age: 28,
    weight_kg: 75,
    height_cm: 175,
    heart_rate: 130,
    duration_min: 45,
    body_temp: 37.2,
    gender_encoded: 0,
  });
  const [calResult, setCalResult] = useState(null);
  const runCalories = useCallback(async () => {
    setLoading((p) => ({ ...p, cal: true }));
    try {
      const { data } = await predictCalories(calForm);
      setCalResult(data);
      toast.success(`Predicted burn: ${data.predicted_calories.toFixed(0)} kcal`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Calorie prediction failed');
    } finally {
      setLoading((p) => ({ ...p, cal: false }));
    }
  }, [calForm, toast]);

  const [adhForm, setAdhForm] = useState({
    age: 28,
    weight_kg: 75,
    daily_calorie_intake: 2200,
    workout_duration_min: 60,
    workout_intensity: 6,
    sleep_hours: 7,
    stress_level: 4,
  });
  const [adhResult, setAdhResult] = useState(null);
  const runAdherence = useCallback(async () => {
    setLoading((p) => ({ ...p, adh: true }));
    try {
      const { data } = await predictAdherence(adhForm);
      setAdhResult(data);
      toast.success(`Churn risk: ${data.risk_label}`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Adherence prediction failed');
    } finally {
      setLoading((p) => ({ ...p, adh: false }));
    }
  }, [adhForm, toast]);

  const [progForm, setProgForm] = useState({
    age: 28,
    weight_kg: 75,
    height_cm: 175,
    body_fat_pct: 18,
    systolic_bp: 120,
    diastolic_bp: 80,
    grip_force: 42,
    sit_and_bend_cm: 18,
    sit_ups_count: 35,
    broad_jump_cm: 210,
  });
  const [progResult, setProgResult] = useState(null);
  const runProgress = useCallback(async () => {
    setLoading((p) => ({ ...p, prog: true }));
    try {
      const { data } = await predictProgress(progForm);
      setProgResult(data);
      toast.success(`Performance score: ${data.predicted_performance_score.toFixed(2)}`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || 'Progress prediction failed');
    } finally {
      setLoading((p) => ({ ...p, prog: false }));
    }
  }, [progForm, toast]);

  const wf = (k, v) => setWrForm((p) => ({ ...p, [k]: v }));
  const cf = (k, v) => setCalForm((p) => ({ ...p, [k]: v }));
  const af = (k, v) => setAdhForm((p) => ({ ...p, [k]: v }));
  const pf = (k, v) => setProgForm((p) => ({ ...p, [k]: v }));

  // ──────────────────────────────────────────────────────────────
  //  RENDER
  // ──────────────────────────────────────────────────────────────
  return (
    <PageShell
      title="AI Intelligence Hub"
      subtitle="Chatbot · Insights · Anomaly Detection · ML Predictions"
      right={
        <span
          className={`inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full ${
            mlOnline === false
              ? 'bg-red-100 text-red-700'
              : mlOnline
                ? 'bg-emerald-100 text-emerald-700'
                : 'bg-slate-100 text-slate-500'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              mlOnline === false ? 'bg-red-500' : mlOnline ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'
            }`}
          />
          {mlOnline === false
            ? 'ML Offline'
            : mlOnline
              ? `AI Online — ${mlOnline.models_loaded?.length || 0} models${mlOnline.llm_available ? ' + LLM' : ''}`
              : 'Checking…'}
        </span>
      }
    >
      {/* ─── Tab bar ──────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2 mb-6">
        <Tab active={tab === 'chat'} onClick={() => setTab('chat')} Icon={MessageSquare} label="AI Chat" />
        <Tab
          active={tab === 'insights'}
          onClick={() => setTab('insights')}
          Icon={Lightbulb}
          label="Insights"
          badge={insights.length || null}
        />
        <Tab
          active={tab === 'clusters'}
          onClick={() => setTab('clusters')}
          Icon={Users}
          label="Segments"
          badge={clusters.length || null}
        />
        <Tab
          active={tab === 'anomalies'}
          onClick={() => setTab('anomalies')}
          Icon={ShieldAlert}
          label="Anomalies"
          badge={anomalyData?.total_anomalies || null}
        />
        <Tab active={tab === 'predict'} onClick={() => setTab('predict')} Icon={Brain} label="Predictions" />
      </div>

      {/* ═══ TAB: AI Chat ═════════════════════════════════════════ */}
      {tab === 'chat' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-3xl mx-auto">
          <div className="mb-3 px-4 py-3 rounded-xl border border-amber-200 bg-amber-50 text-amber-800 text-xs flex items-start gap-2">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0 text-amber-600" />
            <div>
              <div className="font-semibold">AI notice</div>
              <div className="opacity-90">
                Responses may be inaccurate. Don’t share passwords, payment info, or sensitive personal data. Verify
                before acting.
              </div>
            </div>
          </div>
          <AIChatPanel className="h-[600px]" />
        </motion.div>
      )}

      {/* ═══ TAB: Insights ════════════════════════════════════════ */}
      {tab === 'insights' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
          {insightsLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 size={24} className="animate-spin text-indigo-500" />
              <span className="ml-2 text-sm text-slate-500">Generating AI insights...</span>
            </div>
          ) : insights.length === 0 ? (
            <div className="text-center py-16 text-slate-400 text-sm">
              No insights available. Run the ML training pipeline first.
            </div>
          ) : (
            insights.map((insight, i) => <InsightCard key={i} insight={insight} index={i} />)
          )}
        </motion.div>
      )}

      {/* ═══ TAB: Clusters ════════════════════════════════════════ */}
      {tab === 'clusters' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {clustersLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 size={24} className="animate-spin text-indigo-500" />
              <span className="ml-2 text-sm text-slate-500">Loading member segments...</span>
            </div>
          ) : clusters.length === 0 ? (
            <div className="text-center py-16 text-slate-400 text-sm">
              No clustering data. Run the ML training pipeline first.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {clusters.map((c, i) => (
                <ClusterCard key={c.cluster_id ?? i} cluster={c} index={i} />
              ))}
            </div>
          )}
        </motion.div>
      )}

      {/* ═══ TAB: Anomalies ═══════════════════════════════════════ */}
      {tab === 'anomalies' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          {anomaliesLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 size={24} className="animate-spin text-indigo-500" />
              <span className="ml-2 text-sm text-slate-500">Loading anomaly report...</span>
            </div>
          ) : !anomalyData ? (
            <div className="text-center py-16 text-slate-400 text-sm">
              No anomaly data. Run the ML training pipeline first.
            </div>
          ) : (
            <AnomalySummary data={anomalyData} />
          )}
        </motion.div>
      )}

      {/* ═══ TAB: Predictions ═════════════════════════════════════ */}
      {tab === 'predict' && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
          {/* Model A */}
          <AiCard
            title="Workout Recommender"
            Icon={Dumbbell}
            color="bg-gradient-to-r from-indigo-500 to-purple-600"
            badge="LightGBM"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <Field label="Age" value={wrForm.age} onChange={(v) => wf('age', v)} min={10} max={100} />
              <Field label="Weight (kg)" value={wrForm.weight_kg} onChange={(v) => wf('weight_kg', v)} min={20} />
              <Field
                label="Height (m)"
                value={wrForm.height_m}
                onChange={(v) => wf('height_m', v)}
                min={1}
                step={0.01}
              />
              <Field
                label="Avg Heart Rate"
                value={wrForm.heart_rate_avg}
                onChange={(v) => wf('heart_rate_avg', v)}
                min={40}
                max={220}
              />
              <Field
                label="Session Hrs"
                value={wrForm.session_duration_hr}
                onChange={(v) => wf('session_duration_hr', v)}
                min={0.1}
                step={0.1}
              />
              <Field
                label="Experience (1-3)"
                value={wrForm.experience_level}
                onChange={(v) => wf('experience_level', v)}
                min={1}
                max={3}
              />
            </div>
            <button
              onClick={runWorkout}
              disabled={loading.wr}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading.wr ? <Loader2 size={16} className="animate-spin" /> : <Brain size={16} />} Recommend Workout
            </button>
            {wrResult && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
                <ResultBadge
                  label="Recommended"
                  value={wrResult.recommended_workout}
                  color="bg-purple-100 text-purple-700"
                />
                <ResultBadge label="Confidence" value={`${(wrResult.confidence * 100).toFixed(1)}%`} />
                <p className="text-xs text-slate-400">Inference: {wrResult.inference_ms.toFixed(1)} ms</p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {Object.entries(wrResult.all_probabilities || {}).map(([k, v]) => (
                    <span key={k} className="text-xs bg-slate-100 px-2 py-1 rounded-md">
                      {k}: {(v * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </motion.div>
            )}
          </AiCard>

          {/* Model B */}
          <AiCard
            title="Calorie Burn Predictor"
            Icon={Flame}
            color="bg-gradient-to-r from-orange-500 to-red-500"
            badge="LightGBM"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <Field label="Age" value={calForm.age} onChange={(v) => cf('age', v)} min={10} max={100} />
              <Field label="Weight (kg)" value={calForm.weight_kg} onChange={(v) => cf('weight_kg', v)} min={20} />
              <Field label="Height (cm)" value={calForm.height_cm} onChange={(v) => cf('height_cm', v)} min={100} />
              <Field
                label="Heart Rate"
                value={calForm.heart_rate}
                onChange={(v) => cf('heart_rate', v)}
                min={40}
                max={220}
              />
              <Field
                label="Duration (min)"
                value={calForm.duration_min}
                onChange={(v) => cf('duration_min', v)}
                min={1}
              />
              <Field
                label="Gender (0=M 1=F)"
                value={calForm.gender_encoded}
                onChange={(v) => cf('gender_encoded', v)}
                min={0}
                max={1}
              />
            </div>
            <button
              onClick={runCalories}
              disabled={loading.cal}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading.cal ? <Loader2 size={16} className="animate-spin" /> : <Flame size={16} />} Predict Calories
            </button>
            {calResult && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
                <ResultBadge
                  label="Predicted Burn"
                  value={`${calResult.predicted_calories.toFixed(0)} kcal`}
                  color="bg-orange-100 text-orange-700"
                />
                <p className="text-xs text-slate-400">Inference: {calResult.inference_ms.toFixed(1)} ms</p>
              </motion.div>
            )}
          </AiCard>

          {/* Model C */}
          <AiCard
            title="Churn Risk Predictor"
            Icon={AlertTriangle}
            color="bg-gradient-to-r from-amber-500 to-yellow-500"
            badge="XGBoost"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <Field label="Age" value={adhForm.age} onChange={(v) => af('age', v)} min={10} max={100} />
              <Field label="Weight (kg)" value={adhForm.weight_kg} onChange={(v) => af('weight_kg', v)} min={20} />
              <Field
                label="Calorie Intake"
                value={adhForm.daily_calorie_intake}
                onChange={(v) => af('daily_calorie_intake', v)}
                min={500}
              />
              <Field
                label="Workout Dur (min)"
                value={adhForm.workout_duration_min}
                onChange={(v) => af('workout_duration_min', v)}
                min={0}
              />
              <Field
                label="Intensity (1-10)"
                value={adhForm.workout_intensity}
                onChange={(v) => af('workout_intensity', v)}
                min={1}
                max={10}
              />
              <Field
                label="Sleep (hrs)"
                value={adhForm.sleep_hours}
                onChange={(v) => af('sleep_hours', v)}
                min={0}
                max={24}
              />
            </div>
            <button
              onClick={runAdherence}
              disabled={loading.adh}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading.adh ? <Loader2 size={16} className="animate-spin" /> : <Activity size={16} />} Predict Churn Risk
            </button>
            {adhResult && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
                <ResultBadge
                  label="Churn Risk"
                  value={adhResult.risk_label.toUpperCase()}
                  color={
                    adhResult.risk_label === 'high'
                      ? 'bg-red-100 text-red-700'
                      : adhResult.risk_label === 'medium'
                        ? 'bg-amber-100 text-amber-700'
                        : 'bg-emerald-100 text-emerald-700'
                  }
                />
                <ResultBadge label="Probability" value={`${(adhResult.churn_probability * 100).toFixed(1)}%`} />
                <p className="text-xs text-slate-400">Inference: {adhResult.inference_ms.toFixed(1)} ms</p>
              </motion.div>
            )}
          </AiCard>

          {/* Model D */}
          <AiCard
            title="Progress Forecaster"
            Icon={TrendingUp}
            color="bg-gradient-to-r from-emerald-500 to-teal-500"
            badge="XGBoost"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <Field label="Age" value={progForm.age} onChange={(v) => pf('age', v)} min={10} max={100} />
              <Field label="Weight (kg)" value={progForm.weight_kg} onChange={(v) => pf('weight_kg', v)} min={20} />
              <Field label="Height (cm)" value={progForm.height_cm} onChange={(v) => pf('height_cm', v)} min={100} />
              <Field
                label="Body Fat %"
                value={progForm.body_fat_pct}
                onChange={(v) => pf('body_fat_pct', v)}
                min={3}
                max={60}
              />
              <Field label="Grip Force" value={progForm.grip_force} onChange={(v) => pf('grip_force', v)} min={0} />
              <Field
                label="Sit-ups Count"
                value={progForm.sit_ups_count}
                onChange={(v) => pf('sit_ups_count', v)}
                min={0}
              />
            </div>
            <button
              onClick={runProgress}
              disabled={loading.prog}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading.prog ? <Loader2 size={16} className="animate-spin" /> : <Target size={16} />} Forecast
              Performance
            </button>
            {progResult && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
                <ResultBadge
                  label="Performance Score"
                  value={progResult.predicted_performance_score.toFixed(2)}
                  color="bg-teal-100 text-teal-700"
                />
                <p className="text-xs text-slate-400">Inference: {progResult.inference_ms.toFixed(1)} ms</p>
              </motion.div>
            )}
          </AiCard>
        </motion.div>
      )}
    </PageShell>
  );
}
