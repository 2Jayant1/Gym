/**
 * ML API client — talks to the FastAPI serving layer via the Vite /ml proxy.
 * v2: adds AI chatbot, insights, clusters, anomalies.
 */
import axios from 'axios';

const AUTH_STORAGE_KEY = 'gms.auth';

function getTokenFromStorage() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed?.token || null;
  } catch {
    return null;
  }
}

const ml = axios.create({
  baseURL: '/ml',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// Attach JWT for ML endpoints (shared AUTH_SECRET with backend)
ml.interceptors.request.use((config) => {
  const token = getTokenFromStorage();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

let _onUnauthorized = null;
export function setMlOnUnauthorized(fn) {
  _onUnauthorized = fn;
}

ml.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err?.response?.status === 401 && _onUnauthorized) {
      _onUnauthorized();
    }
    return Promise.reject(err);
  },
);

/* ─── Health ──────────────────────────────────────────── */
export const mlHealth = () => ml.get('/health');

/* ═══ AI CHATBOT ══════════════════════════════════════════ */
/** Non-streaming chat (returns full response) */
export const chatMessage = (message, sessionId = 'default') =>
  ml.post('/chat', { message, session_id: sessionId, stream: false });

/** Get suggested conversation starters */
export const chatSuggestions = () => ml.get('/chat/suggestions');

/** Clear conversation history */
export const chatClear = (sessionId = 'default') => ml.delete(`/chat/${sessionId}`);

/* ═══ NLG INSIGHTS ════════════════════════════════════════ */
export const getInsights = () => ml.get('/insights');

/* ═══ CLUSTERS ════════════════════════════════════════════ */
export const getClusters = () => ml.get('/clusters');

/* ═══ ANOMALIES ═══════════════════════════════════════════ */
export const getAnomalies = () => ml.get('/anomalies');

/* ─── Model A: Workout Recommender ───────────────────── */
export const recommendWorkout = (params) => ml.post('/recommend-workout', params);

/* ─── Model B: Calorie Predictor ─────────────────────── */
export const predictCalories = (params) => ml.post('/predict-calories', params);

/* ─── Model C: Adherence / Churn ─────────────────────── */
export const predictAdherence = (params) => ml.post('/predict-adherence', params);

/* ─── Model D: Progress Forecaster ───────────────────── */
export const predictProgress = (params) => ml.post('/predict-progress', params);

export default ml;
