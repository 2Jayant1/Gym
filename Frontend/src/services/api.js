import axios from 'axios';

// Determine API base URL - use environment variable or default to root for dev
const getBaseURL = () => {
  // In development, Vite proxy handles /api to backend
  // In production, use full backend URL if provided
  const backendUrl = import.meta.env.VITE_API_BASE || import.meta.env.BACKEND_URL;
  
  if (backendUrl) {
    return backendUrl;
  }
  
  // Default: root so /api paths keep working with Vite proxy
  return '';
};

const api = axios.create({
  baseURL: getBaseURL(),
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export function setAuthToken(token) {
  if (token) {
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common.Authorization;
  }
}

// Stores a callback that App.jsx will set so we can force-logout on 401
let _onUnauthorized = null;
export function setOnUnauthorized(fn) {
  _onUnauthorized = fn;
}

api.interceptors.response.use(
  (res) => {
    // Auto-unwrap paginated responses: { data: [...], pagination: {...} }
    // so existing code like `res.data || []` keeps working without changes.
    const body = res.data;
    if (body && Array.isArray(body.data) && body.pagination) {
      res.pagination = body.pagination;
      res.data = body.data;
    }
    return res;
  },
  (err) => {
    if (err?.response?.status === 401 && _onUnauthorized) {
      _onUnauthorized();
    }
    return Promise.reject(err);
  }
);

export default api;
