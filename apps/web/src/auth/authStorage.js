const AUTH_STORAGE_KEY = 'gms.auth';

export function loadAuth() {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveAuth(data) {
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(data));
}

export function clearAuth() {
  localStorage.removeItem(AUTH_STORAGE_KEY);
}
