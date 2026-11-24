export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export function getToken() {
  try {
    return localStorage.getItem('vt_token');
  } catch {
    return null;
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem('vt_token', token);
    else localStorage.removeItem('vt_token');
  } catch {
    /* storage blocked: stay logged in for this tab only */
  }
}

export async function api(path, { method = 'GET', body, headers = {}, auth = true } = {}) {
  const token = getToken();
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(auth && token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && auth) {
    setToken(null);
    window.dispatchEvent(new Event('vt:logout'));
  }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

export const fmtTime = (d) => (d ? new Date(d).toLocaleString() : '—');
export const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);
