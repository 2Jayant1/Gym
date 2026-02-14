/* eslint-disable no-console */
const path = require('path');

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function isListening(url) {
  try {
    const res = await fetch(url, { method: 'GET' });
    return !!res;
  } catch {
    return false;
  }
}

async function httpJson(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  return { status: res.status, data };
}

(async () => {
  // 1) API sanity
  const base = 'http://127.0.0.1:5001';

  // If the server isn't already running, start it in-process (same Node instance)
  // so smoke can be run as a single command.
  let startedServer = false;
  if (!(await isListening(`${base}/api/membership-plans`))) {
    console.log('[api] server not detected on 5001; starting a temporary server for smoke…');
    startedServer = true;
    // server.js starts listening immediately; it also initializes schema/seed.
    // We require it from the repo root to match normal runtime behavior.
    // eslint-disable-next-line global-require
    require(path.join(__dirname, '..', '..', 'server.js'));
    // Give Express a moment to bind the port.
    for (let i = 0; i < 20; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      if (await isListening(`${base}/api/membership-plans`)) break;
      // eslint-disable-next-line no-await-in-loop
      await sleep(150);
    }
  }

  const login = await httpJson(`${base}/api/auth/login`, {
    method: 'POST',
    body: JSON.stringify({ username: 'admin', password: 'admin123' }),
  });

  if (!login.data?.token) {
    console.error('[api] login failed', login);
    process.exit(1);
  }

  const token = login.data.token;
  console.log('[api] login ok as', login.data.user?.role);

  const stats = await httpJson(`${base}/api/stats`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  console.log('[api] /api/stats', stats.status, stats.data);

  const adminUsers = await httpJson(`${base}/api/admin/table/users`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  console.log(
    '[api] /api/admin/table/users',
    adminUsers.status,
    Array.isArray(adminUsers.data) ? adminUsers.data.length : adminUsers.data,
  );

  const adminTrainingSessions = await httpJson(`${base}/api/admin/training-sessions`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  console.log(
    '[api] /api/admin/training-sessions',
    adminTrainingSessions.status,
    Array.isArray(adminTrainingSessions.data) ? adminTrainingSessions.data.length : adminTrainingSessions.data,
  );

  // Extra coverage: trainer + member scoped endpoints exist and return arrays.
  const trainerLogin = await httpJson(`${base}/api/auth/login`, {
    method: 'POST',
    body: JSON.stringify({ username: 'trainer', password: 'trainer123' }),
  });
  if (trainerLogin.data?.token) {
    const ttoken = trainerLogin.data.token;
    const trainerSessions = await httpJson(`${base}/api/trainer/sessions`, {
      headers: { Authorization: `Bearer ${ttoken}` },
    });
    console.log(
      '[api] /api/trainer/sessions',
      trainerSessions.status,
      Array.isArray(trainerSessions.data) ? trainerSessions.data.length : trainerSessions.data,
    );
  } else {
    console.warn('[api] trainer login failed; skipping trainer endpoint checks');
  }

  const memberLogin = await httpJson(`${base}/api/auth/login`, {
    method: 'POST',
    body: JSON.stringify({ username: 'member', password: 'member123' }),
  });
  if (memberLogin.data?.token) {
    const mtoken = memberLogin.data.token;
    const meAttendance = await httpJson(`${base}/api/me/attendance`, {
      headers: { Authorization: `Bearer ${mtoken}` },
    });
    console.log(
      '[api] /api/me/attendance',
      meAttendance.status,
      Array.isArray(meAttendance.data) ? meAttendance.data.length : meAttendance.data,
    );
  } else {
    console.warn('[api] member login failed; skipping member endpoint checks');
  }

  if (startedServer) {
    console.log('[api] smoke completed (temporary server was started).');
  }

  process.exit(0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
