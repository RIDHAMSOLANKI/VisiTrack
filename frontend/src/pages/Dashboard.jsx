import { useCallback, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { API_URL, api, fmtTime, pct } from '../api';

export default function Dashboard({ token }) {
  const [stats, setStats] = useState(null);
  const [feed, setFeed] = useState([]);
  const [live, setLive] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    try {
      const [s, v] = await Promise.all([api('/api/stats'), api('/api/visits?limit=15')]);
      setStats(s);
      setFeed(v.visits);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
    // Live updates: the server pushes every new check-in over Socket.IO.
    const socket = io(API_URL, { auth: { token } });
    socket.on('connect', () => setLive(true));
    socket.on('disconnect', () => setLive(false));
    socket.on('visit:new', (visit) => {
      setFeed((f) => [visit, ...f].slice(0, 15));
      setStats((s) => s && {
        ...s,
        today: s.today + 1,
        unknownToday: s.unknownToday + (visit.status === 'unknown' ? 1 : 0),
        inside: s.inside + (visit.status === 'recognized' ? 1 : 0),
      });
    });
    socket.on('visit:updated', (visit) => {
      setFeed((f) => f.map((v) => (v._id === visit._id ? visit : v)));
      setStats((s) => s && { ...s, inside: Math.max(0, s.inside - 1) });
    });
    return () => socket.disconnect();
  }, [load, token]);

  const max = Math.max(1, ...(stats?.perDay || []).map((d) => d.count));

  return (
    <div>
      <header className="page-head">
        <h2>Dashboard</h2>
        <span className={`pill ${live ? 'ok' : ''}`}>{live ? '● Live' : '○ Connecting…'}</span>
      </header>
      {error && <div className="error">{error}</div>}

      <div className="stats">
        <Stat label="Visits today" value={stats?.today} />
        <Stat label="Inside now" value={stats?.inside} />
        <Stat label="Unknown today" value={stats?.unknownToday} warn />
        <Stat label="Registered visitors" value={stats?.visitors} />
      </div>

      <div className="grid-2">
        <section className="card">
          <h3>Live entries</h3>
          {feed.length === 0 && <p className="muted">No visits yet. Open the gate kiosk to check someone in.</p>}
          <ul className="feed">
            {feed.map((v) => (
              <li key={v._id} className={v.status}>
                <strong>{v.visitorName}</strong>
                <span className="muted">{v.gate} · {fmtTime(v.checkInAt)}</span>
                <span className="muted">match {pct(v.matchRatio)}</span>
              </li>
            ))}
          </ul>
        </section>
        <section className="card">
          <h3>Last 7 days</h3>
          <div className="bars">
            {(stats?.perDay || []).map((d) => (
              <div key={d._id} className="bar-col" title={`${d._id}: ${d.count}`}>
                <div className="bar" style={{ height: `${(d.count / max) * 100}%` }} />
                <span>{d._id.slice(5)}</span>
              </div>
            ))}
            {stats && stats.perDay.length === 0 && <p className="muted">No data yet.</p>}
          </div>
        </section>
      </div>
    </div>
  );
}

function Stat({ label, value, warn }) {
  return (
    <div className={`card stat ${warn && value ? 'warn' : ''}`}>
      <div className="stat-value">{value ?? '—'}</div>
      <div className="muted">{label}</div>
    </div>
  );
}
