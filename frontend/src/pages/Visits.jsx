import { useCallback, useEffect, useState } from 'react';
import { api, fmtTime, pct } from '../api';

export default function Visits() {
  const [filters, setFilters] = useState({ q: '', status: '', gate: '', from: '', to: '' });
  const [page, setPage] = useState(1);
  const [data, setData] = useState({ visits: [], total: 0, pages: 1 });
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    const params = new URLSearchParams({ page, limit: 25 });
    Object.entries(filters).forEach(([k, v]) => {
      if (!v) return;
      if (k === 'from') params.set(k, new Date(`${v}T00:00:00`).toISOString());
      else if (k === 'to') params.set(k, new Date(`${v}T23:59:59`).toISOString());
      else params.set(k, v);
    });
    try {
      setData(await api(`/api/visits?${params}`));
    } catch (e) {
      setError(e.message);
    }
  }, [filters, page]);

  useEffect(() => {
    const t = setTimeout(load, 250);
    return () => clearTimeout(t);
  }, [load]);

  const set = (k) => (e) => {
    setPage(1);
    setFilters({ ...filters, [k]: e.target.value });
  };

  const checkout = async (id) => {
    try {
      await api(`/api/visits/${id}/checkout`, { method: 'PATCH' });
      load();
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div>
      <header className="page-head"><h2>Visit log</h2><span className="muted">{data.total} visits</span></header>
      {error && <div className="error">{error}</div>}
      <div className="filters">
        <input placeholder="Name" value={filters.q} onChange={set('q')} />
        <select value={filters.status} onChange={set('status')}>
          <option value="">All</option>
          <option value="recognized">Recognized</option>
          <option value="unknown">Unknown</option>
        </select>
        <input placeholder="Gate" value={filters.gate} onChange={set('gate')} />
        <label className="inline">From<input type="date" value={filters.from} onChange={set('from')} /></label>
        <label className="inline">To<input type="date" value={filters.to} onChange={set('to')} /></label>
      </div>
      <table>
        <thead><tr><th>Visitor</th><th>Status</th><th>Gate</th><th>Check-in</th><th>Check-out</th><th>Match</th><th /></tr></thead>
        <tbody>
          {data.visits.map((v) => (
            <tr key={v._id}>
              <td>{v.visitorName}</td>
              <td><span className={`pill ${v.status === 'recognized' ? 'ok' : 'warn'}`}>{v.status}</span></td>
              <td>{v.gate}</td>
              <td>{fmtTime(v.checkInAt)}</td>
              <td>{fmtTime(v.checkOutAt)}</td>
              <td>{pct(v.matchRatio)}</td>
              <td>{v.status === 'recognized' && !v.checkOutAt && <button className="small" onClick={() => checkout(v._id)}>Check out</button>}</td>
            </tr>
          ))}
          {data.visits.length === 0 && <tr><td colSpan={7} className="muted">No visits match.</td></tr>}
        </tbody>
      </table>
      <div className="row pager">
        <button className="secondary" disabled={page <= 1} onClick={() => setPage(page - 1)}>Prev</button>
        <span>Page {data.page || 1} of {Math.max(1, data.pages)}</span>
        <button className="secondary" disabled={page >= data.pages} onClick={() => setPage(page + 1)}>Next</button>
      </div>
    </div>
  );
}
