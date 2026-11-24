import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, fmtTime } from '../api';
import FaceCapture from '../components/FaceCapture.jsx';

const EMPTY = { name: '', phone: '', email: '', purpose: '', host: '' };

export default function Visitors() {
  const [visitors, setVisitors] = useState([]);
  const [q, setQ] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(EMPTY);
  const [photos, setPhotos] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const load = useCallback(async () => {
    try {
      const data = await api(`/api/visitors?q=${encodeURIComponent(q)}`);
      setVisitors(data.visitors);
    } catch (e) {
      setError(e.message);
    }
  }, [q]);

  useEffect(() => {
    const t = setTimeout(load, 250);
    return () => clearTimeout(t);
  }, [load]);

  const submit = async (e) => {
    e.preventDefault();
    if (photos.length === 0) return setError('Take at least one photo');
    setBusy(true);
    setError('');
    try {
      const res = await api('/api/visitors', { method: 'POST', body: { ...form, images: photos } });
      setNotice(`Registered ${res.visitor.name}: ${res.facesUsed} face(s) stored${res.skipped ? `, ${res.skipped} photo(s) had no face` : ''}.`);
      setForm(EMPTY);
      setPhotos([]);
      setShowForm(false);
      load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const field = (key, label, props = {}) => (
    <label>{label}<input value={form[key]} onChange={(e) => setForm({ ...form, [key]: e.target.value })} {...props} /></label>
  );

  return (
    <div>
      <header className="page-head">
        <h2>Visitors</h2>
        <button onClick={() => setShowForm((s) => !s)}>{showForm ? 'Cancel' : '+ Register visitor'}</button>
      </header>
      {notice && <div className="notice">{notice}</div>}
      {error && <div className="error">{error}</div>}

      {showForm && (
        <form className="card grid-2" onSubmit={submit}>
          <div className="form-col">
            {field('name', 'Full name *', { required: true })}
            {field('phone', 'Phone')}
            {field('email', 'Email', { type: 'email' })}
            {field('purpose', 'Purpose of visit')}
            {field('host', 'Visiting (host)')}
            <button disabled={busy}>{busy ? 'Saving…' : 'Save visitor'}</button>
          </div>
          <FaceCapture photos={photos} setPhotos={setPhotos} />
        </form>
      )}

      <input className="search" placeholder="Search name, phone, email, host…" value={q} onChange={(e) => setQ(e.target.value)} />
      <table>
        <thead><tr><th>Name</th><th>Phone</th><th>Host</th><th>Faces</th><th>Registered</th></tr></thead>
        <tbody>
          {visitors.map((v) => (
            <tr key={v._id}>
              <td><Link to={`/visitors/${v._id}`}>{v.name}</Link></td>
              <td>{v.phone || '—'}</td>
              <td>{v.host || '—'}</td>
              <td>{v.faceCount}</td>
              <td>{fmtTime(v.createdAt)}</td>
            </tr>
          ))}
          {visitors.length === 0 && <tr><td colSpan={5} className="muted">No visitors found.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}
