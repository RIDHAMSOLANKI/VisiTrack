import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api, fmtTime, pct } from '../api';
import FaceCapture from '../components/FaceCapture.jsx';

export default function VisitorDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const load = useCallback(async () => {
    try {
      setData(await api(`/api/visitors/${id}`));
    } catch (e) {
      setError(e.message);
    }
  }, [id]);

  useEffect(() => { load(); }, [load]);

  const addFaces = async () => {
    try {
      const res = await api(`/api/visitors/${id}/faces`, { method: 'POST', body: { images: photos } });
      setNotice(`Added ${res.facesUsed} face(s).`);
      setPhotos([]);
      setAdding(false);
      load();
    } catch (e) {
      setError(e.message);
    }
  };

  const remove = async () => {
    if (!window.confirm('Delete this visitor and their face data? Visit history is kept.')) return;
    try {
      await api(`/api/visitors/${id}`, { method: 'DELETE' });
      navigate('/visitors');
    } catch (e) {
      setError(e.message);
    }
  };

  if (!data) return <div>{error ? <div className="error">{error}</div> : 'Loading…'}</div>;
  const { visitor, visits } = data;

  return (
    <div>
      <Link to="/visitors">← Visitors</Link>
      <header className="page-head">
        <h2>{visitor.name}</h2>
        <div className="row">
          <button className="secondary" onClick={() => setAdding((a) => !a)}>{adding ? 'Cancel' : 'Add photos'}</button>
          <button className="danger" onClick={remove}>Delete</button>
        </div>
      </header>
      {notice && <div className="notice">{notice}</div>}
      {error && <div className="error">{error}</div>}

      <div className="card details">
        <div><span className="muted">Phone</span>{visitor.phone || '—'}</div>
        <div><span className="muted">Email</span>{visitor.email || '—'}</div>
        <div><span className="muted">Purpose</span>{visitor.purpose || '—'}</div>
        <div><span className="muted">Host</span>{visitor.host || '—'}</div>
        <div><span className="muted">Stored faces</span>{visitor.faceCount}</div>
        <div><span className="muted">Registered</span>{fmtTime(visitor.createdAt)}</div>
      </div>

      {adding && (
        <div className="card">
          <FaceCapture photos={photos} setPhotos={setPhotos} />
          <button onClick={addFaces} disabled={!photos.length}>Save photos</button>
        </div>
      )}

      <h3>Visit history</h3>
      <table>
        <thead><tr><th>Check-in</th><th>Check-out</th><th>Gate</th><th>Match</th></tr></thead>
        <tbody>
          {visits.map((v) => (
            <tr key={v._id}>
              <td>{fmtTime(v.checkInAt)}</td>
              <td>{fmtTime(v.checkOutAt)}</td>
              <td>{v.gate}</td>
              <td>{pct(v.matchRatio)}</td>
            </tr>
          ))}
          {visits.length === 0 && <tr><td colSpan={4} className="muted">No visits yet.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}
