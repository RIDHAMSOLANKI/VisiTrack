import { useEffect, useRef, useState } from 'react';
import { API_URL } from '../api';
import Webcam from '../components/Webcam.jsx';

const SCAN_EVERY_MS = 2000;

function readSetting(key, fallback = '') {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function saveSetting(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

/** Gate kiosk: grabs a webcam frame every 2 s and sends it for check-in. */
export default function Gate() {
  const cam = useRef(null);
  const busy = useRef(false);
  const [gateKey, setGateKey] = useState(readSetting('vt_gate_key'));
  const [gate, setGate] = useState(readSetting('vt_gate_name', 'main'));
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!running) return undefined;
    const tick = async () => {
      if (busy.current) return; // never send a new frame while the last one is in flight
      const image = cam.current?.capture();
      if (!image) return;
      busy.current = true;
      try {
        const res = await fetch(`${API_URL}/api/checkin`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-Gate-Key': gateKey },
          body: JSON.stringify({ image, gate }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `Error ${res.status}`);
        setError('');
        if (data.status !== 'no_face') setResult({ ...data, at: Date.now() });
      } catch (e) {
        setError(e.message);
      } finally {
        busy.current = false;
      }
    };
    const id = setInterval(tick, SCAN_EVERY_MS);
    return () => clearInterval(id);
  }, [running, gateKey, gate]);

  // Clear the welcome message after a few seconds.
  useEffect(() => {
    if (!result) return undefined;
    const t = setTimeout(() => setResult(null), 4000);
    return () => clearTimeout(t);
  }, [result]);

  if (!running) {
    return (
      <div className="center-page">
        <form className="card login" onSubmit={(e) => {
          e.preventDefault();
          saveSetting('vt_gate_key', gateKey);
          saveSetting('vt_gate_name', gate);
          setRunning(true);
        }}>
          <h1>Gate kiosk</h1>
          <label>Gate name<input value={gate} onChange={(e) => setGate(e.target.value)} required /></label>
          <label>Gate key<input type="password" value={gateKey} onChange={(e) => setGateKey(e.target.value)} required /></label>
          <button>Start scanning</button>
        </form>
      </div>
    );
  }

  const name = result?.visit?.visitorName;
  return (
    <div className="kiosk">
      <h1>Welcome — please look at the camera</h1>
      <Webcam ref={cam} width={640} />
      <div className={`kiosk-result ${result?.status || ''}`}>
        {!result && 'Scanning…'}
        {result?.status === 'recognized' && <>✓ Welcome, {name}{result.duplicate ? ' (already checked in)' : ''}</>}
        {result?.status === 'unknown' && <>Not recognised. Please register at the front desk.</>}
      </div>
      {error && <div className="error">{error}</div>}
      <button className="secondary" onClick={() => setRunning(false)}>Stop</button>
    </div>
  );
}
