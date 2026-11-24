import { useRef, useState } from 'react';
import Webcam from './Webcam.jsx';

/** Webcam + "take photo" button. Collects up to `max` frames for registration. */
export default function FaceCapture({ photos, setPhotos, max = 5 }) {
  const cam = useRef(null);
  const [msg, setMsg] = useState('');

  const take = () => {
    const shot = cam.current?.capture();
    if (!shot) return setMsg('Camera not ready yet');
    setMsg('');
    setPhotos((p) => [...p, shot].slice(0, max));
  };

  return (
    <div className="capture">
      <Webcam ref={cam} width={360} />
      <div className="row">
        <button type="button" onClick={take} disabled={photos.length >= max}>
          Take photo ({photos.length}/{max})
        </button>
        {photos.length > 0 && <button type="button" className="secondary" onClick={() => setPhotos([])}>Clear</button>}
      </div>
      <p className="muted small">Take 3–5 photos: look straight, then turn slightly left and right. Good light helps.</p>
      {msg && <div className="error">{msg}</div>}
      <div className="thumbs">
        {photos.map((p, i) => <img key={i} src={p} alt={`capture ${i + 1}`} />)}
      </div>
    </div>
  );
}
