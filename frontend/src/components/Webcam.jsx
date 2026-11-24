import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

/**
 * Live webcam preview. The parent calls ref.current.capture() to get a JPEG
 * frame as a base64 data URL.
 */
const Webcam = forwardRef(function Webcam({ width = 480, mirrored = true }, ref) {
  const videoRef = useRef(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let stream;
    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((s) => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch((e) => setError(`Camera not available: ${e.message}`));
    if (!navigator.mediaDevices) setError('This browser has no camera access (use localhost or https).');
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  useImperativeHandle(ref, () => ({
    capture() {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return null;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      return canvas.toDataURL('image/jpeg', 0.85);
    },
  }));

  if (error) return <div className="cam-error">{error}</div>;
  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className="webcam"
      style={{ width, transform: mirrored ? 'scaleX(-1)' : 'none' }}
    />
  );
});

export default Webcam;
