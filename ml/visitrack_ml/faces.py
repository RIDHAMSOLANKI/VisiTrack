"""Face detection (OpenCV Haar cascade) and turning a face into model input."""

from functools import lru_cache

import cv2
import numpy as np

from .config import IMG_SIZE


@lru_cache(maxsize=1)
def _cascade():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {path}")
    return cascade


def detect_faces(image_bgr: np.ndarray, min_size: int = 60):
    """Return face boxes as a list of (x, y, w, h), biggest first.

    A Haar cascade slides a window over the image and runs a chain of quick
    light/dark rectangle tests, throwing away non-face areas early. It is fast
    on a CPU, but weaker with tilted faces and bad lighting.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = _cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    boxes = [tuple(int(v) for v in b) for b in boxes]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_face(image_bgr: np.ndarray, box, margin: float = 0.15) -> np.ndarray:
    """Cut the face out with a small margin so it looks like the LFW training crops."""
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    H, W = image_bgr.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return image_bgr[y0:y1, x0:x1]


def to_model_input(face_bgr: np.ndarray) -> np.ndarray:
    """BGR face crop -> float32 RGB array of shape (100, 100, 3) scaled to 0..1."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (rgb.astype(np.float32) / 255.0)


def largest_face_input(image_bgr: np.ndarray):
    """Detect, crop and prepare the biggest face. Returns (model_input, box) or (None, None)."""
    boxes = detect_faces(image_bgr)
    if not boxes:
        return None, None
    box = boxes[0]
    return to_model_input(crop_face(image_bgr, box)), box


def decode_image(data: bytes) -> np.ndarray:
    """Raw JPEG/PNG bytes -> BGR image. Raises ValueError if the bytes are not an image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img
