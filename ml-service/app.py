"""VisiTrack ML service (Flask).

Serves the trained Siamese model over REST so the Node backend can use it.

Endpoints
  GET    /health
  POST   /detect                {image}                   -> face boxes
  POST   /register              {visitor_id, images: [...]} -> store embeddings
  POST   /identify              {image}                   -> who is this?
  DELETE /visitors/<visitor_id>

Images are base64 strings (a "data:image/jpeg;base64," prefix is fine).
If ML_SERVICE_KEY is set, every request except /health needs header X-Service-Key.

Run:  python ml-service/app.py      (dev)
      waitress-serve --port 5001 --call app:create_app   (production-style, from ml-service/)
"""

import base64
import binascii
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ml"))

import numpy as np  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402
from flask_cors import CORS  # noqa: E402

from gallery import Gallery  # noqa: E402
from visitrack_ml.config import METRICS_FILE, MODEL_FILE  # noqa: E402
from visitrack_ml.faces import decode_image, detect_faces, largest_face_input  # noqa: E402

MAX_IMAGES_PER_REGISTER = 10


def default_pair_threshold() -> float:
    """Use the threshold picked by ml/evaluate.py if it exists, else 0.5."""
    if os.getenv("PAIR_THRESHOLD"):
        return float(os.environ["PAIR_THRESHOLD"])
    try:
        return float(json.loads(METRICS_FILE.read_text())["recognition"]["gate_threshold"])
    except Exception:
        return 0.5


class Recognizer:
    """Wraps the trained model: embed faces and score them against the gallery."""

    def __init__(self, siamese):
        from visitrack_ml.model import split_model
        self.embedding, self.classifier = split_model(siamese)

    def embed(self, faces: np.ndarray) -> np.ndarray:
        return np.asarray(self.embedding.predict_on_batch(faces))

    def scores(self, live: np.ndarray, stored: np.ndarray) -> np.ndarray:
        from visitrack_ml.model import score_embeddings
        return score_embeddings(self.classifier, live, stored)


def decode_b64(value) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ValueError("image must be a base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image is not valid base64")
    return decode_image(raw)


def create_app(recognizer=None, gallery_dir=None, model_path=None,
               pair_threshold=None, verify_threshold=None, service_key=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    CORS(app)

    gallery = Gallery(gallery_dir or os.getenv("GALLERY_DIR", HERE / "gallery"))
    pair_t = pair_threshold if pair_threshold is not None else default_pair_threshold()
    # Share of a visitor's stored photos that must match the live face.
    verify_t = verify_threshold if verify_threshold is not None else float(os.getenv("VERIFY_THRESHOLD", 0.5))
    key = service_key if service_key is not None else os.getenv("ML_SERVICE_KEY", "")

    state = {"recognizer": recognizer, "error": None}
    if recognizer is None:
        path = Path(model_path or os.getenv("MODEL_PATH", MODEL_FILE))
        if path.exists():
            from visitrack_ml.model import load_siamese
            state["recognizer"] = Recognizer(load_siamese(path))  # loaded ONCE at startup
        else:
            state["error"] = f"Model not found at {path}. Train it first: python ml/train.py"

    def need_model():
        if state["recognizer"] is None:
            return jsonify(error=state["error"]), 503
        return None

    @app.before_request
    def check_key():
        if request.method == "OPTIONS" or request.endpoint in ("health", None):
            return None
        if key and request.headers.get("X-Service-Key") != key:
            return jsonify(error="unauthorized"), 401

    @app.errorhandler(ValueError)
    def bad_input(e):
        return jsonify(error=str(e)), 400

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=state["recognizer"] is not None,
                       model_error=state["error"], visitors=len(gallery),
                       pair_threshold=pair_t, verify_threshold=verify_t)

    @app.post("/detect")
    def detect():
        img = decode_b64((request.get_json(silent=True) or {}).get("image"))
        boxes = detect_faces(img)
        return jsonify(faces=[dict(x=x, y=y, w=w, h=h) for x, y, w, h in boxes])

    @app.post("/register")
    def register():
        if (resp := need_model()):
            return resp
        body = request.get_json(silent=True) or {}
        visitor_id = Gallery.check_id(body.get("visitor_id"))
        images = body.get("images") or []
        if not isinstance(images, list) or not images:
            raise ValueError("images must be a non-empty list")
        if len(images) > MAX_IMAGES_PER_REGISTER:
            raise ValueError(f"at most {MAX_IMAGES_PER_REGISTER} images per request")

        faces, skipped = [], 0
        for b64 in images:
            face, _ = largest_face_input(decode_b64(b64))
            if face is None:
                skipped += 1
            else:
                faces.append(face)
        if not faces:
            return jsonify(error="No face found in any image. Face the camera in good light."), 422

        embeddings = state["recognizer"].embed(np.stack(faces))
        total = gallery.add(visitor_id, embeddings, replace=bool(body.get("replace")))
        return jsonify(visitor_id=visitor_id, faces_used=len(faces), skipped=skipped, stored=total)

    @app.post("/identify")
    def identify():
        if (resp := need_model()):
            return resp
        t0 = time.perf_counter()
        img = decode_b64((request.get_json(silent=True) or {}).get("image"))
        face, box = largest_face_input(img)
        if face is None:
            return jsonify(face_found=False, recognized=False)

        live = state["recognizer"].embed(face[None, ...])[0]
        candidates = []
        for visitor_id, stored in gallery.items():
            s = state["recognizer"].scores(live, stored)
            candidates.append({
                "visitor_id": visitor_id,
                "score": float(np.mean(s)),
                "best": float(np.max(s)),
                "match_ratio": float(np.mean(s >= pair_t)),
            })
        candidates.sort(key=lambda c: (c["match_ratio"], c["score"]), reverse=True)

        best = candidates[0] if candidates else None
        recognized = bool(best and best["match_ratio"] >= verify_t)
        x, y, w, h = box
        return jsonify(
            face_found=True,
            recognized=recognized,
            visitor_id=best["visitor_id"] if recognized else None,
            score=best["score"] if best else None,
            match_ratio=best["match_ratio"] if best else None,
            box=dict(x=x, y=y, w=w, h=h),
            candidates=candidates[:3],
            ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @app.delete("/visitors/<visitor_id>")
    def delete_visitor(visitor_id):
        return jsonify(removed=gallery.remove(visitor_id))

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)))
