import base64
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ml"))

cv2 = pytest.importorskip("cv2")

import app as service  # noqa: E402


class FakeRecognizer:
    """Embedding = mean colour of the face; score = 1 when colours match."""

    def embed(self, faces):
        return faces.reshape(len(faces), -1, 3).mean(axis=1)

    def scores(self, live, stored):
        return np.clip(1 - np.abs(stored - live).sum(axis=1) * 5, 0, 1)


def b64_image(color):
    img = np.full((120, 120, 3), color, dtype=np.uint8)
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Pretend every image has a face filling the frame, so tests do not need real photos.
    from visitrack_ml.faces import to_model_input

    def fake_largest_face(img):
        return to_model_input(img), (0, 0, img.shape[1], img.shape[0])

    monkeypatch.setattr(service, "largest_face_input", fake_largest_face)
    app = service.create_app(recognizer=FakeRecognizer(), gallery_dir=tmp_path,
                             pair_threshold=0.5, verify_threshold=0.5, service_key="secret")
    c = app.test_client()
    c.environ_base["HTTP_X_SERVICE_KEY"] = "secret"
    return c


def test_health_needs_no_key(tmp_path):
    app = service.create_app(recognizer=FakeRecognizer(), gallery_dir=tmp_path, service_key="secret")
    r = app.test_client().get("/health")
    assert r.status_code == 200 and r.json["model_loaded"] is True


def test_key_required(tmp_path):
    app = service.create_app(recognizer=FakeRecognizer(), gallery_dir=tmp_path, service_key="secret")
    assert app.test_client().post("/identify", json={"image": b64_image(10)}).status_code == 401


def test_register_then_identify(client):
    r = client.post("/register", json={"visitor_id": "red1", "images": [b64_image((0, 0, 200))] * 2})
    assert r.status_code == 200 and r.json["faces_used"] == 2
    client.post("/register", json={"visitor_id": "green1", "images": [b64_image((0, 200, 0))]})

    r = client.post("/identify", json={"image": b64_image((0, 0, 200))})
    assert r.json["recognized"] is True and r.json["visitor_id"] == "red1"

    r = client.post("/identify", json={"image": b64_image((200, 0, 0))})
    assert r.json["recognized"] is False and r.json["visitor_id"] is None


def test_delete_visitor(client):
    client.post("/register", json={"visitor_id": "v1", "images": [b64_image(50)]})
    assert client.delete("/visitors/v1").json["removed"] is True
    assert client.get("/health").json["visitors"] == 0


def test_bad_inputs(client):
    assert client.post("/identify", json={"image": "not-base64!!"}).status_code == 400
    assert client.post("/register", json={"visitor_id": "../x", "images": [b64_image(1)]}).status_code == 400
    assert client.post("/register", json={"visitor_id": "ok", "images": []}).status_code == 400


def test_no_model_gives_503(tmp_path):
    app = service.create_app(gallery_dir=tmp_path, model_path=tmp_path / "missing.keras", service_key="")
    r = app.test_client().post("/identify", json={"image": b64_image(1)})
    assert r.status_code == 503 and "Train it first" in r.json["error"]
