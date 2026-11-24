# VisiTrack: Face-Recognition Visitor Management System

A webcam at the gate recognises visitors, logs every entry automatically, and an admin dashboard shows visits live.

- **Model:** Siamese network (TensorFlow Functional API) with a custom `L1Dist` layer, trained with a custom `tf.GradientTape` loop on LFW pairs through a cached, prefetched `tf.data` pipeline.
- **Detection:** OpenCV Haar cascade finds and crops the face before recognition.
- **Serving:** Flask API (`ml-service/`).
- **Backend:** Express.js + MongoDB with JWT admin auth and Socket.IO live updates (`backend/`).
- **Frontend:** React admin dashboard + gate kiosk page (`frontend/`).

```
 Webcam (gate kiosk page)
      │  JPEG frame every 2 s
      ▼
 Express API  /api/checkin ──────────► Flask ML service  /identify
      │                                  1. Haar cascade: find + crop face
      │                                  2. embedding CNN on the live face
      │                                  3. L1 + Dense head vs stored embeddings
      │  ◄──────────── {visitor_id, score, match_ratio}
      ▼
 MongoDB (visitors, visits) ──► Socket.IO ──► React dashboard (live)
```

## Why a Siamese network

A normal classifier has fixed classes, so every new visitor would mean retraining with many photos.
A Siamese network learns **"are these two faces the same person?"** instead, so a new visitor needs only
the few photos taken at registration and **no retraining** (one-shot learning, Koch et al., 2015).

```
image A ─► [embedding CNN] ─► e1 ─┐
                                  ├─► L1Dist |e1 − e2| ─► Dense(1, sigmoid) ─► same-person score
image B ─► [same CNN]      ─► e2 ─┘
```

At registration each photo is embedded **once** and stored. At check-in only the live face goes through the CNN.
Then the cheap L1 + Dense head scores it against every stored embedding. A visitor is recognised when at least
`VERIFY_THRESHOLD` of their stored photos score above `PAIR_THRESHOLD`.

## Repository layout

| Folder | What's inside |
|---|---|
| `ml/visitrack_ml/` | model (`model.py`), face detection (`faces.py`), dataset + `tf.data` pipeline (`data.py`) |
| `ml/train.py` | custom GradientTape training loop, checkpoints, early stopping |
| `ml/evaluate.py` | recognition metrics on **unseen people**: accuracy, precision, recall, ROC-AUC, FAR/FRR, EER, gate threshold |
| `ml/eval_detection.py` | face-detection accuracy (Haar) on LFW or on your own gate photos |
| `ml/collect_faces.py` | capture your own face crops from the webcam (optional extra training data) |
| `ml-service/` | Flask API: `/health`, `/detect`, `/register`, `/identify`, `DELETE /visitors/<id>` |
| `backend/` | Express API: auth, visitors, check-in, visit log, stats, Socket.IO |
| `frontend/` | React (Vite) dashboard, visitor registration with webcam, visit log, gate kiosk |
| `notebooks/train_colab.ipynb` | train on a free Colab GPU |

## How the data is split

LFW (Labeled Faces in the Wild, ~13k photos of ~5.7k people) is split **by person** into train / val / test (80/10/10).
Nobody in the test set is seen during training, so the test score measures recognition of *new* people, which is
exactly the gate situation. Pairs are balanced: as many "same person" as "different people" pairs.

## Setup

Requirements: Python 3.10–3.12, Node 18+, MongoDB (local install, Atlas, or `docker compose up -d`).

### 1. Train the model

Easiest: open `notebooks/train_colab.ipynb` in Google Colab with a GPU, run all cells, and download `siamese.keras` + `metrics.json` into `artifacts/`.

Locally:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r ml-service/requirements.txt

python ml/train.py --epochs 20          # downloads LFW on first run
python ml/evaluate.py                   # writes artifacts/metrics.json + roc.png
python ml/eval_detection.py --limit 3000
```

Quick CPU smoke run: `python ml/train.py --epochs 1 --max-people 300 --embedding-dim 512`.

Optional: add your own faces to training with `python ml/collect_faces.py --name yourname --count 50`.

### 2. Start the ML service

```bash
python ml-service/app.py               # http://localhost:5001
```

It loads `artifacts/siamese.keras` once at startup and reads the gate threshold from `artifacts/metrics.json`.
Settings (environment variables): `MODEL_PATH`, `GALLERY_DIR`, `PAIR_THRESHOLD`, `VERIFY_THRESHOLD` (default 0.5), `ML_SERVICE_KEY`.
For production-style serving: `cd ml-service && waitress-serve --port 5001 --call app:create_app`.

### 3. Start the backend

```bash
cd backend
cp .env.example .env        # then set JWT_SECRET, GATE_KEY, ADMIN_EMAIL, ADMIN_PASSWORD
npm install
npm start                   # http://localhost:5000
```

### 4. Start the dashboard

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

Log in with the admin account from `.env`, register a visitor with 3–5 webcam photos, then open
**Gate kiosk** (`/gate`), enter the gate key, and look at the camera.

## Results

Numbers come from running the evaluation scripts and are saved in `artifacts/metrics.json`:

- `detection.detection_accuracy`: Haar cascade finds the face in the right place (IoU ≥ 0.3) on LFW photos.
- `recognition.at_gate_threshold.accuracy / precision / recall`: same/different decisions on test pairs of unseen people.
- `recognition.eer`, `recognition.roc_auc`: threshold-free quality measures.

Measured so far: Haar-cascade face detection found the face in the right place in **98.6%** of 2,000 random LFW
photos (`python ml/eval_detection.py --limit 2000`). Recognition numbers come from the full training run.

## Tests

```bash
python -m pytest ml/tests ml-service/tests     # model, data pipeline, Flask API
cd backend && npm test                          # API with an in-memory MongoDB
```

## Privacy and security

- Raw face photos are not stored: the ML service keeps only embeddings, and deleting a visitor deletes them.
- Admin routes need a JWT. Gate kiosks use a separate gate key. The ML service can require `ML_SERVICE_KEY` and should not be exposed to the internet.
- Known limitation: no liveness detection, so a printed photo could fool the camera. Next step: blink or head-turn checks.
