"""Shared settings for the ML code and the Flask service."""

from pathlib import Path

# Every face crop is resized to this before it reaches the model.
IMG_SIZE = 100
CHANNELS = 3

# LFW photos are 250x250 with the face in the middle.
# This box (top, bottom, left, right) cuts out the face, same as scikit-learn's default.
LFW_FACE_BOX = (70, 195, 78, 172)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

MODEL_FILE = ARTIFACTS_DIR / "siamese.keras"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"
