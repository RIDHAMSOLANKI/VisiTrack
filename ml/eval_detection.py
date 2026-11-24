"""Measure face DETECTION (finding the face), separately from recognition.

Mode 1, LFW (default): every LFW photo has one centred face. A detection counts
as correct when the biggest box overlaps the known face area with IoU >= --iou.

Mode 2, your own photos: --images-dir path/to/gate_photos, one face per photo.
There are no reference boxes, so this reports how often a face was found.

    python ml/eval_detection.py --limit 3000
    python ml/eval_detection.py --images-dir data/gate_photos
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from evaluate import update_metrics_file  # noqa: E402
from visitrack_ml import data  # noqa: E402
from visitrack_ml.config import LFW_FACE_BOX  # noqa: E402
from visitrack_ml.faces import detect_faces  # noqa: E402


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-dir")
    p.add_argument("--images-dir", help="Folder of your own photos, one face each")
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.images_dir:
        paths = sorted(str(x) for x in Path(args.images_dir).rglob("*") if x.suffix.lower() in data.IMAGE_EXTS)
        reference = None
        source = args.images_dir
    else:
        root = data.find_or_download_lfw(args.lfw_dir)
        paths = [img for imgs in data.scan_people(root).values() for img in imgs]
        top, bottom, left, right = LFW_FACE_BOX
        reference = (left, top, right, bottom)
        source = "LFW"

    random.Random(args.seed).shuffle(paths)
    paths = paths[: args.limit]

    found = correct = 0
    for i, path in enumerate(paths, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        boxes = detect_faces(img, min_size=40)
        if boxes:
            found += 1
            x, y, w, h = boxes[0]
            if reference is None or iou((x, y, x + w, y + h), reference) >= args.iou:
                correct += 1
        if i % 500 == 0:
            print(f"  {i}/{len(paths)}  correct so far {correct / i:.4f}")

    n = len(paths)
    result = {
        "source": source,
        "images": n,
        "detector": "OpenCV Haar cascade (haarcascade_frontalface_default)",
        "face_found_rate": found / n if n else 0.0,
        "detection_accuracy": correct / n if n else 0.0,
        "iou_threshold": args.iou if reference else None,
    }
    update_metrics_file("detection", result)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
