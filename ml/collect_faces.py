"""Capture face crops of one person from the webcam into data/custom/<name>/.

These can be added to training (train.py picks up data/custom automatically).

    python ml/collect_faces.py --name ridham --count 50

Keys: SPACE = capture now, A = toggle auto-capture, Q = quit.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from visitrack_ml.config import DATA_DIR  # noqa: E402
from visitrack_ml.faces import crop_face, detect_faces  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--interval", type=float, default=0.4, help="Seconds between auto captures")
    args = p.parse_args()

    out = DATA_DIR / "custom" / args.name
    out.mkdir(parents=True, exist_ok=True)
    saved = len(list(out.glob("*.jpg")))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam")

    auto, last = False, 0.0
    print("SPACE = capture, A = auto on/off, Q = quit")
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect_faces(frame)
        view = frame.copy()
        for (x, y, w, h) in boxes[:1]:
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(view, f"{saved}/{args.count}  auto={'on' if auto else 'off'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imshow("VisiTrack - collect faces", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            auto = not auto
        want = key == ord(" ") or (auto and time.time() - last > args.interval)
        if want and boxes:
            cv2.imwrite(str(out / f"{args.name}_{saved:04d}.jpg"), crop_face(frame, boxes[0]))
            saved += 1
            last = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images in {out}")


if __name__ == "__main__":
    main()
