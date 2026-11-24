"""Measure recognition quality on the held-out test pairs (people never seen in training).

Reports accuracy, precision, recall, ROC-AUC, FAR, FRR and the EER, and picks the
gate threshold: the lowest false-reject rate whose false-accept rate stays under
--max-far (default 1%), because letting a stranger in is worse than asking a real
visitor to try again.

    python ml/evaluate.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from visitrack_ml import data  # noqa: E402
from visitrack_ml.config import ARTIFACTS_DIR, METRICS_FILE, MODEL_FILE  # noqa: E402
from visitrack_ml.model import load_siamese  # noqa: E402


def confusion(y, scores, t):
    pred = scores >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return tp, fp, tn, fn


def metrics_at(y, scores, t):
    tp, fp, tn, fn = confusion(y, scores, t)
    total = tp + fp + tn + fn
    return {
        "threshold": round(float(t), 3),
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "far": fp / (fp + tn) if fp + tn else 0.0,   # strangers accepted
        "frr": fn / (fn + tp) if fn + tp else 0.0,   # real people rejected
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def update_metrics_file(section: str, values: dict, path: Path = METRICS_FILE):
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[section] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODEL_FILE))
    p.add_argument("--pairs", default=str(ARTIFACTS_DIR / "pairs_test.csv"))
    p.add_argument("--max-far", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    pairs = data.load_pairs(Path(args.pairs))
    print(f"Test set: {data.summarize(pairs)}")
    model = load_siamese(args.model)

    ds = data.make_dataset(pairs, args.batch_size, cache=False)
    scores = np.concatenate([model.predict_on_batch(x).reshape(-1) for x, _ in ds])
    y = np.array([p.label for p in pairs])

    sweep = [metrics_at(y, scores, t) for t in np.linspace(0.01, 0.99, 99)]
    eer_row = min(sweep, key=lambda r: abs(r["far"] - r["frr"]))
    ok = [r for r in sweep if r["far"] <= args.max_far]
    chosen = min(ok, key=lambda r: r["frr"]) if ok else eer_row

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = None

    result = {
        "test_pairs": len(pairs),
        "roc_auc": auc,
        "at_0.5": metrics_at(y, scores, 0.5),
        "eer": round((eer_row["far"] + eer_row["frr"]) / 2, 4),
        "eer_threshold": eer_row["threshold"],
        "gate_threshold": chosen["threshold"],
        "at_gate_threshold": chosen,
        "max_far_target": args.max_far,
    }
    update_metrics_file("recognition", result)
    print(json.dumps(result, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        far = [r["far"] for r in sweep]
        tpr = [1 - r["frr"] for r in sweep]
        plt.figure(figsize=(5, 5))
        plt.plot(far, tpr, marker=".")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.xlabel("False accept rate")
        plt.ylabel("True accept rate")
        plt.title(f"ROC (AUC={auc:.3f})" if auc else "ROC")
        plt.savefig(ARTIFACTS_DIR / "roc.png", dpi=120, bbox_inches="tight")
        print(f"Saved {ARTIFACTS_DIR / 'roc.png'}")
    except Exception as e:  # plotting is optional
        print(f"(ROC plot skipped: {e})")


if __name__ == "__main__":
    main()
