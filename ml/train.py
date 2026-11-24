"""Train the Siamese network with a custom tf.GradientTape loop.

Example (full run, best on a GPU / Google Colab):
    python ml/train.py --epochs 20

Quick smoke run on a laptop CPU:
    python ml/train.py --epochs 1 --max-people 300 --embedding-dim 512
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from visitrack_ml import data  # noqa: E402
from visitrack_ml.config import ARTIFACTS_DIR, MODEL_FILE  # noqa: E402
from visitrack_ml.model import build_siamese  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lfw-dir", help="Existing LFW folder (<root>/<Person>/<img>.jpg). Downloads if omitted.")
    p.add_argument("--extra-dir", default="data/custom", help="Your own face crops, added to training")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embedding-dim", type=int, default=4096)
    p.add_argument("--max-pos-per-person", type=int, default=10)
    p.add_argument("--max-people", type=int, help="Use only the first N people (for quick runs)")
    p.add_argument("--patience", type=int, default=4, help="Early stopping: epochs without val-loss gain")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ARTIFACTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    lfw_root = data.find_or_download_lfw(args.lfw_dir)
    print(f"LFW folder: {lfw_root}")
    splits = data.make_splits(lfw_root, Path(args.extra_dir), seed=args.seed,
                              max_pos_per_person=args.max_pos_per_person, max_people=args.max_people)
    for name, pairs in splits.items():
        data.save_pairs(pairs, out / f"pairs_{name}.csv")
        print(f"{name:>5}: {data.summarize(pairs)}")

    train_ds = data.make_dataset(splits["train"], args.batch_size, training=True)
    val_ds = data.make_dataset(splits["val"], args.batch_size)

    # ---------------------------------------------------------------- model
    model = build_siamese(args.embedding_dim)
    model.summary()

    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(args.lr)

    train_loss = keras.metrics.Mean()
    val_loss = keras.metrics.Mean()
    val_metrics = {
        "accuracy": keras.metrics.BinaryAccuracy(),
        "precision": keras.metrics.Precision(),
        "recall": keras.metrics.Recall(),
    }

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, str(out / "checkpoints"), max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Resumed from {manager.latest_checkpoint}")

    @tf.function
    def train_step(x, y):
        # GradientTape records the forward pass so we can get d(loss)/d(weights).
        with tf.GradientTape() as tape:
            yhat = tf.reshape(model(x, training=True), [-1])
            loss = loss_fn(y, yhat)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x, y):
        yhat = tf.reshape(model(x, training=False), [-1])
        return loss_fn(y, yhat), yhat

    # ---------------------------------------------------------------- loop
    history = []
    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss.reset_state()
        for step, (x, y) in enumerate(train_ds, 1):
            train_loss.update_state(train_step(x, y))
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step} loss {train_loss.result():.4f}", flush=True)

        val_loss.reset_state()
        for m in val_metrics.values():
            m.reset_state()
        for x, y in val_ds:
            loss, yhat = val_step(x, y)
            val_loss.update_state(loss)
            for m in val_metrics.values():
                m.update_state(y, yhat)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.result()),
            "val_loss": float(val_loss.result()),
            **{f"val_{k}": float(m.result()) for k, m in val_metrics.items()},
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        manager.save()

        if row["val_loss"] < best_val:
            best_val, bad_epochs = row["val_loss"], 0
            model.save(out / MODEL_FILE.name)
            print(f"  saved best model (val_loss {best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no val-loss improvement for {args.patience} epochs")
                break

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    print("Done. Next: python ml/evaluate.py")


if __name__ == "__main__":
    main()
