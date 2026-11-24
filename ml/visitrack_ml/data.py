"""Dataset: LFW download, identity-disjoint splits, pair building and the tf.data pipeline.

Pairs
-----
(anchor, positive)  -> label 1   two photos of the same person
(anchor, negative)  -> label 0   photos of two different people
Positives and negatives are kept in equal numbers so the model cannot score
well by always answering "different".

Splits are made by PERSON, not by image, so nobody in the test set was seen
during training. Otherwise test accuracy would look better than it really is.
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from .config import DATA_DIR, IMG_SIZE, LFW_FACE_BOX

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- download

def find_or_download_lfw(lfw_dir: str | None = None) -> Path:
    """Return the folder holding LFW as <root>/<Person_Name>/<image>.jpg.

    If lfw_dir is not given, scikit-learn downloads LFW (about 230 MB) into data/.
    """
    if lfw_dir:
        root = Path(lfw_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"LFW folder not found: {root}")
        return root

    data_home = DATA_DIR / "sklearn"
    for name in ("lfw_funneled", "lfw"):
        candidate = data_home / "lfw_home" / name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate

    from sklearn.datasets import fetch_lfw_people

    print("Downloading LFW through scikit-learn (one time, ~230 MB)...")
    # Loading a tiny subset is enough to make scikit-learn download and extract everything.
    fetch_lfw_people(data_home=str(data_home), min_faces_per_person=100, resize=0.1)
    for name in ("lfw_funneled", "lfw"):
        candidate = data_home / "lfw_home" / name
        if candidate.is_dir():
            return candidate
    raise RuntimeError("LFW download finished but the image folder was not found")


def scan_people(root: Path) -> dict[str, list[str]]:
    """{person_name: [image paths]} for every sub-folder of root."""
    people: dict[str, list[str]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(str(p) for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if images:
            people[person_dir.name] = images
    return people


# --------------------------------------------------------------------------- pairs

@dataclass
class Pair:
    path_a: str
    path_b: str
    crop_a: bool  # True for raw 250x250 LFW photos that still need the face cut out
    crop_b: bool
    label: int


def split_people(names: list[str], seed: int, val_frac: float, test_frac: float):
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n_test = int(len(names) * test_frac)
    n_val = int(len(names) * val_frac)
    return {
        "test": names[:n_test],
        "val": names[n_test:n_test + n_val],
        "train": names[n_test + n_val:],
    }


def build_pairs(people: dict[str, list[str]], names: list[str], needs_crop: dict[str, bool],
                max_pos_per_person: int, seed: int) -> list[Pair]:
    rng = random.Random(seed)
    positives: list[Pair] = []

    for name in names:
        imgs = people[name]
        if len(imgs) < 2:
            continue
        combos = [(a, b) for i, a in enumerate(imgs) for b in imgs[i + 1:]]
        rng.shuffle(combos)
        c = needs_crop[name]
        for a, b in combos[:max_pos_per_person]:
            positives.append(Pair(a, b, c, c, 1))

    # Same number of negatives: random images of two different people.
    negatives: list[Pair] = []
    if len(names) >= 2:
        while len(negatives) < len(positives):
            n1, n2 = rng.sample(names, 2)
            negatives.append(Pair(rng.choice(people[n1]), rng.choice(people[n2]),
                                  needs_crop[n1], needs_crop[n2], 0))

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def make_splits(lfw_root: Path, extra_dir: Path | None = None, seed: int = 42,
                val_frac: float = 0.1, test_frac: float = 0.1,
                max_pos_per_person: int = 10, max_people: int | None = None):
    """Build train/val/test pair lists.

    extra_dir: optional folder of your own face crops (data/custom/<name>/*.jpg,
    made by scripts/collect_faces.py). Those people are added to the TRAIN split.
    """
    people = scan_people(lfw_root)
    if max_people:
        keep = sorted(people)[:max_people]
        people = {k: people[k] for k in keep}
    needs_crop = {name: True for name in people}

    splits = split_people(list(people), seed, val_frac, test_frac)

    if extra_dir and Path(extra_dir).is_dir():
        extra = scan_people(Path(extra_dir))
        for name, imgs in extra.items():
            key = f"custom::{name}"
            people[key] = imgs
            needs_crop[key] = False
            splits["train"].append(key)

    return {
        split: build_pairs(people, names, needs_crop, max_pos_per_person, seed + i)
        for i, (split, names) in enumerate(splits.items())
    }


def save_pairs(pairs: list[Pair], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_a", "path_b", "crop_a", "crop_b", "label"])
        for p in pairs:
            w.writerow([p.path_a, p.path_b, int(p.crop_a), int(p.crop_b), p.label])


def load_pairs(path: Path) -> list[Pair]:
    with open(path, newline="") as f:
        return [Pair(r["path_a"], r["path_b"], r["crop_a"] == "1", r["crop_b"] == "1", int(r["label"]))
                for r in csv.DictReader(f)]


def summarize(pairs: list[Pair]) -> str:
    pos = sum(p.label for p in pairs)
    return f"{len(pairs)} pairs ({pos} same, {len(pairs) - pos} different)"


# --------------------------------------------------------------------------- tf.data

def load_image(path, crop):
    """File path -> (100, 100, 3) float32 in 0..1. Cuts the face out of raw LFW photos."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    top, bottom, left, right = LFW_FACE_BOX
    img = tf.cond(crop, lambda: img[top:bottom, left:right, :], lambda: img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32) / 255.0


def augment(img):
    """Random light/flip changes so the model copes with gate lighting and pose."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 1.0)


def make_dataset(pairs: list[Pair], batch_size: int = 16, training: bool = False,
                 cache: bool | str = True) -> tf.data.Dataset:
    """Pairs -> batched ((img_a, img_b), label) dataset.

    Order matters:
      load + resize (slow) -> cache -> shuffle -> augment -> batch -> prefetch
    cache comes BEFORE the random augment, otherwise every epoch would replay
    the same "random" changes. prefetch lets the CPU prepare the next batch
    while the GPU trains on the current one.
    """
    ds = tf.data.Dataset.from_tensor_slices((
        [p.path_a for p in pairs], [p.path_b for p in pairs],
        [p.crop_a for p in pairs], [p.crop_b for p in pairs],
        [float(p.label) for p in pairs],
    ))
    ds = ds.map(lambda a, b, ca, cb, y: (load_image(a, ca), load_image(b, cb), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
    if training:
        ds = ds.shuffle(min(len(pairs), 10_000), reshuffle_each_iteration=True)
        ds = ds.map(lambda a, b, y: (augment(a), augment(b), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda a, b, y: ((a, b), y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
