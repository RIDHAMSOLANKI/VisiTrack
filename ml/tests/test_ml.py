import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")
tf = pytest.importorskip("tensorflow")

from visitrack_ml import data  # noqa: E402
from visitrack_ml.model import L1Dist, build_siamese, load_siamese, score_embeddings, split_model  # noqa: E402


def make_fake_lfw(root: Path, people: int = 12, per_person: int = 4, size: int = 250):
    rng = np.random.default_rng(0)
    for p in range(people):
        d = root / f"Person_{p:02d}"
        d.mkdir(parents=True)
        for i in range(per_person if p % 3 else 1):  # every 3rd person has one photo
            img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"Person_{p:02d}_{i:04d}.jpg"), img)


def test_l1dist_is_absolute_difference():
    a = tf.constant([[1.0, -2.0, 3.0]])
    b = tf.constant([[0.5, 2.0, 3.0]])
    np.testing.assert_allclose(L1Dist()(a, b).numpy(), [[0.5, 4.0, 0.0]])


def test_siamese_shapes_and_shared_weights():
    model = build_siamese(embedding_dim=16)
    x = np.random.rand(2, 100, 100, 3).astype("float32")
    out = model([x, x]).numpy()
    assert out.shape == (2, 1)
    assert np.all((out >= 0) & (out <= 1))
    # One embedding sub-model is reused for both inputs.
    assert sum(1 for layer in model.layers if layer.name == "embedding") == 1


def test_save_load_and_split_head_matches_full_model(tmp_path):
    model = build_siamese(embedding_dim=16)
    path = tmp_path / "m.keras"
    model.save(path)
    loaded = load_siamese(path)

    a = np.random.rand(3, 100, 100, 3).astype("float32")
    b = np.random.rand(3, 100, 100, 3).astype("float32")
    full = loaded.predict_on_batch([a, b]).reshape(-1)

    emb, clf = split_model(loaded)
    ea, eb = emb.predict_on_batch(a), emb.predict_on_batch(b)
    via_head = np.array([score_embeddings(clf, ea[i], eb[i:i + 1])[0] for i in range(3)])
    np.testing.assert_allclose(full, via_head, rtol=1e-4, atol=1e-5)


def test_pairs_balanced_and_people_disjoint(tmp_path):
    make_fake_lfw(tmp_path)
    splits = data.make_splits(tmp_path, seed=1, val_frac=0.25, test_frac=0.25, max_pos_per_person=3)
    people_in = {}
    for name, pairs in splits.items():
        assert pairs, name
        pos = sum(p.label for p in pairs)
        assert pos == len(pairs) - pos  # equal same / different
        people_in[name] = {Path(x).parent.name for p in pairs for x in (p.path_a, p.path_b)}
    assert not people_in["train"] & people_in["test"]
    assert not people_in["train"] & people_in["val"]


def test_dataset_yields_cropped_resized_batches(tmp_path):
    make_fake_lfw(tmp_path, people=6)
    pairs = data.make_splits(tmp_path, seed=0, val_frac=0, test_frac=0)["train"]
    ds = data.make_dataset(pairs, batch_size=4, training=True)
    (a, b), y = next(iter(ds))
    assert a.shape[1:] == (100, 100, 3) and b.shape[1:] == (100, 100, 3)
    assert float(tf.reduce_max(a)) <= 1.0 and float(tf.reduce_min(a)) >= 0.0
    assert set(np.unique(y.numpy())) <= {0.0, 1.0}


def test_pairs_csv_roundtrip(tmp_path):
    make_fake_lfw(tmp_path / "lfw", people=6)
    pairs = data.make_splits(tmp_path / "lfw", seed=0, val_frac=0, test_frac=0)["train"]
    data.save_pairs(pairs, tmp_path / "p.csv")
    assert data.load_pairs(tmp_path / "p.csv") == pairs
