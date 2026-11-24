"""Stores each registered visitor's face embeddings on disk.

Embeddings are computed ONCE when a visitor registers. At check-in we only embed
the live face and run the small L1 + Dense head against these stored vectors,
instead of pushing every stored photo through the CNN again.
Raw photos are not kept here: an embedding is enough to compare faces.
"""

import re
import threading
from pathlib import Path

import numpy as np

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Gallery:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, np.ndarray] = {}
        for f in self.folder.glob("*.npy"):
            self._items[f.stem] = np.load(f)

    @staticmethod
    def check_id(visitor_id: str) -> str:
        if not isinstance(visitor_id, str) or not _SAFE_ID.match(visitor_id):
            raise ValueError("visitor_id must be 1-64 letters, digits, _ or -")
        return visitor_id

    def add(self, visitor_id: str, embeddings: np.ndarray, replace: bool = False) -> int:
        visitor_id = self.check_id(visitor_id)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        with self._lock:
            if not replace and visitor_id in self._items:
                embeddings = np.concatenate([self._items[visitor_id], embeddings])
            self._items[visitor_id] = embeddings
            np.save(self.folder / f"{visitor_id}.npy", embeddings)
            return len(embeddings)

    def remove(self, visitor_id: str) -> bool:
        visitor_id = self.check_id(visitor_id)
        with self._lock:
            existed = self._items.pop(visitor_id, None) is not None
            (self.folder / f"{visitor_id}.npy").unlink(missing_ok=True)
            return existed

    def items(self):
        with self._lock:
            return list(self._items.items())

    def __len__(self):
        return len(self._items)
