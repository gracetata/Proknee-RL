"""Dataset loader for full-muscle distillation with masked observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class FullMuscleMaskedDistillDataset:
    dataset_dir: str | Path | list[str | Path] | tuple[str | Path, ...]
    split: str = "train"
    val_fraction: float = 0.1
    seed: int = 0
    max_files: int | None = None
    max_frames: int | None = None
    load_into_memory: bool = True

    def __post_init__(self):
        self.dataset_dirs = [Path(p) for p in (self.dataset_dir if isinstance(self.dataset_dir, (list, tuple)) else [self.dataset_dir])]
        files = []
        for dataset_dir in self.dataset_dirs:
            files.extend(sorted(dataset_dir.glob("*.npz")))
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files:
            raise FileNotFoundError(f"No .npz rollout files found in {self.dataset_dirs}")
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(files))
        rng.shuffle(order)
        n_val = max(1, int(round(len(files) * self.val_fraction))) if len(files) > 1 else 0
        val_idx = set(order[:n_val].tolist())
        if self.split == "train":
            self.files = [f for i, f in enumerate(files) if i not in val_idx]
        elif self.split in {"val", "validation"}:
            self.files = [f for i, f in enumerate(files) if i in val_idx]
        elif self.split == "all":
            self.files = files
        else:
            raise ValueError(f"Unknown split={self.split!r}")
        if not self.files:
            self.files = files
        self._data = self._load_all() if self.load_into_memory else None

    def _load_all(self) -> dict[str, np.ndarray]:
        chunks = {"obs": [], "target_action": []}
        total = 0
        for path in self.files:
            with np.load(path, allow_pickle=True) as data:
                n = int(data["obs_student_masked"].shape[0])
                if self.max_frames is not None:
                    remaining = int(self.max_frames) - total
                    if remaining <= 0:
                        break
                    n = min(n, remaining)
                chunks["obs"].append(np.asarray(data["obs_student_masked"][:n], dtype=np.float32))
                chunks["target_action"].append(np.asarray(data["target_full_muscle_action"][:n], dtype=np.float32))
                total += n
        return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    @property
    def data(self) -> dict[str, np.ndarray]:
        if self._data is None:
            self._data = self._load_all()
        return self._data

    def __len__(self) -> int:
        return int(self.data["obs"].shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.data["obs"].shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.data["target_action"].shape[-1])

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(self.seed if seed is None else seed)
            rng.shuffle(indices)
        for start in range(0, n, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            if drop_last and batch_idx.shape[0] < int(batch_size):
                continue
            yield {key: value[batch_idx] for key, value in self.data.items()}
