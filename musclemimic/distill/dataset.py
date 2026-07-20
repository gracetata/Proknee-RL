"""Dataset loader for prosthesis distillation rollout files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class ProsthesisDistillDataset:
    dataset_dir: str | Path | list[str | Path] | tuple[str | Path, ...]
    split: str = "train"
    val_fraction: float = 0.1
    seed: int = 0
    max_files: int | None = None
    max_frames: int | None = None
    load_into_memory: bool = True

    def __post_init__(self):
        if isinstance(self.dataset_dir, list | tuple):
            dataset_dirs = [Path(p) for p in self.dataset_dir]
        else:
            dataset_dirs = [Path(self.dataset_dir)]
        files: list[Path] = []
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
            files.extend(sorted(dataset_dir.glob("*.npz")))
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files:
            raise FileNotFoundError(f"No .npz rollout files found in {', '.join(str(p) for p in dataset_dirs)}")
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
        self.dataset_dir = dataset_dirs[0]
        self._data = self._load_all() if self.load_into_memory else None

    def _load_all(self) -> dict[str, np.ndarray]:
        chunks = {"obs": [], "target_remaining": [], "target_prosthesis": [], "target_tau": [], "tau_pd": []}
        total = 0
        for path in self.files:
            with np.load(path, allow_pickle=True) as data:
                n = int(data["obs_student"].shape[0])
                if self.max_frames is not None:
                    remaining = int(self.max_frames) - total
                    if remaining <= 0:
                        break
                    n = min(n, remaining)
                chunks["obs"].append(np.asarray(data["obs_student"][:n], dtype=np.float32))
                if "target_remaining_muscle_action_force_equiv" in data:
                    rem = np.asarray(data["target_remaining_muscle_action_force_equiv"][:n], dtype=np.float32)
                else:
                    rem = np.asarray(data["target_remaining_muscle_action"][:n], dtype=np.float32)
                chunks["target_remaining"].append(rem)
                chunks["target_prosthesis"].append(np.asarray(data["target_prosthesis_action"][:n], dtype=np.float32))
                target_tau = np.asarray(data["target_prosthesis_tau"][:n], dtype=np.float32)
                chunks["target_tau"].append(target_tau)
                if "tau_pd" in data:
                    chunks["tau_pd"].append(np.asarray(data["tau_pd"][:n], dtype=np.float32))
                else:
                    chunks["tau_pd"].append(np.zeros_like(target_tau, dtype=np.float32))
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
    def n_remaining_muscles(self) -> int:
        return int(self.data["target_remaining"].shape[-1])

    def iter_batches(self, batch_size: int, *, shuffle: bool = True, drop_last: bool = False, seed: int | None = None) -> Iterator[dict[str, np.ndarray]]:
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


@dataclass
class SplitActionProsthesisDataset(ProsthesisDistillDataset):
    """Dataset for split-action prosthesis distillation.

    The file schema intentionally matches the existing prosthesis rollout files:
    obs_student, target_remaining_muscle_action, target_prosthesis_action, and
    target_prosthesis_tau. The only extension is that dataset_dir may be a list
    of rollout directories, which allows later DAgger data to be appended without
    changing old loader behavior.
    """

    dataset_dir: str | Path | list[str | Path] | tuple[str | Path, ...]

    def __post_init__(self):
        if isinstance(self.dataset_dir, list | tuple):
            dataset_dirs = [Path(p) for p in self.dataset_dir]
        else:
            dataset_dirs = [Path(self.dataset_dir)]
        self.dataset_dirs = dataset_dirs
        files: list[Path] = []
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
            files.extend(sorted(dataset_dir.glob("*.npz")))
        if self.max_files is not None:
            files = files[: int(self.max_files)]
        if not files:
            raise FileNotFoundError(f"No .npz rollout files found in {', '.join(str(p) for p in dataset_dirs)}")
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
        self.dataset_dir = dataset_dirs[0]
        self._data = self._load_all() if self.load_into_memory else None
