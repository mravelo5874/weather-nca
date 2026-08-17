"""torch Dataset / DataLoader over the mesh-projected cache.

Second-order Markov conditioning means each sample is a triple: (x_{t-1}, x_t) -> x_{t+1..t+k}.
`n_out` is how many 6 h windows are supervised -- 1 for single-step training, more for the
selection metric's rollout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..config import Config
from .cache import MeshCache


class WeatherSeq(Dataset):
    """Yields (prev, cur, targets) with targets covering `n_out` consecutive 6 h windows.

    Shapes: prev/cur [N, C], targets [n_out, N, C]. The backing array is a read-only memmap,
    so a worker reads only the timesteps it needs rather than materializing the split.
    """

    def __init__(self, series: np.ndarray, n_out: int = 1):
        self.x = series
        self.n_out = int(n_out)
        if len(self.x) < self.n_out + 2:
            raise ValueError(
                f"split has {len(self.x)} timesteps, need at least {self.n_out + 2} for n_out={n_out}"
            )

    def __len__(self) -> int:
        return len(self.x) - self.n_out - 1

    def __getitem__(self, i: int):
        # np.array (not asarray) so each sample is a writable copy rather than a memmap view.
        prev = torch.from_numpy(np.array(self.x[i], dtype=np.float32))
        cur = torch.from_numpy(np.array(self.x[i + 1], dtype=np.float32))
        tgt = torch.from_numpy(np.array(self.x[i + 2 : i + 2 + self.n_out], dtype=np.float32))
        return prev, cur, tgt


def evenly_spaced_subset(dataset: Dataset, fraction: float) -> Dataset:
    """Deterministic evenly-spaced subset.

    Deterministic is the point: the selection metric must be computed on the *same* start times
    every epoch, or "did this epoch improve?" stops meaning anything -- the same class of bug as
    M1's incommensurable checkpoint metrics. Evenly spaced rather than random so the subset
    still covers the whole seasonal cycle of the validation year.
    """
    n = len(dataset)
    if not 0 < fraction <= 1.0:
        raise ValueError(f"subset fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return dataset
    k = max(1, int(round(n * fraction)))
    idx = np.unique(np.linspace(0, n - 1, k).astype(int))
    return Subset(dataset, idx.tolist())


def make_loader(
    cache: MeshCache,
    split: str,
    cfg: Config,
    n_out: int = 1,
    shuffle: bool | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    subsample: float = 1.0,
) -> DataLoader:
    shuffle = (split == "train") if shuffle is None else shuffle
    ds = evenly_spaced_subset(WeatherSeq(cache.split(split).array, n_out), subsample)
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
