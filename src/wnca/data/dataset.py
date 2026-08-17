"""torch Dataset / DataLoader over the mesh-projected cache.

Second-order Markov conditioning means each sample is a triple: (x_{t-1}, x_t) -> x_{t+1..t+k}.
`n_out` is how many 6 h windows are supervised -- 1 for single-step training, more for the
selection metric's rollout.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


def make_loader(
    cache: MeshCache,
    split: str,
    cfg: Config,
    n_out: int = 1,
    shuffle: bool | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    shuffle = (split == "train") if shuffle is None else shuffle
    return DataLoader(
        WeatherSeq(cache.split(split).array, n_out),
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
