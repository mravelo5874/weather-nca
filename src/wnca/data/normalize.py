"""Per-variable, per-level normalization stats; specific-humidity log transform.

Two hard constraints, both from the plan and both easy to violate silently:

1. Statistics come from the **train split only**. Fitting on val or test leaks.
2. Specific humidity spans orders of magnitude across levels. Without the log transform the
   850 hPa channel dominates every gradient, and the 250 hPa channel contributes nothing.

Stats are keyed by `Channel.key`, so adding a level or reordering `variables` invalidates
nothing silently -- the keys either match or the load raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config


class Normalizer:
    """Per-channel affine normalization with an optional preceding log transform."""

    def __init__(self, keys: tuple[str, ...], mean: np.ndarray, std: np.ndarray,
                 log_mask: np.ndarray, log_offset: float = 1e-9):
        self.keys = tuple(keys)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.log_mask = np.asarray(log_mask, dtype=bool)
        self.log_offset = float(log_offset)
        if not (len(self.keys) == len(self.mean) == len(self.std) == len(self.log_mask)):
            raise ValueError("normalizer arrays disagree on channel count")

    # ---- transform ----
    def _pre(self, x: np.ndarray) -> np.ndarray:
        if not self.log_mask.any():
            return x
        out = x.copy()
        m = self.log_mask
        out[..., m] = np.log(np.maximum(out[..., m], 0.0) + self.log_offset)
        return out

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Physical units -> normalized. Last axis is channels."""
        return ((self._pre(x) - self.mean) / self.std).astype(np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """Normalized -> physical units. Inverse of `encode`."""
        out = np.asarray(x, dtype=np.float32) * self.std + self.mean
        if self.log_mask.any():
            out = out.copy()
            m = self.log_mask
            out[..., m] = np.exp(out[..., m]) - self.log_offset
        return out

    def decode_scale(self, channel: int) -> float:
        """Multiplicative factor taking a normalized *difference* to physical units.

        Only meaningful for channels without a log transform -- a difference in log space has
        no single physical scale, so RMSE for those channels is reported in log units and the
        scorecard says so.
        """
        return float(self.std[channel])

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "keys": list(self.keys),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "log_mask": self.log_mask.tolist(),
            "log_offset": self.log_offset,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["keys"]), np.array(d["mean"]), np.array(d["std"]),
                   np.array(d["log_mask"]), d.get("log_offset", 1e-9))

    def assert_matches(self, cfg: Config) -> None:
        want = tuple(c.key for c in cfg.variables.channels())
        if self.keys != want:
            raise ValueError(
                f"normalizer channels do not match config.\n  stats: {self.keys}\n  config: {want}"
            )


def fit_normalizer(train: np.ndarray, cfg: Config) -> Normalizer:
    """Fit per-channel statistics on the TRAIN split only.

    `train` is [T, N, C] in raw physical units. Statistics are pooled over time and nodes,
    which is the same convention M1 used for its single channel.
    """
    channels = cfg.variables.channels()
    if train.shape[-1] != len(channels):
        raise ValueError(f"train has {train.shape[-1]} channels, config declares {len(channels)}")

    log_mask = np.array([c.name in cfg.variables.log_transform for c in channels], dtype=bool)
    x = np.asarray(train, dtype=np.float64)
    if log_mask.any():
        x = x.copy()
        x[..., log_mask] = np.log(np.maximum(x[..., log_mask], 0.0) + cfg.variables.log_offset)

    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    degenerate = std < 1e-12
    if degenerate.any():
        bad = [channels[i].key for i in np.nonzero(degenerate)[0]]
        raise ValueError(f"channels are constant on the train split, cannot normalize: {bad}")

    return Normalizer(
        tuple(c.key for c in channels), mean.astype(np.float32), std.astype(np.float32),
        log_mask, cfg.variables.log_offset,
    )
