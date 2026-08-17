"""Memory-mapped mesh-tensor cache, resumable.

Recomputing the regrid every epoch on a spot instance is the single biggest avoidable time
sink in the milestone, so mesh-projected tensors are cached to local disk as a memmap and
opened read-only by the dataset.

The cache is resumable independently of training state: progress is recorded per split in
timesteps, so a preempted build restarts at a chunk boundary rather than from zero. Sizing,
for the record -- 39 years, 6-hourly, 10242 nodes, 28 channels, float32 is about 65 GB; the
2-year phase-2b split is about 3.3 GB. `data.cache_dtype: float16` halves both.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from . import era5 as era5_mod
from .normalize import Normalizer, fit_normalizer

CHUNK_STEPS = 256
SPLITS = ("train", "val", "test")


def cache_tag(cfg: Config) -> str:
    """Identity of a cache: anything that changes the bytes on disk."""
    payload = {
        "source": cfg.data.source,
        "res": cfg.data.wb2_res if cfg.data.source == "era5" else "synthetic",
        "n_sub": cfg.mesh.n_sub,
        "channels": [c.key for c in cfg.variables.channels()],
        "log": sorted(cfg.variables.log_transform),
        "years": {s: list(getattr(cfg.data, f"{s}_years")) for s in SPLITS},
        "max_steps": cfg.data.max_steps_per_split,
        "dtype": cfg.data.cache_dtype,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    src = "era5" if cfg.data.source == "era5" else "synth"
    return f"{src}_sub{cfg.mesh.n_sub}_c{cfg.c_phys}_{h}"


@dataclass
class SplitView:
    """Read-only normalized view of one split: [T, N, C]."""

    array: np.memmap
    times: np.ndarray | None

    def __len__(self) -> int:
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape


class MeshCache:
    """Built artifacts for one (config, mesh) pair: normalized splits, static, normalizer."""

    def __init__(self, root: Path, cfg: Config):
        self.root = Path(root)
        self.cfg = cfg
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.normalizer = Normalizer.load(self.root / "normalizer.json")
        self.normalizer.assert_matches(cfg)
        self.static = np.load(self.root / "static.npy")

    def _path(self, split: str) -> Path:
        return self.root / f"{split}.dat"

    def split(self, name: str) -> SplitView:
        if name not in SPLITS:
            raise KeyError(f"unknown split {name!r}")
        shape = tuple(self.manifest["shapes"][name])
        arr = np.memmap(self._path(name), dtype=np.dtype(self.manifest["dtype"]), mode="r", shape=shape)
        tp = self.root / f"times_{name}.npy"
        return SplitView(arr, np.load(tp, allow_pickle=True) if tp.exists() else None)

    def climatology(self) -> np.ndarray:
        """Train-split time mean, [N, C], normalized units. The baseline comes from train only."""
        p = self.root / "climatology.npy"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- rebuild the cache")
        return np.load(p)


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------


def _write_manifest(root: Path, man: dict) -> None:
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    tmp.replace(root / "manifest.json")


def _split_lengths(cfg: Config, ds) -> dict[str, int]:
    out = {}
    for s in SPLITS:
        years = tuple(getattr(cfg.data, f"{s}_years"))
        if cfg.data.source == "era5":
            sub = era5_mod.select_split(cfg, ds, years)
            out[s] = int(sub.sizes["time"])
        else:
            per_year = 4 * 365
            n = per_year * len(years)
            out[s] = min(n, cfg.data.max_steps_per_split or n)
    return out


def build_cache(cfg: Config, mesh: dict[str, np.ndarray], force: bool = False,
                verbose: bool = True) -> MeshCache:
    """Build (or resume, or reuse) the mesh-projected cache for this config.

    Raw fields are written first, then the normalizer is fitted on the completed train split
    and applied in place. Interrupting mid-normalize is safe: the manifest only flips
    `normalized` to true once every split has been converted.
    """
    root = cfg.cache_dir / cache_tag(cfg)
    root.mkdir(parents=True, exist_ok=True)
    man_path = root / "manifest.json"

    if force and man_path.exists():
        man_path.unlink()

    ds = era5_mod.open_wb2(cfg) if cfg.data.source == "era5" else None
    dtype = np.dtype(cfg.data.cache_dtype)
    N, C = len(mesh["v"]), cfg.c_phys

    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
    else:
        lengths = _split_lengths(cfg, ds)
        man = {
            "tag": cache_tag(cfg),
            "dtype": dtype.name,
            "shapes": {s: [lengths[s], N, C] for s in SPLITS},
            "progress": {s: 0 for s in SPLITS},
            "normalized": False,
            "channels": [c.key for c in cfg.variables.channels()],
        }
        _write_manifest(root, man)

    if man["normalized"]:
        if verbose:
            print(f"cache ready: {root}")
        return MeshCache(root, cfg)

    # ---- static conditioning ----
    static_p = root / "static.npy"
    if not static_p.exists():
        np.save(static_p, era5_mod.load_static(cfg, mesh, ds))

    # ---- raw fields, chunk by chunk ----
    idx = w = None
    if cfg.data.source == "era5":
        idx, w = era5_mod.mesh_weights(cfg, mesh, ds)

    for split in SPLITS:
        T = man["shapes"][split][0]
        done = man["progress"][split]
        if done >= T:
            continue
        path = root / f"{split}.dat"
        arr = np.memmap(path, dtype=dtype, mode="r+" if path.exists() else "w+", shape=(T, N, C))
        years = tuple(getattr(cfg.data, f"{split}_years"))

        if cfg.data.source == "era5":
            sub = era5_mod.select_split(cfg, ds, years)
            np.save(root / f"times_{split}.npy", sub.time.values)
            buf = np.empty((min(CHUNK_STEPS, T), N, C), dtype=np.float32)
            while done < T:
                t1 = min(done + CHUNK_STEPS, T)
                era5_mod.regrid_chunk(cfg, sub, mesh, idx, w, done, t1, _Offset(buf, done))
                arr[done:t1] = buf[: t1 - done].astype(dtype)
                arr.flush()
                done = t1
                man["progress"][split] = done
                _write_manifest(root, man)
                if verbose:
                    print(f"  {split}: {done}/{T} timesteps", end="\r", flush=True)
        else:
            arr[:] = era5_mod.make_synthetic(cfg, mesh, T, seed=hash(split) % 2**31).astype(dtype)
            arr.flush()
            man["progress"][split] = T
            _write_manifest(root, man)

        del arr
        if verbose:
            print(f"  {split}: {T}/{T} timesteps  ")

    # ---- normalize in place, stats from train only ----
    #
    # This pass rewrites every byte of the cache and is the most dangerous part of the build:
    # normalization is not idempotent, so applying it twice to the same chunk silently corrupts
    # the data with no error and no obvious symptom. At 65 GB (phase 2c) the pass is long enough
    # that a spot preemption landing inside it is a real possibility, so progress is tracked per
    # split at chunk granularity, exactly like the streaming pass above.
    man.setdefault("norm_progress", {s: 0 for s in SPLITS})

    # The normalizer must be fitted ONCE, on raw data. Refitting on resume would read a
    # partially normalized train split and produce meaningless statistics -- which would then
    # be applied to the remaining chunks, leaving the cache internally inconsistent.
    norm_path = root / "normalizer.json"
    if norm_path.exists() and any(man["norm_progress"].values()):
        norm = Normalizer.load(norm_path)
        norm.assert_matches(cfg)
        if verbose:
            print("  resuming normalization with the previously fitted statistics")
    else:
        train = np.memmap(root / "train.dat", dtype=dtype, mode="r",
                          shape=tuple(man["shapes"]["train"]))
        stat_steps = min(len(train), 2000)  # strided sample; the full array is I/O bound
        sample = np.asarray(train[:: max(1, len(train) // stat_steps)], dtype=np.float32)
        norm = fit_normalizer(sample, cfg)
        norm.save(norm_path)
        del train, sample

    for split in SPLITS:
        shape = tuple(man["shapes"][split])
        done = man["norm_progress"][split]
        if done >= shape[0]:
            continue
        arr = np.memmap(root / f"{split}.dat", dtype=dtype, mode="r+", shape=shape)
        for t0 in range(done, shape[0], CHUNK_STEPS):
            t1 = min(t0 + CHUNK_STEPS, shape[0])
            arr[t0:t1] = norm.encode(np.asarray(arr[t0:t1], dtype=np.float32)).astype(dtype)
            arr.flush()
            man["norm_progress"][split] = t1
            _write_manifest(root, man)
        del arr
        if verbose:
            print(f"  normalized {split}: {shape[0]}/{shape[0]}")

    # Climatology comes from the TRAIN split only, after it is normalized.
    if not (root / "climatology.npy").exists():
        shape = tuple(man["shapes"]["train"])
        arr = np.memmap(root / "train.dat", dtype=dtype, mode="r", shape=shape)
        acc = np.zeros((shape[1], shape[2]), dtype=np.float64)
        for t0 in range(0, shape[0], CHUNK_STEPS):
            t1 = min(t0 + CHUNK_STEPS, shape[0])
            acc += np.asarray(arr[t0:t1], dtype=np.float64).sum(axis=0)
        np.save(root / "climatology.npy", (acc / shape[0]).astype(np.float32))
        del arr

    man["normalized"] = True
    _write_manifest(root, man)
    if verbose:
        print(f"cache built: {root}")
    return MeshCache(root, cfg)


class _Offset:
    """Shim letting `regrid_chunk` write absolute timesteps into a chunk-local buffer."""

    def __init__(self, buf: np.ndarray, offset: int):
        self.buf, self.offset = buf, offset

    def __setitem__(self, key, value):
        sl, rest = key[0], key[1:]
        self.buf[(slice(sl.start - self.offset, sl.stop - self.offset), *rest)] = value
