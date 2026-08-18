#!/usr/bin/env bash
# Is the cache itself sound? Run when training produces non-finite gradients from step one.
#
# The 2-year cache was verified locally (per-channel mean ~0, std ~1). The 39-year cache never
# was -- it was built on the instance and used immediately.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
python3 - <<'PY'
import glob, json
import numpy as np

root = sorted(glob.glob("wnca_cache/era5_sub5_c28_*"))[-1]
man = json.load(open(f"{root}/manifest.json"))
print(f"cache: {root}")
print(f"normalized: {man['normalized']}  dtype: {man['dtype']}")
print(f"norm_progress: {man.get('norm_progress')}")
print()

norm = json.load(open(f"{root}/normalizer.json"))
mu, sd = np.array(norm["mean"]), np.array(norm["std"])
print(f"normalizer: finite mean={np.isfinite(mu).all()} finite std={np.isfinite(sd).all()}")
print(f"  std range {sd.min():.4g} .. {sd.max():.4g}   (a near-zero std would blow up encode)")
bad = [i for i in range(len(sd)) if not np.isfinite(sd[i]) or sd[i] < 1e-6]
print(f"  degenerate channels: {bad if bad else 'none'}")
print()

for split in ("train", "val", "test"):
    shape = tuple(man["shapes"][split])
    a = np.memmap(f"{root}/{split}.dat", dtype=man["dtype"], mode="r", shape=shape)
    # stride so this is seconds, not minutes, on a 65 GB file
    step = max(1, shape[0] // 400)
    s = np.asarray(a[::step], dtype=np.float32)
    finite = np.isfinite(s)
    n_bad = int((~finite).sum())
    print(f"{split:>6}: {shape[0]:>6} steps | sampled {s.shape[0]:>4} "
          f"| non-finite {n_bad:>8} | min {np.nanmin(s):>9.2f} max {np.nanmax(s):>9.2f} "
          f"| mean {np.nanmean(s):>7.3f} std {np.nanstd(s):>6.3f}")
    if n_bad:
        t_bad = np.unique(np.where(~finite)[0])
        print(f"         non-finite in {len(t_bad)} sampled timesteps, first at index {t_bad[0]*step}")
    # per-channel extremes -- a single wild channel is the usual culprit
    cmax = np.nanmax(np.abs(s), axis=(0, 1))
    worst = np.argsort(-cmax)[:4]
    keys = man["channels"]
    print(f"         largest |value| by channel: " +
          "  ".join(f"{keys[i]}={cmax[i]:.1f}" for i in worst))
print()
print("normalized data should be roughly mean 0, std 1, |max| under ~20.")
PY
