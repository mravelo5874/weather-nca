#!/usr/bin/env python
"""Split the validation loss into its clean and pushforward halves, per arm.

Phase 2d's control shows an inverted train/val relationship against the NCA: the NCA's val loss
sits BELOW its train loss (it trains on harder self-generated states and validates on clean
single steps), while the control's sits ~4.5x ABOVE. That has at least two readings, which imply
different fixes:

  (a) it falls apart on rollout -- its own generated states are far off-distribution; or
  (b) it is simply over-regularised / overfitting on the clean single-step val, plausibly
      because `weight_decay: 0.1` was derived to stop the NCA's weight-norm ratchet and the
      control has no ratchet to stop.

The discriminator is cheap: evaluate the same checkpoint two ways on the same val batches.

  val_clean  -- one step from the ERA5 analysis state   (what `run_epoch(train=False)` reports)
  val_push   -- one step from the model's OWN stepped state, exactly as training does

Reading:
  * (a) rollout instability  ->  val_push >> val_clean, and the ratio is much worse for the
    control than for the NCA.
  * (b) regularisation/overfit ->  both are elevated together and the ratio is similar across
    arms; the level, not the gap, is what differs.

    python scripts/diag_val_split.py -c configs/phase2d_control.yaml --checkpoint <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wnca.config import load_config  # noqa: E402
from wnca.data.dataset import make_loader  # noqa: E402
from wnca.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from wnca.train.loop import Batch, Trainer  # noqa: E402
from wnca.train.phases import setup  # noqa: E402


@torch.no_grad()
def val_loss(tr: Trainer, loader, *, pushforward: bool, max_batches: int) -> float:
    """Mean clean field loss over `loader`.

    `pushforward=True` passes `train=True` into `_forward`, which is what gates the extra
    unrolled window -- under `no_grad`, so no graph is built and nothing updates. That is the
    whole trick: same weights, same batches, only the states the model is scored FROM differ.
    """
    tr.split = "val"
    tr.model.eval()
    acc, n = 0.0, 0
    m = tr.cfg.ensemble.m_val
    for i, (prev, cur, tgt, idx) in enumerate(loader):
        if i >= max_batches:
            break
        b = Batch(prev.to(tr.device), cur.to(tr.device), tgt.to(tr.device), idx)
        with torch.autocast(tr.amp_device, dtype=tr.amp_dtype, enabled=tr.use_amp):
            pred, ovf, off = tr._forward(b, 1, m, pushforward)
            _, clean, _ = tr._loss(pred.float(), b.tgt[:, off:], ovf.float())
        if np.isfinite(clean):
            acc += clean
            n += 1
    return acc / max(n, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny mesh + synthetic data")
    ap.add_argument("--set", nargs="*", dest="sets", default=[], metavar="KEY=VAL")
    a = ap.parse_args(argv)

    overrides = {}
    for pair in a.sets:
        key, val = pair.split("=", 1)
        node = overrides
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    cfg = load_config(a.config, overrides, smoke=a.smoke)
    mesh, cache, model, bands, device = setup(cfg, a.device)
    ckpt = a.checkpoint or latest_checkpoint(cfg.tracking.out_dir)
    if ckpt is None:
        print("no checkpoint -- pass --checkpoint", file=sys.stderr)
        return 2
    blob = load_checkpoint(ckpt, model, cfg, map_location=device)
    tr = Trainer(cfg, model, mesh, cache, device, bands)

    pf = 1 if cfg.train.pushforward else 0
    loader = make_loader(cache, "val", cfg, n_out=1 + pf, shuffle=False)
    clean = val_loss(tr, loader, pushforward=False, max_batches=a.batches)
    push = val_loss(tr, loader, pushforward=True, max_batches=a.batches)

    print(f"\nconfig      {a.config}")
    print(f"checkpoint  {Path(ckpt).name}  (epoch {blob.get('epoch')})")
    print(f"batches     {a.batches}\n")
    print(f"  val_clean (from analysis)      {clean:.6f}")
    print(f"  val_push  (from own state)     {push:.6f}")
    ratio = push / clean if clean else float("nan")
    print(f"  ratio push/clean               {ratio:.2f}x")
    print("\n  A large ratio means the model's own states are far off-distribution -- rollout")
    print("  instability. A ratio near 1 with both values elevated points at regularisation or")
    print("  overfitting instead. Compare the RATIO across arms, not the level.")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"config": a.config, "checkpoint": Path(ckpt).name, "epoch": blob.get("epoch"),
             "val_clean": clean, "val_push": push, "ratio": ratio}, indent=1), encoding="utf-8")
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
