"""Timestamped, asserted, resumable checkpointing.

Every rule here is a regression test for something that already happened in M1:

* **Timestamped filenames.** `best.pt` as a single fixed path clobbered comparisons.
* **Architecture hash on load.** Three evaluation rounds ran on a zero-init identity model that
  reproduced persistence exactly, and the table looked plausible until someone noticed the
  skill column was 0.0% at every lead.
* **Non-zero head norm on load.** The direct check for that same failure -- an untrained model
  is exactly the identity map, and that is detectable in one line.
* **Finiteness guard.** On the gradient step and on checkpoint selection both.

Optimizer state, RNG state, epoch and step are all stored, so a preempted spot instance
resumes rather than restarting.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import Config

ARCH_KEYS = ("n_sub", "channels", "c_hidden", "c_static", "kind", "hidden_dim", "n_layers", "noise_dim")


def _head_norm(model: torch.nn.Module) -> float:
    """Norm of the output head, whatever the model calls it."""
    for attr in ("update.head", "decoder"):
        mod = model
        try:
            for part in attr.split("."):
                mod = getattr(mod, part)
            return float(mod.weight.norm().item())
        except AttributeError:
            continue
    raise AttributeError("model exposes neither update.head nor decoder; cannot check head norm")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    metric: float | None = None,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": model.state_dict(),
        "cfg": cfg.to_dict(),
        "arch_hash": cfg.arch_hash(),
        "head_norm": _head_norm(model),
        "epoch": epoch,
        "step": step,
        "metric": metric,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "extra": extra or {},
    }
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    return path


def timestamped_path(out_dir: str | Path, phase: str, tag: str = "best") -> Path:
    """`best.pt` as a fixed path clobbered M1 comparisons. Never reuse a filename."""
    return Path(out_dir) / f"{tag}_{phase}_{time.strftime('%Y%m%d_%H%M%S')}.pt"


def latest_checkpoint(out_dir: str | Path, pattern: str = "best_*.pt") -> Path | None:
    """Newest checkpoint under `out_dir`, searching per-run subdirectories too.

    Runs write into `out_dir/<phase>_<timestamp>/`, so a flat glob finds nothing.
    """
    root = Path(out_dir)
    if not root.exists():
        return None
    paths = sorted({*root.glob(pattern), *root.glob(f"*/{pattern}")}, key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
    strict_arch: bool = True,
    allow_untrained: bool = False,
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint, asserting it is the architecture we think it is and that it trained.

    `allow_untrained` exists only for `test_checkpoint.py`, which has to construct the failure
    in order to check the guard fires. Do not set it in a training or evaluation path.
    """
    path = Path(path)
    blob = torch.load(path, map_location=map_location, weights_only=False)

    saved_hash = blob.get("arch_hash")
    if strict_arch and saved_hash is not None and saved_hash != cfg.arch_hash():
        saved_cfg = blob.get("cfg", {})
        diff = _arch_diff(saved_cfg, cfg)
        raise RuntimeError(
            f"checkpoint/config architecture mismatch for {path.name}\n"
            f"  saved arch_hash {saved_hash}, live {cfg.arch_hash()}\n"
            f"  differing fields (saved -> live): {diff}"
        )

    model.load_state_dict(blob["model"])
    hn = _head_norm(model)
    if not allow_untrained and not hn > 0:
        raise RuntimeError(
            f"{path.name}: head weight norm is {hn} -- this checkpoint is an untrained "
            "(identity-map) model. It will reproduce persistence exactly and score 0.0% skill "
            "at every lead. This assert is a regression test for M1 incident 1; do not remove it."
        )
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])

    if restore_rng and blob.get("rng"):
        rng = blob["rng"]
        try:
            torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            if rng.get("numpy"):
                np.random.set_state(rng["numpy"])
        except (RuntimeError, ValueError, TypeError) as e:  # different device count, etc.
            print(f"  warning: could not restore RNG state ({e}); continuing with fresh RNG")

    return blob


def warm_start(path: str | Path, model: torch.nn.Module, cfg: Config, map_location="cpu") -> dict:
    """Initialize from a previous phase's weights, without its optimizer or RNG state.

    Phase 3a warm-starts from the best 2c checkpoint. Because the FiLM projection is
    zero-initialized, the stochastic model begins numerically identical to the deterministic
    one it came from; a mismatch here is loaded non-strictly so the new noise pathway can
    appear without invalidating the port.
    """
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    unexpected = [k for k in unexpected]
    if unexpected:
        raise RuntimeError(f"warm start from {path}: unexpected keys {unexpected}")
    if missing:
        print(f"  warm start: {len(missing)} new parameter tensors (expected for the FiLM pathway)")
    hn = _head_norm(model)
    if not hn > 0:
        raise RuntimeError(f"warm start source {Path(path).name} is an untrained model (head norm {hn})")
    print(f"  warm started from {Path(path).name} (head |W| = {hn:.4e})")
    return blob


def _arch_diff(saved: dict, cfg: Config) -> dict:
    live = cfg.to_dict()
    out = {}
    for section in ("mesh", "variables", "state", "model"):
        s, l = saved.get(section, {}), live.get(section, {})
        for k in set(s) | set(l):
            if s.get(k) != l.get(k):
                out[f"{section}.{k}"] = (s.get(k), l.get(k))
    return out


def assert_finite(value: float, what: str) -> bool:
    """Finiteness guard for the gradient step and for checkpoint selection."""
    ok = bool(np.isfinite(value))
    if not ok:
        print(f"  non-finite {what} ({value}) -- skipping")
    return ok
