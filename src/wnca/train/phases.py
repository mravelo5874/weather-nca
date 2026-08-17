"""Per-phase setup and dispatch.

Assembles the objects a phase needs -- mesh, cache, model, band filters, tracker -- and hands
them to `loop.fit`. Every run writes its fully-resolved config next to its checkpoints, so
reproducing a run means pointing at that file rather than remembering what was edited.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, pick_device
from ..data.cache import build_cache
from ..mesh.icosphere import build_mesh, mean_spacing_km
from ..mesh.spectral import build_band_filters
from ..models.nca import build_model
from .checkpoint import warm_start
from .loop import fit


class Tracker:
    """Weights & Biases if enabled, a JSONL file either way.

    The JSONL is not a fallback -- it is the record that survives a run where wandb was off,
    offline, or rate-limited, which on spot instances is most of them.
    """

    def __init__(self, cfg: Config, out_dir: Path):
        self.path = Path(out_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = None
        if cfg.tracking.wandb:
            try:
                import wandb

                self.run = wandb.init(
                    project=cfg.tracking.project,
                    name=cfg.tracking.run_name or f"{cfg.phase}_{time.strftime('%Y%m%d_%H%M')}",
                    config=cfg.to_dict(),
                )
            except Exception as e:  # noqa: BLE001 - never let tracking kill a training run
                print(f"  wandb unavailable ({e}); logging to {self.path} only")

    def log(self, d: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: _jsonable(v) for k, v in d.items()}) + "\n")
        if self.run is not None:
            self.run.log(d)

    def finish(self):
        if self.run is not None:
            self.run.finish()


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def setup(cfg: Config, device: str | None = None, verbose: bool = True):
    """Build every shared artifact for a phase. Used by both training and evaluation."""
    device = device or pick_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    # Free accuracy-for-speed trade on Ampere and later; a no-op on the local 1660 Ti.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mesh = build_mesh(cfg, verbose=verbose)
    n = len(mesh["v"])
    if verbose:
        print(f"mesh: {n} nodes, {len(mesh['edges'])} edges, ~{mean_spacing_km(n):.0f} km spacing")

    cache = build_cache(cfg, mesh, verbose=verbose)
    model = build_model(cfg, mesh, device)

    bands = None
    if cfg.loss.w_spec > 0 or cfg.model.stochastic:
        bands = build_band_filters(mesh, cfg, device=device, cache_dir=cfg.data.cache_dir)

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        film = getattr(model, "film_parameters", None) or getattr(model.update, "film_parameters", None)
        n_film = sum(p.numel() for p in film()) if film else 0
        note = f" ({total - n_film:,} + {n_film:,} FiLM, inactive)" if not cfg.model.stochastic and n_film else ""
        print(f"model: {cfg.model.kind}, {total:,} params{note}, device {device}")
    return mesh, cache, model, bands, device


def run_phase(cfg: Config, device: str | None = None, out_dir: str | Path | None = None) -> dict:
    """Train one phase end to end."""
    mesh, cache, model, bands, device = setup(cfg, device)

    out_dir = Path(out_dir or Path(cfg.tracking.out_dir) / f"{cfg.phase}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump_yaml(out_dir / "config.resolved.yaml")

    if cfg.train.warm_start:
        warm_start(cfg.train.warm_start, model, cfg, map_location=device)

    tracker = Tracker(cfg, out_dir)
    try:
        result = fit(cfg, model, mesh, cache, device, out_dir, bands=bands, tracker=tracker)
    finally:
        tracker.finish()

    (out_dir / "result.json").write_text(
        json.dumps({k: _jsonable(v) for k, v in result.items() if k != "history"}, indent=1),
        encoding="utf-8",
    )
    print(f"\nphase {cfg.phase} done | best selection metric {result['best']:.5f}")
    print(f"artifacts -> {out_dir}")
    return result
