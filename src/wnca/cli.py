"""wnca command-line entry point.

    wnca train    --config configs/phase2a_data.yaml
    wnca eval     --config configs/phase2a_data.yaml --checkpoint runs/.../best_*.pt
    wnca cache    --config configs/base.yaml
    wnca mesh     --config configs/base.yaml
    wnca benchmark --config configs/base.yaml

`--smoke` layers the smoke overrides onto any of these; `--set a.b=c` overrides a single field.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .config import load_config, pick_device


def _parse_sets(pairs: list[str]) -> dict:
    """`--set train.lr=1e-4 model.n_substeps=30` -> nested dict."""
    out: dict = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        key, val = p.split("=", 1)
        node = out
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    return out


def _common(p: argparse.ArgumentParser):
    p.add_argument("--config", "-c", default=None, help="phase config; defaults to configs/base.yaml")
    p.add_argument("--smoke", action="store_true", help="tiny mesh + one month of data")
    p.add_argument("--device", default=None)
    p.add_argument("--set", nargs="*", dest="sets", default=[], metavar="KEY=VAL")


def cmd_mesh(args) -> int:
    from .mesh.icosphere import build_mesh, mean_spacing_km

    cfg = load_config(args.config, _parse_sets(args.sets), smoke=args.smoke)
    mesh = build_mesh(cfg)
    n = len(mesh["v"])
    print(f"nodes={n}  edges={len(mesh['edges'])}  mean spacing ~{mean_spacing_km(n):.0f} km")
    return 0


def cmd_cache(args) -> int:
    from .data.cache import build_cache
    from .mesh.icosphere import build_mesh

    cfg = load_config(args.config, _parse_sets(args.sets), smoke=args.smoke)
    mesh = build_mesh(cfg)
    cache = build_cache(cfg, mesh, force=args.force)
    for s in ("train", "val", "test"):
        print(f"  {s}: {cache.split(s).shape}")
    print(f"  channels: {len(cfg.variables.channels())}")
    return 0


def cmd_train(args) -> int:
    from .train.phases import run_phase

    cfg = load_config(args.config, _parse_sets(args.sets), smoke=args.smoke)
    run_phase(cfg, device=args.device, out_dir=args.out, resume=args.resume)
    return 0


def cmd_eval(args) -> int:
    from .eval.metrics import evaluate, format_scorecard
    from .eval.perturbation import perturbation_growth, summarize as pg_summary
    from .train.checkpoint import latest_checkpoint, load_checkpoint
    from .train.phases import setup

    cfg = load_config(args.config, _parse_sets(args.sets), smoke=args.smoke)
    mesh, cache, model, bands, device = setup(cfg, args.device)

    ckpt = args.checkpoint or latest_checkpoint(args.out or cfg.tracking.out_dir)
    if ckpt is None:
        print("no checkpoint found -- train first, or pass --checkpoint", file=sys.stderr)
        return 2
    blob = load_checkpoint(ckpt, model, cfg, map_location=device)
    print(f"loaded {Path(ckpt).name}  (epoch {blob.get('epoch')}, metric {blob.get('metric')})")

    sc = evaluate(model, cfg, cache, mesh, split=args.split, device=device)
    print()
    print(format_scorecard(sc, cfg, cache.normalizer, channel=args.channel))

    if cfg.c_phys > 1:
        from .eval.metrics import format_channel_summary, format_level_table

        print()
        print(format_level_table(sc, cfg, cache.normalizer, lead_hours=args.lead))
        print()
        print(format_channel_summary(sc, cfg, cache.normalizer))
        print("\n  specific humidity is reported in log units; d(log q) ~ dq/q, so 0.10 "
              "reads as roughly a 10% error in q.")

    print("\n--- perturbation growth (the direct diagnostic) ---")
    pg = perturbation_growth(model, cfg, cache, split=args.split, device=device,
                             n_windows=min(cfg.eval.max_windows, 20))
    print(pg_summary(pg))
    if cfg.c_phys > 1:
        from .eval.perturbation import format_per_channel

        print()
        print(format_per_channel(pg, cfg))

    if bands is not None and cfg.model.stochastic:
        from .eval.spectrum import member_spectra, summarize as sp_summary

        print("\n--- band energies, individual members ---")
        print(sp_summary(member_spectra(model, cfg, cache, bands, split=args.split,
                                        lead_windows=12, device=device), channel=args.channel))
    return 0


def cmd_benchmark(args) -> int:
    from .train.phases import setup

    cfg = load_config(args.config, _parse_sets(args.sets), smoke=args.smoke)
    mesh, cache, model, bands, device = setup(cfg, args.device)
    _benchmark(cfg, model, cache, device)
    return 0


def _benchmark(cfg, model, cache, device, n_iter: int = 5):
    """Measure a forecast step, forward and backward, and report memory.

    The dominant term is `M x n_substeps` network evaluations per forecast step. Put the
    measured number in the plan rather than trusting the estimate in it.
    """
    import time

    # Benchmark what training will actually do: deterministic phases run a single member.
    B = cfg.train.batch_size
    M = cfg.ensemble.m_train if cfg.model.stochastic else 1
    series = cache.split("train").array
    cur = torch.from_numpy(np.array(series[1 : 1 + B], dtype=np.float32)).to(device)
    prev = torch.from_numpy(np.array(series[0:B], dtype=np.float32)).to(device)
    st = torch.from_numpy(cache.static).float().to(device).unsqueeze(0).expand(B, -1, -1)

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def once():
        pred, ovf = model.rollout_ensemble(model.seed(cur), st, 1, prev_phys=prev,
                                           n_members=M, return_aux=True)
        loss = pred.pow(2).mean() + ovf
        loss.backward()
        model.zero_grad(set_to_none=True)

    once()  # warm up
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        once()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter

    evals = M * cfg.model.n_substeps
    print("\n--- benchmark: one forecast step, forward + backward ---")
    print(f"  batch {B} x members {M} x substeps {cfg.model.n_substeps} = {B * evals} MLP passes")
    print(f"  {dt * 1000:.1f} ms/step  ->  {B / dt:.1f} samples/s")
    if device == "cuda":
        print(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    n_train = len(cache.split("train")) - 2
    print(f"  one epoch over {n_train} samples: ~{n_train / max(B / dt, 1e-9) / 60:.1f} min")
    return dt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wnca", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("mesh", cmd_mesh), ("cache", cmd_cache), ("train", cmd_train),
                     ("eval", cmd_eval), ("benchmark", cmd_benchmark)):
        p = sub.add_parser(name)
        _common(p)
        p.set_defaults(func=fn)
        if name == "cache":
            p.add_argument("--force", action="store_true", help="rebuild from scratch")
        if name in ("train", "eval", "benchmark"):
            p.add_argument("--out", default=None, help="output directory")
        if name == "train":
            p.add_argument("--resume", default=None, metavar="PATH|auto",
                           help="resume a crashed run from a checkpoint, restoring "
                                "optimizer state, epoch and LR-schedule position")
        if name == "eval":
            p.add_argument("--checkpoint", default=None)
            p.add_argument("--split", default="test", choices=("train", "val", "test"))
            p.add_argument("--channel", default="geopotential_500")
            p.add_argument("--lead", type=int, default=24,
                           help="lead hours for the variable x level table")

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
