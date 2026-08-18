#!/usr/bin/env python
"""Training diagnostics: find out why a run produces non-finite gradients, cheaply.

Built after phase 2c failed three times on a cloud instance (fp16, bf16, fp32) with the same
symptom -- finite loss, `inf` gradient norms -- costing ~$9 and most of a day. Each attempt was
a multi-hour run that failed slowly. This does the same detective work in minutes.

The central design choice: **detect explosion by trend, not by waiting for `inf`.** The real
failure took ~2,000 steps to appear, far too slow to probe directly at several settings. But
gradient norms grow steadily before they overflow, so a few hundred steps is enough to see
where a configuration is heading.

Stages (each independently selectable, roughly ordered by cost):

    env       GPU, torch, precision support                       seconds
    cache     non-finite scan, per-channel stats, outlier steps   ~1 min
    numerics  perception magnitudes; forward+backward per dtype   ~1 min
    substeps  how gradient norm scales with n_substeps            ~2 min
    lr        LR sweep with early explosion detection             ~15 min
    matrix    amp x pushforward x solar_forcing, all finite       ~3 min

    python scripts/diagnose.py -c configs/phase2c_full.yaml --stages all
    python scripts/diagnose.py -c configs/phase2c_full.yaml --stages lr --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from wnca.config import load_config
from wnca.data.dataset import make_loader
from wnca.train.loop import Batch, Trainer
from wnca.train.phases import setup

OK, WARN, FAIL = "  ok  ", " WARN ", " FAIL "


def _hdr(title: str) -> None:
    print(f"\n{'=' * 74}\n== {title}\n{'=' * 74}")


def _verdict(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


# --------------------------------------------------------------------------------------


def stage_env(cfg, ctx) -> dict:
    _hdr("ENV")
    dev = ctx["device"]
    out = {}
    if dev == "cuda":
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability()
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        native_bf16 = cap[0] >= 8
        print(f"  {name}  capability {cap[0]}.{cap[1]}  {mem:.0f} GB")
        print(f"  torch {torch.__version__}  native bf16: {native_bf16}")
        out = {"gpu": name, "capability": cap, "native_bf16": native_bf16}
    try:
        import triton

        print(f"  triton {triton.__version__} (inductor available)")
    except ImportError:
        print("  triton missing -- torch.compile/inductor unavailable")
    return out


def stage_cache(cfg, ctx) -> dict:
    """Is the data sound, and does any single timestep carry an extreme value?

    A pathological sample is a plausible source of a huge gradient, and it would be invisible
    in aggregate statistics -- so look at per-timestep extremes, not just the global mean/std.
    """
    _hdr("CACHE")
    cache = ctx["cache"]
    keys = [c.key for c in cfg.variables.channels()]
    worst_overall = 0.0
    bad_total = 0

    for split in ("train", "val", "test"):
        arr = cache.split(split).array
        step = max(1, len(arr) // 400)
        s = np.asarray(arr[::step], dtype=np.float32)
        n_bad = int((~np.isfinite(s)).sum())
        bad_total += n_bad
        per_t = np.abs(s).max(axis=(1, 2))  # worst value in each sampled timestep
        worst_overall = max(worst_overall, float(per_t.max()))
        print(f"  {split:>5}: {len(arr):>6} steps, sampled {len(s):>4} | "
              f"non-finite {n_bad} | mean {np.nanmean(s):+.3f} std {np.nanstd(s):.3f} | "
              f"max|x| {per_t.max():.1f} (p99 {np.percentile(per_t, 99):.1f})")

    norm = cache.normalizer
    degenerate = [keys[i] for i in range(len(norm.std)) if not np.isfinite(norm.std[i]) or norm.std[i] < 1e-6]
    if bad_total:
        _verdict(FAIL, f"{bad_total} non-finite values in the cache -- rebuild it")
    elif degenerate:
        _verdict(FAIL, f"degenerate normalizer channels: {degenerate}")
    elif worst_overall > 25:
        _verdict(WARN, f"extreme sample present (max|x| = {worst_overall:.1f} sigma) -- "
                       "a plausible source of large gradients")
    else:
        _verdict(OK, f"cache clean, worst sample {worst_overall:.1f} sigma")
    return {"max_sigma": worst_overall, "non_finite": bad_total}


def _one_batch(cfg, ctx, B=None):
    """A real batch plus its forcing, ready to feed the model."""
    cache, mesh, dev = ctx["cache"], ctx["mesh"], ctx["device"]
    B = B or cfg.train.batch_size
    s = cache.split("train").array
    cur = torch.from_numpy(np.array(s[1:1 + B], dtype=np.float32)).to(dev)
    prev = torch.from_numpy(np.array(s[0:B], dtype=np.float32)).to(dev)
    st = torch.from_numpy(cache.static).float().to(dev).unsqueeze(0).expand(B, -1, -1)
    forcing = None
    if cfg.state.solar_forcing:
        from wnca.data.forcing import SolarForcing

        forcing = SolarForcing(cache.times("train"), mesh, dev).window(
            torch.arange(1, 1 + B, device=dev), 1)
    return cur, prev, st, forcing


def stage_numerics(cfg, ctx) -> dict:
    """Perception magnitudes, and whether each dtype survives forward AND backward.

    Mesh perception spans a wide dynamic range because the cotangent Laplacian scales as 1/h^2.
    That is what overflowed fp16 at n_sub=5; the number is worth printing for any new mesh.
    """
    _hdr("NUMERICS")
    from wnca.models.nca import build_model

    dev, mesh = ctx["device"], ctx["mesh"]
    model = build_model(cfg, mesh, dev)
    cur, prev, st, forcing = _one_batch(cfg, ctx)
    state = model.seed(cur)

    with torch.no_grad():
        p = model.perceive(state)
    C = cfg.c_state
    blocks = {}
    for i, nm in enumerate(("identity", "grad_x", "grad_y", "laplacian")):
        blocks[nm] = float(p[..., i * C:(i + 1) * C].abs().max())
    print("  perception block magnitudes (fp16 ceiling is 65504):")
    for nm, v in blocks.items():
        print(f"    {nm:>10} {v:>12.1f}")
    ratio = blocks["laplacian"] / max(blocks["identity"], 1e-9)
    headroom = 65504 / max(blocks["laplacian"], 1e-9)
    print(f"    dynamic range {ratio:.0f}x, fp16 headroom {headroom:.0f}x")

    results = {}
    for label, dtype in (("fp32", None), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        if dtype is not None and dev != "cuda":
            continue
        if dtype is torch.bfloat16 and dev == "cuda" and torch.cuda.get_device_capability()[0] < 8:
            print(f"  {label}: skipped (no native support)")
            continue
        model.zero_grad(set_to_none=True)
        try:
            with torch.autocast(dev, dtype=dtype or torch.float32, enabled=dtype is not None):
                out = model.rollout(model.seed(cur), st, 1, prev_phys=prev, forcing=forcing)
            loss = out.float().pow(2).mean()
            fwd_ok = bool(torch.isfinite(loss))
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(model.parameters(), 1e30))
            bwd_ok = np.isfinite(gn)
            results[label] = {"forward": fwd_ok, "grad_norm": gn}
            print(f"  {label}: forward {'finite' if fwd_ok else 'NON-FINITE'} | "
                  f"grad norm {gn:.4g} {'' if bwd_ok else '<- NON-FINITE'}")
        except Exception as e:  # noqa: BLE001
            results[label] = {"error": str(e)[:80]}
            print(f"  {label}: RAISED {type(e).__name__}: {str(e)[:70]}")

    if headroom < 10:
        _verdict(WARN, f"fp16 headroom only {headroom:.0f}x -- perception will overflow as "
                       "weights grow. Use bf16 or fp32.")
    else:
        _verdict(OK, "perception magnitudes leave adequate headroom")
    return {"blocks": blocks, "dtypes": results, "fp16_headroom": headroom}


def stage_substeps(cfg, ctx) -> dict:
    """Does the gradient norm compound with n_substeps?

    The backward multiplies by the per-sub-step Jacobian once per sub-step. If the norm grows
    super-linearly in `n_substeps`, the depth of the residual chain is the fragility -- and the
    model will be unusually sensitive to learning rate.
    """
    _hdr("SUBSTEP SCALING")
    from wnca.models.nca import build_model

    dev, mesh = ctx["device"], ctx["mesh"]
    cur, prev, st, forcing = _one_batch(cfg, ctx)
    print(f"  {'n_substeps':>11} {'grad norm':>12} {'vs n=1':>9}")
    base = None
    out = {}
    for n in (1, 5, 10, 20, 40):
        c = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, n_substeps=n))
        torch.manual_seed(0)
        m = build_model(c, mesh, dev)
        if ctx.get("checkpoint"):
            from wnca.train.checkpoint import load_checkpoint

            load_checkpoint(ctx["checkpoint"], m, c, map_location=dev, strict_arch=False)
        else:
            # Zero-init would understate the effect (no gradient at all through the head), so
            # stand in for a trained model. std=0.02 matches the measured per-element RMS of
            # the phase 2b checkpoint (head norm 4.39 over 30,720 elements = 0.025).
            nn.init.normal_(m.update.head.weight, std=0.02)
        o = m.rollout(m.seed(cur), st, 1, prev_phys=prev, forcing=forcing)
        o.float().pow(2).mean().backward()
        gn = float(nn.utils.clip_grad_norm_(m.parameters(), 1e30))
        base = gn if base is None else base
        out[n] = gn
        print(f"  {n:>11} {gn:>12.4g} {gn / max(base, 1e-30):>8.1f}x")
        del m
        torch.cuda.empty_cache() if dev == "cuda" else None

    growth = out.get(40, 0) / max(out.get(1, 1e-30), 1e-30)
    src = "the supplied checkpoint" if ctx.get("checkpoint") else "a randomly initialised head"
    if not np.isfinite(growth) or growth > 100:
        _verdict(WARN, f"gradient norm grows {growth:.3g}x from 1 to 40 sub-steps -- the "
                       "backward compounds through the residual chain, so the model is "
                       "intrinsically LR-sensitive")
        print(f"       measured with {src}. Weights a real run settles into may be tamer, so")
        print("       read the SHAPE of the growth, not the absolute numbers.")
    else:
        _verdict(OK, f"gradient norm grows {growth:.1f}x from 1 to 40 sub-steps ({src})")
    return out


def stage_lr(cfg, ctx, steps: int, lrs) -> dict:
    """Short runs at several learning rates, watching the gradient-norm trend.

    Explosion is preceded by growth, so a few hundred steps at fixed LR reveals where a setting
    is heading without waiting the ~2,000 steps the real failure took. LR is held FIXED here
    (no warmup, no decay) so the effect of the value is not confounded by the schedule.
    """
    _hdr(f"LR SWEEP ({steps} steps each, fixed LR, no schedule)")
    from wnca.models.nca import build_model

    dev, mesh, cache = ctx["device"], ctx["mesh"], ctx["cache"]
    pf = 1 if cfg.train.pushforward else 0
    print(f"  {'lr':>9} {'gn @start':>10} {'gn @end':>10} {'trend':>8} {'max gn':>10} "
          f"{'nonfinite':>10} verdict")
    results = {}

    for lr in lrs:
        c = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, lr=lr, warmup_steps=0))
        torch.manual_seed(0)
        np.random.seed(0)
        model = build_model(c, mesh, dev)
        tr = Trainer(c, model, mesh, cache, dev)
        loader = make_loader(cache, "train", c, n_out=1 + pf, shuffle=True)
        tr.split = "train"
        model.train()

        norms, n_bad = [], 0
        for i, (prev, cur, tgt, idx) in enumerate(loader):
            if i >= steps:
                break
            b = Batch(prev.to(dev), cur.to(dev), tgt.to(dev), idx)
            tr._set_lr(lr)  # fixed, deliberately
            # Autocast lives in Trainer.run_epoch, and this bypasses it -- so apply it here or
            # every "amp" measurement is silently fp32. (It was, until this was caught by two
            # precisions producing byte-identical numbers.)
            with torch.autocast(tr.amp_device, dtype=tr.amp_dtype, enabled=tr.use_amp):
                pred, ovf, off = tr._forward(b, 1, 1, True)
            obj, clean, _ = tr._loss(pred.float(), b.tgt[:, off:], ovf.float())
            tr.opt.zero_grad(set_to_none=True)
            obj.backward()
            gn = float(nn.utils.clip_grad_norm_(model.parameters(), c.train.grad_clip))
            if np.isfinite(gn):
                tr.opt.step()
                norms.append(gn)
            else:
                n_bad += 1

        if not norms:
            print(f"  {lr:>9.1e} {'--':>10} {'--':>10} {'--':>8} {'--':>10} {n_bad:>9} EXPLODED")
            results[lr] = {"exploded": True}
            continue
        k = max(3, len(norms) // 5)
        start, end = float(np.median(norms[:k])), float(np.median(norms[-k:]))
        trend = end / max(start, 1e-30)
        mx = float(np.max(norms))
        clip = cfg.train.grad_clip
        hot = mx > 50 * clip          # peak gradient far above the clip threshold
        if n_bad:
            v = "NON-FINITE"
        elif trend >= 3:
            v = "RISING"
        elif hot:
            v = "SPIKY"
        else:
            v = "ok"
        print(f"  {lr:>9.1e} {start:>10.3g} {end:>10.3g} {trend:>7.1f}x {mx:>10.3g} "
              f"{n_bad:>9} {v}")
        results[lr] = {"start": start, "end": end, "trend": trend, "max": mx, "non_finite": n_bad}
        del model, tr
        torch.cuda.empty_cache() if dev == "cuda" else None

    safe = [lr for lr, r in results.items()
            if not r.get("exploded") and r.get("non_finite", 1) == 0
            and r.get("trend", 9) < 3 and r.get("max", 1e30) <= 50 * cfg.train.grad_clip]
    if safe:
        _verdict(OK, f"largest LR with a flat gradient trend: {max(safe):.1e}")
        print(f"       recommend lr <= {max(safe):.1e} for a long run "
              "(a long cosine schedule holds near peak for thousands of steps)")
    else:
        _verdict(FAIL, "every LR tested shows a rising or non-finite gradient trend -- "
                       "try lower values, or reduce n_substeps")
    return results


def stage_matrix(cfg, ctx, steps: int = 12) -> dict:
    """amp x pushforward x solar_forcing. Interactions are where the bugs have been."""
    _hdr(f"CONFIG MATRIX ({steps} steps each)")
    from wnca.models.nca import build_model

    dev, mesh, cache = ctx["device"], ctx["mesh"], ctx["cache"]
    print(f"  {'amp':>5} {'pushfwd':>8} {'solar':>6} | {'loss':>10} {'grad norm':>11} result")
    out = {}
    amps = (False, True) if dev == "cuda" else (False,)
    for amp, pf, solar in itertools.product(amps, (False, True), (False, True)):
        c = dataclasses.replace(
            cfg,
            state=dataclasses.replace(cfg.state, solar_forcing=solar),
            train=dataclasses.replace(cfg.train, amp=amp, pushforward=pf, warmup_steps=0),
        )
        torch.manual_seed(0)
        try:
            model = build_model(c, mesh, dev)
            tr = Trainer(c, model, mesh, cache, dev)
            loader = make_loader(cache, "train", c, n_out=1 + (1 if pf else 0), shuffle=False)
            tr.split = "train"
            model.train()
            last_loss, last_gn = float("nan"), float("nan")
            for i, (prev, cur, tgt, idx) in enumerate(loader):
                if i >= steps:
                    break
                b = Batch(prev.to(dev), cur.to(dev), tgt.to(dev), idx)
                with torch.autocast(tr.amp_device, dtype=tr.amp_dtype, enabled=tr.use_amp):
                    pred, ovf, off = tr._forward(b, 1, 1, True)
                obj, last_loss, _ = tr._loss(pred.float(), b.tgt[:, off:], ovf.float())
                tr.opt.zero_grad(set_to_none=True)
                obj.backward()
                last_gn = float(nn.utils.clip_grad_norm_(model.parameters(), c.train.grad_clip))
                if np.isfinite(last_gn):
                    tr.opt.step()
            good = np.isfinite(last_loss) and np.isfinite(last_gn)
            print(f"  {str(amp):>5} {str(pf):>8} {str(solar):>6} | {last_loss:>10.4g} "
                  f"{last_gn:>11.4g} {'ok' if good else 'NON-FINITE'}")
            out[f"{amp}_{pf}_{solar}"] = good
            del model, tr
            torch.cuda.empty_cache() if dev == "cuda" else None
        except Exception as e:  # noqa: BLE001
            print(f"  {str(amp):>5} {str(pf):>8} {str(solar):>6} | RAISED "
                  f"{type(e).__name__}: {str(e)[:40]}")
            out[f"{amp}_{pf}_{solar}"] = False
    bad = [k for k, v in out.items() if not v]
    _verdict(FAIL if bad else OK,
             f"failing combinations: {bad}" if bad else "all combinations finite")
    return out


# --------------------------------------------------------------------------------------

STAGES = {
    "env": stage_env,
    "cache": stage_cache,
    "numerics": stage_numerics,
    "substeps": stage_substeps,
    "lr": stage_lr,
    "matrix": stage_matrix,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="configs/phase2c_full.yaml")
    ap.add_argument("--stages", default="all",
                    help="comma-separated: " + ",".join(STAGES) + ", or 'all'")
    ap.add_argument("--steps", type=int, default=150, help="steps per LR in the sweep")
    ap.add_argument("--lrs", default="1e-3,5e-4,3e-4,1e-4")
    ap.add_argument("--amp", choices=("on", "off"), default=None,
                    help="override train.amp for the lr sweep, to compare precisions at a "
                         "learning rate that is known to be stable")
    ap.add_argument("--checkpoint", default=None,
                    help="probe a trained model rather than a stand-in init (substeps stage)")
    ap.add_argument("--out", default="diagnosis.json")
    args = ap.parse_args(argv)

    names = list(STAGES) if args.stages == "all" else args.stages.split(",")
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        print(f"unknown stages: {unknown}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if args.amp is not None:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, amp=(args.amp == "on")))
    t0 = time.time()
    mesh, cache, _, _, device = setup(cfg, verbose=False)
    ctx = {"mesh": mesh, "cache": cache, "device": device, "checkpoint": args.checkpoint}
    amp_lbl = "off"
    if cfg.train.amp and device == "cuda":
        amp_lbl = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    print(f"config {args.config} | {cfg.c_phys} channels | n_sub {cfg.mesh.n_sub} | "
          f"hidden {cfg.model.hidden_dim} | n_substeps {cfg.model.n_substeps} | "
          f"pushforward {cfg.train.pushforward} | device {device} | amp {amp_lbl}")

    report = {}
    for n in names:
        try:
            if n == "lr":
                report[n] = stage_lr(cfg, ctx, args.steps,
                                     [float(x) for x in args.lrs.split(",")])
            else:
                report[n] = STAGES[n](cfg, ctx)
        except Exception as e:  # noqa: BLE001
            print(f"[{FAIL}] stage {n} raised {type(e).__name__}: {e}")
            report[n] = {"error": str(e)}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\ntotal {time.time() - t0:.0f}s | written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
