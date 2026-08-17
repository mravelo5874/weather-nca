#!/usr/bin/env python
"""Render forecast rollouts as video.

Three modes, each answering a different question:

  rollout   truth | forecast | error, animated over lead time. Where is the model wrong?
  compare   truth | model A | model B | error difference. Does more data help, and where?
  spectrum  the map alongside its power spectrum and error curve. Is it blurring?

Maps are nearest-neighbour scattered to a lat-lon grid for DISPLAY ONLY -- never for scoring.
Scoring uses the conservative regrid in `mesh/regrid.py`.

    python scripts/animate.py rollout --config configs/phase2a_data.yaml
    python scripts/animate.py compare --config configs/phase2a_data.yaml \
        --checkpoint-b runs/phase0_.../best_*.pt --label-b "2 years" --label-a "39 years"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from wnca.config import HOURS_PER_WINDOW, load_config  # noqa: E402
from wnca.eval.metrics import channel_units  # noqa: E402
from wnca.mesh.regrid import nearest_grid_index  # noqa: E402
from wnca.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from wnca.train.phases import setup  # noqa: E402

TRUTH_CMAP = "RdBu_r"
ERR_CMAP = "PuOr"


# --------------------------------------------------------------------------------------


def _grid(nn_idx, vals):
    return np.asarray(vals).reshape(-1)[nn_idx]


_UNIT_TEX = {"m2/s2": r"m$^2$/s$^2$", "m/s": "m/s", "K": "K", "log": "log units", "-": ""}


def _label(cfg, normalizer, ci, chan):
    """Axis label with the channel's real unit. Log-transformed channels are marked as such,
    because those values are log(kg/kg) and calling them physical would be wrong."""
    u = _UNIT_TEX.get(channel_units(cfg, normalizer)[ci], "")
    return f"{chan}" + (f"  ({u})" if u else "")


def _style(ax, title):
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, lw=0.4)


def _imshow(ax, data, vmin, vmax, cmap):
    return ax.imshow(data, origin="lower", extent=[-180, 180, -90, 90], cmap=cmap,
                     vmin=vmin, vmax=vmax, aspect="auto", interpolation="bilinear")


@torch.no_grad()
def _rollout(model, cfg, cache, start, n_windows, device, n_members=1):
    series = cache.split(args_split).array
    prev = torch.from_numpy(np.array(series[start - 1], dtype=np.float32)).unsqueeze(0).to(device)
    cur = torch.from_numpy(np.array(series[start], dtype=np.float32)).unsqueeze(0).to(device)
    st = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)
    pred = model.rollout_ensemble(model.seed(cur), st, n_windows, prev_phys=prev,
                                  n_members=n_members)
    return pred.mean(dim=1)[0].cpu().numpy()  # [W, N, C]


def _load(cfg, path, device, mesh):
    from wnca.models.nca import build_model

    model = build_model(cfg, mesh, device)
    load_checkpoint(path, model, cfg, map_location=device)
    model.eval()
    return model


def _area_rmse(a, b, area):
    w = area / area.mean()
    return float(np.sqrt(np.average((a - b) ** 2, weights=w)))


# --------------------------------------------------------------------------------------


def mode_rollout(args, cfg, mesh, cache, device, ci, chan, out_dir):
    model = _load(cfg, args.checkpoint, device, mesh)
    pred = _rollout(model, cfg, cache, args.start, args.windows, device)
    series = cache.split(args.split).array
    norm = cache.normalizer

    nn_idx, _, _ = nearest_grid_index(mesh)
    area = mesh["area"]
    sd, mu = float(norm.std[ci]), float(norm.mean[ci])

    truth = np.stack([np.array(series[args.start + 1 + k, :, ci]) for k in range(args.windows)])
    fc = pred[:, :, ci]
    truth_p, fc_p = truth * sd + mu, fc * sd + mu
    err = fc_p - truth_p

    lo, hi = np.percentile(truth_p, [1, 99])
    emax = np.percentile(np.abs(err), 99)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    ims = [
        _imshow(axes[0], _grid(nn_idx, truth_p[0]), lo, hi, TRUTH_CMAP),
        _imshow(axes[1], _grid(nn_idx, fc_p[0]), lo, hi, TRUTH_CMAP),
        _imshow(axes[2], _grid(nn_idx, err[0]), -emax, emax, ERR_CMAP),
    ]
    fig.colorbar(ims[1], ax=axes[:2], shrink=0.8, label=_label(cfg, norm, ci, chan), pad=0.01)
    fig.colorbar(ims[2], ax=axes[2], shrink=0.8, label="error", pad=0.01)
    sup = fig.suptitle("", fontsize=12)

    def update(k):
        ims[0].set_data(_grid(nn_idx, truth_p[k]))
        ims[1].set_data(_grid(nn_idx, fc_p[k]))
        ims[2].set_data(_grid(nn_idx, err[k]))
        r = _area_rmse(fc[k], truth[k], area) * sd
        _style(axes[0], "ERA5 truth")
        _style(axes[1], f"NCA forecast")
        _style(axes[2], f"error   (area-weighted RMSE {r:,.0f})")
        sup.set_text(f"{chan}  —  lead +{(k + 1) * HOURS_PER_WINDOW} h "
                     f"({(k + 1) * HOURS_PER_WINDOW / 24:.2f} days)")
        return ims

    path = out_dir / f"rollout_{cfg.phase}_{chan}.mp4"
    _save(fig, update, args.windows, path, args.fps)
    return path


def mode_compare(args, cfg, mesh, cache, device, ci, chan, out_dir):
    """Two models on the same truth.

    Each model is run against **its own cache**, because each was trained under its own
    normalization (phase 0's z500 std is 2796.0, phase 2a's is 2735.5 -- the 2-year window is a
    different sample of the climate). Feeding one model the other's normalized values would
    quietly mis-scale its inputs and make it look worse than it is. Both are decoded to physical
    units before anything is compared, which is the only space in which they are commensurable.
    """
    model_a = _load(cfg, args.checkpoint, device, mesh)
    pred_a = _rollout(model_a, cfg, cache, args.start, args.windows, device)
    norm_a = cache.normalizer

    cfg_b = load_config(args.config_b or args.config,
                        overrides={"data": {"test_years": list(cfg.data.test_years)}})
    _, cache_b, _, _, _ = setup(cfg_b, device, verbose=False)
    model_b = _load(cfg_b, args.checkpoint_b, device, mesh)
    pred_b = _rollout(model_b, cfg_b, cache_b, args.start, args.windows, device)
    norm_b = cache_b.normalizer

    keys_b = [c.key for c in cfg_b.variables.channels()]
    ci_b = keys_b.index(chan)
    sda, mua = float(norm_a.std[ci]), float(norm_a.mean[ci])
    sdb, mub = float(norm_b.std[ci_b]), float(norm_b.mean[ci_b])
    print(f"  normalizers: A std {sda:.1f} / B std {sdb:.1f} -> decoding both to physical units")

    series = cache.split(args.split).array
    nn_idx, _, _ = nearest_grid_index(mesh)
    area = mesh["area"]
    sd, mu = sda, mua

    truth = np.stack([np.array(series[args.start + 1 + k, :, ci]) for k in range(args.windows)])
    a, b = pred_a[:, :, ci], pred_b[:, :, ci_b]
    truth_p, a_p, b_p = truth * sda + mua, a * sda + mua, b * sdb + mub
    # Re-express B in A's normalized units so the RMSE overlay stays in one scale.
    b = (b_p - mua) / sda
    lo, hi = np.percentile(truth_p, [1, 99])
    err_a, err_b = np.abs(a_p - truth_p), np.abs(b_p - truth_p)
    emax = np.percentile(np.abs(err_b - err_a), 99)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.2), constrained_layout=True)
    ims = [
        _imshow(axes[0], _grid(nn_idx, truth_p[0]), lo, hi, TRUTH_CMAP),
        _imshow(axes[1], _grid(nn_idx, a_p[0]), lo, hi, TRUTH_CMAP),
        _imshow(axes[2], _grid(nn_idx, b_p[0]), lo, hi, TRUTH_CMAP),
        _imshow(axes[3], _grid(nn_idx, (err_b - err_a)[0]), -emax, emax, "RdYlGn"),
    ]
    fig.colorbar(ims[2], ax=axes[:3], shrink=0.8, label=_label(cfg, norm_a, ci, chan), pad=0.01)
    fig.colorbar(ims[3], ax=axes[3], shrink=0.8, label=f"|err {args.label_b}| − |err {args.label_a}|", pad=0.01)
    sup = fig.suptitle("", fontsize=12)

    def update(k):
        ims[0].set_data(_grid(nn_idx, truth_p[k]))
        ims[1].set_data(_grid(nn_idx, a_p[k]))
        ims[2].set_data(_grid(nn_idx, b_p[k]))
        ims[3].set_data(_grid(nn_idx, (err_b - err_a)[k]))
        ra = _area_rmse(a[k], truth[k], area) * sd
        rb = _area_rmse(b[k], truth[k], area) * sd
        _style(axes[0], "ERA5 truth")
        _style(axes[1], f"{args.label_a}   (RMSE {ra:,.0f})")
        _style(axes[2], f"{args.label_b}   (RMSE {rb:,.0f})")
        _style(axes[3], f"green = {args.label_a} better")
        sup.set_text(f"{chan}  —  lead +{(k + 1) * HOURS_PER_WINDOW} h "
                     f"({(k + 1) * HOURS_PER_WINDOW / 24:.2f} days)")
        return ims

    # Include both labels: every comparison used to write compare_<chan>.mp4 and silently
    # overwrite the previous one.
    slug = lambda t: re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")
    path = out_dir / f"compare_{slug(args.label_a)}_vs_{slug(args.label_b)}_{chan}.mp4"
    _save(fig, update, args.windows, path, args.fps)
    return path


def mode_spectrum(args, cfg, mesh, cache, device, ci, chan, out_dir):
    """Map + zonal power spectrum + error curve. Blurring is easier to see than to read off a table."""
    model = _load(cfg, args.checkpoint, device, mesh)
    pred = _rollout(model, cfg, cache, args.start, args.windows, device)
    series = cache.split(args.split).array
    norm = cache.normalizer
    nn_idx, glat, _ = nearest_grid_index(mesh)
    area = mesh["area"]
    sd, mu = float(norm.std[ci]), float(norm.mean[ci])

    truth = np.stack([np.array(series[args.start + 1 + k, :, ci]) for k in range(args.windows)])
    fc = pred[:, :, ci]
    truth_p, fc_p = truth * sd + mu, fc * sd + mu
    lo, hi = np.percentile(truth_p, [1, 99])

    def zonal(vals):
        g = _grid(nn_idx, vals)
        w = np.cos(np.radians(glat))[:, None]
        p = np.abs(np.fft.rfft(g - g.mean(axis=1, keepdims=True), axis=1)) ** 2
        return (p * w).sum(0) / w.sum()

    rmse = np.array([_area_rmse(fc[k], truth[k], area) * sd for k in range(args.windows)])
    leads = (np.arange(args.windows) + 1) * HOURS_PER_WINDOW

    fig = plt.figure(figsize=(16, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_t = fig.add_subplot(gs[0, 0]); ax_f = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 0]); ax_e = fig.add_subplot(gs[1, 1])

    im_t = _imshow(ax_t, _grid(nn_idx, truth_p[0]), lo, hi, TRUTH_CMAP)
    im_f = _imshow(ax_f, _grid(nn_idx, fc_p[0]), lo, hi, TRUTH_CMAP)
    fig.colorbar(im_f, ax=[ax_t, ax_f], shrink=0.85, pad=0.01)

    k_ax = np.arange(len(zonal(truth[0])))
    (ln_t,) = ax_s.loglog(k_ax[1:], zonal(truth_p[0])[1:], color="#868E96", label="ERA5")
    (ln_f,) = ax_s.loglog(k_ax[1:], zonal(fc_p[0])[1:], color="#4C6EF5", label="NCA")
    ax_s.set_xlabel("zonal wavenumber"); ax_s.set_ylabel("power")
    ax_s.set_title("power spectrum — below truth at high k = blurring", fontsize=10)
    ax_s.legend(frameon=False, fontsize=8)
    ax_s.spines[["top", "right"]].set_visible(False)

    ax_e.plot(leads, rmse, color="#4C6EF5", lw=1.5)
    dot, = ax_e.plot([leads[0]], [rmse[0]], "o", color="#E8590C", ms=7)
    ax_e.set_xlabel("lead time (h)"); ax_e.set_ylabel("area-weighted RMSE (m$^2$/s$^2$)")
    ax_e.set_title("error growth", fontsize=10)
    ax_e.spines[["top", "right"]].set_visible(False)
    sup = fig.suptitle("", fontsize=13)

    def update(k):
        im_t.set_data(_grid(nn_idx, truth_p[k]))
        im_f.set_data(_grid(nn_idx, fc_p[k]))
        st, sf = zonal(truth_p[k]), zonal(fc_p[k])
        ln_t.set_ydata(st[1:]); ln_f.set_ydata(sf[1:])
        ax_s.relim(); ax_s.autoscale_view()
        dot.set_data([leads[k]], [rmse[k]])
        _style(ax_t, "ERA5 truth"); _style(ax_f, "NCA forecast")
        hi_k = slice(len(k_ax) // 3, None)
        ratio = sf[hi_k].sum() / max(st[hi_k].sum(), 1e-30)
        sup.set_text(f"{chan}  —  +{leads[k]} h   |   RMSE {rmse[k]:,.0f}   |   "
                     f"high-k power {ratio:.0%} of truth")
        return [im_t, im_f, ln_t, ln_f, dot]

    path = out_dir / f"spectrum_{cfg.phase}_{chan}.mp4"
    _save(fig, update, args.windows, path, args.fps)
    return path


def _save(fig, update, n_frames, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(str(path), writer=animation.FFMpegWriter(fps=fps, bitrate=3200,
                                                      extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print(f"  wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["rollout", "compare", "spectrum", "all"])
    ap.add_argument("--config", "-c", default="configs/phase2a_data.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config-b", default=None)
    ap.add_argument("--checkpoint-b", default=None)
    ap.add_argument("--label-a", default="model A")
    ap.add_argument("--label-b", default="model B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--channel", default="geopotential_500")
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", default="media")
    args = ap.parse_args(argv)

    global args_split
    args_split = args.split

    cfg = load_config(args.config)
    mesh, cache, _, _, device = setup(cfg, verbose=True)
    args.checkpoint = args.checkpoint or latest_checkpoint(f"runs")
    if args.checkpoint is None:
        print("no checkpoint found", file=sys.stderr)
        return 2

    keys = [c.key for c in cfg.variables.channels()]
    chan = args.channel if args.channel in keys else keys[0]
    ci = keys.index(chan)
    out_dir = Path(args.out)
    args.windows = min(args.windows, len(cache.split(args.split)) - args.start - 2)

    print(f"rendering {args.mode} | {chan} | start {args.start} | {args.windows} windows "
          f"({args.windows * HOURS_PER_WINDOW / 24:.1f} days)")

    modes = ["rollout", "spectrum"] if args.mode == "all" else [args.mode]
    if args.mode == "all" and args.checkpoint_b:
        modes.append("compare")
    for m in modes:
        {"rollout": mode_rollout, "compare": mode_compare, "spectrum": mode_spectrum}[m](
            args, cfg, mesh, cache, device, ci, chan, out_dir
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
