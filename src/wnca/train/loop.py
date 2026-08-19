"""Training loop.

Deterministic phases (2a-2c) train single-step against area-weighted MSE. The probabilistic
phase (3a) swaps in fair CRPS over an in-batch ensemble and turns on the FiLM noise pathway.
Everything else -- optimizer, schedule, selection metric, checkpoint discipline -- is shared,
so a phase boundary changes one term in the objective and nothing else.

`rollout_epochs` is off and gated in `config.validate()`: M1 ran the curriculum twice, it
contributed nothing, and it destroyed the long-lead metric both times. The replacement for
rollout drift is `train.pushforward`, which is a genuinely different mechanism -- unroll one
window under `no_grad`, then take the step from that state, so the rule learns to contract
error from states it actually produces.
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..losses.crps import ensemble_spread, fair_crps, spread_skill_ratio
from ..losses.spectral import band_energy_crps
from ..losses.terms import area_weighted_mse, per_channel_rmse
from .checkpoint import assert_finite, save_checkpoint, timestamped_path

_PREEMPTED = {"flag": False}


def _install_sigterm_handler():
    """Spot preemption: never let a run's only state live in GPU memory."""

    def handler(signum, frame):
        _PREEMPTED["flag"] = True
        print("\n  SIGTERM received -- checkpointing at the next safe point")

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # not on the main thread, or unsupported
        pass


_TRACE_LAYERS_EVERY = 50


@dataclass
class Batch:
    prev: torch.Tensor  # [B, N, C]
    cur: torch.Tensor
    tgt: torch.Tensor  # [B, n_out, N, C]
    idx: torch.Tensor | None = None  # absolute time index of `cur`, for the solar forcing


class Trainer:
    def __init__(self, cfg: Config, model: nn.Module, mesh, cache, device: str = "cpu",
                 bands=None, tracker=None):
        from ..losses.terms import area_weights, channel_weights

        self.cfg = cfg
        self.model = model
        self.device = device
        self.cache = cache
        self.bands = bands
        self.tracker = tracker

        self.area_w = area_weights(mesh["area"], device)  # [N, 1]
        self.chan_w = channel_weights(cfg, device)  # [1, C]
        self.static = torch.from_numpy(cache.static).float().to(device)  # [N, c_static]

        # Solar forcing is keyed on absolute time index, and the index spaces of the splits are
        # different, so one table per split rather than one global one.
        self.solar = {}
        if cfg.state.solar_forcing and cache is not None:
            from ..data.forcing import SolarForcing

            for sp in ("train", "val", "test"):
                self.solar[sp] = SolarForcing(cache.times(sp), mesh, device)
        self.split = "train"

        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)
        self.step = 0
        self.epoch = 0  # fit() keeps this current, for mid-epoch checkpoints
        self.best = float("inf")
        # Optional per-step gradient trace (diagnostics only; see fit()).
        self.grad_trace: Path | None = None
        # Optional rolling mid-epoch checkpoint path (fit() sets it from ckpt_every_steps).
        self.rolling_ckpt: Path | None = None
        # Finite-but-absurd loss guard: a diverged model does not always produce a NaN --
        # phase 2c's epoch-2 train loss was 3.69e23, perfectly finite in fp32. A batch loss
        # beyond `_absurd_factor` x the running median of recent losses is skipped and
        # counted; sustained occurrences abort like the non-finite guards do. The window is
        # Trainer-level (NOT reset per epoch): a model destroyed mid-epoch must still be
        # judged against the healthy losses that preceded it. Absurd values are excluded
        # from the window so the median cannot chase the runaway upward.
        self._loss_window: deque[float] = deque(maxlen=200)
        self._absurd_factor = 100.0
        self._absurd_arm = 20  # batches in the window before the guard arms

        # AMP matters on the cloud GPUs, not on a 1660 Ti: Turing has no tensor cores, so
        # locally this is roughly a no-op. Autocast has no mps backend, hence the fallback.
        self.amp_device = device if device in ("cuda", "cpu") else "cpu"
        self.use_amp = cfg.train.amp and device == "cuda"

        # **bf16 over fp16 whenever the card supports it.** Mesh perception spans a ~1000x
        # dynamic range -- measured at n_sub=5, the Laplacian block reaches 8000 while the
        # identity block is 7.2 -- because the cotangent Laplacian scales as 1/h^2. fp16 tops
        # out at 65504, so that leaves under 10x headroom and the forward overflows once the
        # weights grow. Phase 2c died this way partway through epoch 1: 143 non-finite batches.
        # bf16 has fp32's exponent range, runs at the same tensor-core speed on Ampere and
        # later, and needs no loss scaling.
        # Compute capability >= 8.0 (Ampere) is where bf16 is NATIVE.
        # `torch.cuda.is_bf16_supported()` also returns True for emulated support on Turing,
        # which would be correct but slow, so check the capability directly.
        self.amp_dtype = torch.float16
        if self.use_amp and torch.cuda.get_device_capability()[0] >= 8:
            self.amp_dtype = torch.bfloat16
        # GradScaler exists to manage fp16 overflow; bf16 does not need it.
        self.scaler = (torch.amp.GradScaler(device)
                       if self.use_amp and self.amp_dtype is torch.float16 else None)
        _install_sigterm_handler()

    # ---- schedule ----
    def _lr_at(self, step: int, total: int) -> float:
        w = self.cfg.train.warmup_steps
        if step < w:
            return self.cfg.train.lr * (step + 1) / max(w, 1)
        prog = (step - w) / max(total - w, 1)
        return self.cfg.train.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    def _set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ---- forward ----
    def _static_for(self, b: int) -> torch.Tensor:
        return self.static.unsqueeze(0).expand(b, -1, -1)

    def _forcing_for(self, batch: Batch, n_out: int, shift: int = 0):
        """[B, n_out, N, 3] solar forcing at each window's target time, or None if disabled."""
        sf = self.solar.get(self.split)
        if sf is None or batch.idx is None:
            return None
        return sf.window(batch.idx.to(self.device) + shift, n_out)

    def _forward(self, batch: Batch, n_out: int, n_members: int, train: bool):
        """Returns (pred, overflow, offset).

        pred is [B, M, n_out, N, C]; the member axis is always present so the loss has one path.
        `offset` is how many windows the predictions are shifted forward relative to `batch.tgt`
        -- 1 under pushforward, 0 otherwise. The caller must slice `tgt[:, offset:]`.
        """
        cfg = self.cfg
        cur, prev = batch.cur, batch.prev
        st = self._static_for(cur.shape[0])
        forcing = self._forcing_for(batch, n_out)

        # Sanchez-Gonzalez-style input noise on the physical channels (M1's noise_std).
        state = self.model.seed(cur)
        if train and cfg.train.noise_std > 0:
            kick = torch.zeros_like(state)
            kick[..., : cfg.c_phys] = cfg.train.noise_std * torch.randn_like(kick[..., : cfg.c_phys])
            state = state + kick

        # Brandstetter pushforward: advance one window with no gradient, then train from the
        # state the model actually produces. Distinct mechanism from the input noise above.
        offset = 0
        if train and cfg.train.pushforward:
            # `enabled=False` is load-bearing, not tidiness. Running a module under no_grad
            # INSIDE an autocast region populates autocast's weight cache with fp16 copies that
            # do not track gradients; the tracked forward then reuses them and its output has no
            # grad_fn, so backward dies with "element 0 of tensors does not require grad". The
            # pushforward step is no-grad anyway, so fp32 here costs nothing.
            with torch.no_grad(), torch.autocast(self.amp_device, enabled=False):
                # The pushforward window is window 0, so it takes the unshifted forcing.
                f0 = forcing[:, 0] if forcing is not None else None
                stepped = self.model.forecast_step(state, st, prev, None, f0)
                # The field one window before `stepped` is the ORIGINAL cur, not stepped's own
                # physical channels -- using the latter zeroes the second-order tendency.
                prev = cur
                if cfg.model.reseed_hidden:
                    stepped = self.model.reseed_hidden(stepped)
            state = stepped.detach()
            offset = 1  # predictions now start at window +2, so targets shift by one
            if forcing is not None:  # the supervised windows also shift by one
                forcing = self._forcing_for(batch, n_out, shift=1)

        M = n_members if cfg.model.stochastic else 1
        pred, ovf = self.model.rollout_ensemble(
            state, st, n_out, prev_phys=prev, n_members=M, return_aux=True, forcing=forcing
        )
        return pred, ovf, offset

    def _loss(self, pred: torch.Tensor, tgt: torch.Tensor, ovf: torch.Tensor):
        """Composite objective. Both scored terms are proper; the overflow term is a penalty.

            L = w_field * fairCRPS(members, truth)
              + w_spec  * fairCRPS(log band energies)
              + w_over  * overflow_penalty(hidden)

        In deterministic phases M = 1 and the field term degenerates to area-weighted MSE,
        which keeps 2a-2c comparable to M1 rather than introducing a silent objective change.
        """
        cfg = self.cfg
        parts = {}

        if cfg.model.stochastic:
            # Score every supervised window, flattening the lead axis into the batch.
            B, M, W = pred.shape[:3]
            p = pred.permute(0, 2, 1, 3, 4).reshape(B * W, M, *pred.shape[3:])
            t = tgt.reshape(B * W, *tgt.shape[2:])
            field = fair_crps(p, t, self.area_w, self.chan_w, alpha=cfg.loss.crps_alpha)
            parts["field_crps"] = float(field.item())
            if cfg.loss.w_spec > 0 and self.bands is not None:
                spec = band_energy_crps(p, t, self.bands, self.chan_w, alpha=cfg.loss.crps_alpha)
                parts["spec_crps"] = float(spec.item())
            else:
                spec = pred.new_zeros(())
        else:
            field = area_weighted_mse(pred[:, 0], tgt, self.area_w, self.chan_w)
            parts["field_mse"] = float(field.item())
            spec = pred.new_zeros(())

        total = cfg.loss.w_field * field + cfg.loss.w_spec * spec + cfg.loss.w_over * ovf
        parts["overflow"] = float(ovf.item())
        return total, float(field.item()), parts

    # ---- diagnostics ----
    def _write_grad_trace(self, gn: float, loss: float, lr: float) -> None:
        """Append one line per optimizer step: the PRE-clip gradient norm, loss and LR.

        `clip_grad_norm_` returns the norm it measured before clipping, which is the quantity
        that matters here -- a clipped step still lands, so the post-clip norm is constant by
        construction and says nothing about divergence onset.

        Per-parameter norms every `_TRACE_LAYERS_EVERY` steps: they cost a pass over the
        parameter list, which is not free at 7k steps/epoch, but the whole point is to see
        WHICH block blows up first.
        """
        rec = {"step": self.step, "grad_norm": gn, "loss": loss, "lr": lr}
        if self.step % _TRACE_LAYERS_EVERY == 0:
            rec["layers"] = {
                n: float(p.grad.norm()) for n, p in self.model.named_parameters()
                if p.grad is not None
            }
            rec["weights"] = {
                n: float(p.detach().norm()) for n, p in self.model.named_parameters()
            }
        with open(self.grad_trace, "a", encoding="utf-8") as fh:
            print(json.dumps(rec), file=fh)

    # ---- epochs ----
    def run_epoch(self, loader, n_out: int, train: bool, total_steps: int = 1,
                  split: str = "train") -> dict:
        """One pass. Returns the CLEAN field term (overflow optimized but not reported), so
        the printed number stays comparable across configurations."""
        cfg = self.cfg
        self.split = split  # selects the right solar-forcing table
        self.model.train(train)
        acc, nb, skipped, bad_grad, bad_loss = 0.0, 0, 0, 0, 0
        # Same tolerance for both divergence guards: a handful of bad batches is an
        # outlier, 2% of an epoch is a diverged model.
        tol = max(10, 0.02 * len(loader))
        M = cfg.ensemble.m_train if train else cfg.ensemble.m_val

        for prev, cur, tgt, idx in loader:
            batch = Batch(prev.to(self.device, non_blocking=True),
                          cur.to(self.device, non_blocking=True),
                          tgt.to(self.device, non_blocking=True), idx)
            with torch.set_grad_enabled(train):
                with torch.autocast(self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
                    pred, ovf, offset = self._forward(batch, n_out, M, train)
                # The loss is computed in fp32: CRPS differences at small M are exactly the
                # quantity fp16 rounds away, and under-dispersion would be the silent result.
                obj, clean, _ = self._loss(pred.float(), batch.tgt[:, offset:], ovf.float())

            # A non-finite loss makes the whole backward pass non-finite, so skip the batch
            # before it can touch the optimizer. Guarding only the gradient norm (as before)
            # works, but wastes a backward pass and hides how often it is happening.
            if not np.isfinite(clean):
                skipped += 1
                if skipped > tol:
                    raise RuntimeError(
                        f"{skipped} non-finite batches this epoch -- the model has diverged, "
                        "not hit an outlier. Resume from the last good checkpoint at a lower "
                        "learning rate (wnca train --resume auto --set train.lr=...)."
                    )
                continue

            # Finite-but-absurd: divergence does not always produce a NaN. Phase 2c's
            # destroyed model reported a FINITE 3.69e23 train loss for ten hours while no
            # guard watched the magnitude. The onset trace goes 0.219 -> 63.9 -> 3.0e8 in
            # four steps, so a 100x-median rule catches a runaway within a few batches.
            if (len(self._loss_window) >= self._absurd_arm
                    and clean > self._absurd_factor * float(np.median(self._loss_window))):
                bad_loss += 1
                if bad_loss > tol:
                    raise RuntimeError(
                        f"{bad_loss} finite-but-absurd batches this epoch (loss {clean:.3g} "
                        f"vs running median {float(np.median(self._loss_window)):.3g}) -- "
                        "the model has diverged, not hit an outlier. Resume from the last "
                        "good checkpoint (wnca train --resume auto)."
                    )
                continue
            self._loss_window.append(clean)

            if train:
                lr_now = self._lr_at(self.step, total_steps)
                self._set_lr(lr_now)
                self.opt.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    self.scaler.scale(obj).backward()
                    self.scaler.unscale_(self.opt)
                    gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                    ok = assert_finite(float(gn), "gradient norm")
                    if ok:
                        self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    obj.backward()
                    gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                    ok = assert_finite(float(gn), "gradient norm")
                    if ok:
                        self.opt.step()

                # A non-finite GRADIENT with a finite loss skips the optimizer step but left the
                # loop running: phase 2c spent 10 hours and ~$9 at 100% GPU taking zero steps,
                # because the guard above only watches the loss. Count these too.
                if not ok:
                    bad_grad += 1
                    if bad_grad > tol:
                        raise RuntimeError(
                            f"{bad_grad} non-finite gradients this epoch -- the model has "
                            "diverged, not hit an outlier. Resume from the last good checkpoint "
                            "at a lower learning rate "
                            "(wnca train --resume auto --set train.lr=...)."
                        )
                if self.grad_trace is not None:
                    self._write_grad_trace(float(gn), clean, lr_now)
                self.step += 1

                # Rolling mid-epoch checkpoint. 2c diverged at step 11,103 of a 7,122-step
                # epoch, and reproducing the onset cost ~4,000 steps from the epoch-1
                # checkpoint; a rolling save caps that loss. Fixed name `last.pt`, separate
                # from the timestamped `best_*.pt` selection checkpoints -- `--resume auto`
                # (pattern best_*.pt) never picks it up accidentally.
                if (self.rolling_ckpt is not None and cfg.train.ckpt_every_steps > 0
                        and self.step % cfg.train.ckpt_every_steps == 0):
                    save_checkpoint(self.rolling_ckpt, self.model, cfg, self.opt,
                                    epoch=self.epoch, step=self.step, metric=None)

            acc += clean
            nb += 1
            if _PREEMPTED["flag"]:
                break
        if skipped or bad_grad or bad_loss:
            print(f"  skipped {skipped}/{len(loader)} non-finite batches, "
                  f"{bad_grad} non-finite gradients, {bad_loss} finite-but-absurd batches")
        return {"loss": acc / max(nb, 1), "batches": nb, "skipped": skipped,
                "bad_grad": bad_grad, "bad_loss": bad_loss}

    @torch.no_grad()
    def selection_metric(self, loader) -> float:  # noqa: D401
        """ONE fixed selection metric for every phase: `ckpt_windows`-window rollout score on
        validation. Never compared across phase boundaries -- M1 incident 2."""
        self.model.eval()
        self.split = "val"
        acc, nb = 0.0, 0
        for prev, cur, tgt, idx in loader:
            batch = Batch(prev.to(self.device), cur.to(self.device), tgt.to(self.device), idx)
            pred, ovf, offset = self._forward(batch, self.cfg.train.ckpt_windows,
                                              self.cfg.ensemble.m_val, False)
            _, clean, _ = self._loss(pred, batch.tgt[:, offset:], ovf)
            acc += clean
            nb += 1
        return acc / max(nb, 1)

    @torch.no_grad()
    def anti_collapse_probe(self, loader, n_windows: int = 4) -> dict:
        """Three independent signals, logged from step zero of phase 3a as first-class metrics.

        1. Ensemble spread. If spread < 0.05 x RMSE by epoch 3, the noise is not reaching the
           dynamics -- stop and fix the conditioning rather than training through it.
        2. Zero-noise ablation: forward with z = 0 and compare to the ensemble mean. If they
           are identical, the noise is decorative.
        3. FiLM gradient norm, logged separately from the rest of the network.
        """
        if not self.cfg.model.stochastic:
            return {}
        self.model.eval()
        self.split = "val"
        prev, cur, tgt, idx = next(iter(loader))
        prev, cur, tgt = prev.to(self.device), cur.to(self.device), tgt.to(self.device)
        probe_forcing = self._forcing_for(Batch(prev, cur, tgt, idx), n_windows)
        # The probe cannot look further ahead than the loader supplies targets for.
        n_windows = max(1, min(n_windows, tgt.shape[1]))
        st = self._static_for(cur.shape[0])
        state = self.model.seed(cur)
        M = self.cfg.ensemble.m_val

        pred = self.model.rollout_ensemble(state, st, n_windows, prev_phys=prev, n_members=M,
                                           forcing=probe_forcing)
        truth = tgt[:, n_windows - 1]
        last = pred[:, :, n_windows - 1]  # [B, M, N, C]

        spread = ensemble_spread(last, self.area_w).mean().item()
        rmse = per_channel_rmse(last.mean(dim=1), truth, self.area_w).mean().item()
        ss = spread_skill_ratio(last, truth, self.area_w).mean().item()

        z0 = torch.zeros(cur.shape[0], 1, self.cfg.model.noise_dim, device=self.device)
        det = self.model.rollout_ensemble(state, st, n_windows, prev_phys=prev, n_members=1, z=z0,
                                          forcing=probe_forcing)
        zero_noise_gap = (det[:, 0, n_windows - 1] - last.mean(dim=1)).abs().mean().item()

        return {
            "spread": spread,
            "rmse": rmse,
            "spread_skill": ss,
            "spread_over_rmse": spread / max(rmse, 1e-12),
            "zero_noise_gap": zero_noise_gap,
        }

    def film_grad_norm(self) -> float:
        ps = getattr(self.model, "film_parameters", None)
        if ps is None:
            ps = getattr(self.model.update, "film_parameters", None)
        if ps is None:
            return float("nan")
        gs = [p.grad.norm().item() ** 2 for p in ps() if p.grad is not None]
        return float(np.sqrt(sum(gs))) if gs else 0.0


def fit(cfg: Config, model: nn.Module, mesh, cache, device: str, out_dir: Path,
        bands=None, tracker=None, resume: dict | None = None,
        optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Run the configured phase end to end. Returns the run history.

    `resume` is a checkpoint blob whose optimizer state has already been loaded into the
    trainer's optimizer by the caller. Restoring `step` matters as much as restoring the
    weights: the learning-rate schedule is a function of the step counter, so a resume that
    forgot it would silently restart the cosine decay from full learning rate and undo the
    run's convergence.
    """
    from ..data.dataset import make_loader

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(cfg, model, mesh, cache, device, bands, tracker)
    # WNCA_GRAD_TRACE=1 writes a per-step gradient trace next to the checkpoints. Off by
    # default: it is a divergence-forensics tool, not something a production run needs.
    if os.environ.get("WNCA_GRAD_TRACE"):
        trainer.grad_trace = out_dir / "grad_trace.jsonl"
        print(f"  gradient trace -> {trainer.grad_trace}")
    if cfg.train.ckpt_every_steps > 0:
        trainer.rolling_ckpt = out_dir / "last.pt"
    if optimizer is not None:
        trainer.opt = optimizer  # carries the restored optimizer state

    # Pushforward consumes one window before supervision starts, so it needs one more target.
    pf = 1 if cfg.train.pushforward else 0
    train_loader = make_loader(cache, "train", cfg, n_out=1 + pf, shuffle=True)
    val_loader = make_loader(cache, "val", cfg, n_out=1, shuffle=False)
    sel_loader = make_loader(cache, "val", cfg, n_out=cfg.train.ckpt_windows, shuffle=False,
                             batch_size=max(1, cfg.train.batch_size // 2),
                             subsample=cfg.train.ckpt_subsample)

    total_steps = max(len(train_loader) * cfg.train.epochs, 1)
    ckpt_path = timestamped_path(out_dir, cfg.phase)
    history: dict[str, list] = {"train": [], "val": [], "sel": [], "probe": []}

    start_epoch = 0
    if resume is not None:
        # `epoch` is the 0-indexed epoch the checkpoint was saved AT, so resume from the next.
        start_epoch = int(resume.get("epoch", -1)) + 1
        trainer.step = int(resume.get("step", 0))
        m = resume.get("metric")
        trainer.best = float(m) if m is not None and np.isfinite(m) else float("inf")
        history = resume.get("extra", {}).get("history", history)
        print(f"resuming at epoch {start_epoch + 1}/{cfg.train.epochs}  "
              f"(step {trainer.step}, best selection {trainer.best:.5f}, "
              f"lr {trainer._lr_at(trainer.step, total_steps):.2e})")
        if start_epoch >= cfg.train.epochs:
            print("  checkpoint is already at or past the configured epoch count; nothing to do")
            return {"history": history, "best": trainer.best, "checkpoint": str(ckpt_path)}

    sel_note = "" if cfg.train.ckpt_subsample >= 1.0 else f" on {cfg.train.ckpt_subsample:.0%} of val"
    print(f"phase {cfg.phase} | {sum(p.numel() for p in model.parameters()):,} params "
          f"| {len(train_loader)} train batches | selection = {cfg.train.ckpt_windows * 6}h rollout"
          f" ({len(sel_loader)} batches{sel_note})")
    print(f"checkpoints -> {ckpt_path.parent}")

    for ep in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        trainer.epoch = ep
        tr = trainer.run_epoch(train_loader, 1, True, total_steps)

        # SIGTERM during the training epoch: CHECKPOINT FIRST, score never. Spot gives ~30 s
        # of notice and validation + selection cost 3+ minutes -- the old order (validate,
        # then checkpoint) would lose the run's only state to a preemption, which is why
        # every 2c run paid on-demand rates (docs/cloud-compute-incidents.md section 4.1).
        if _PREEMPTED["flag"]:
            p = save_checkpoint(out_dir / "preempted.pt", model, cfg, trainer.opt,
                                epoch=ep, step=trainer.step, metric=None,
                                extra={"history": history})
            print(f"  preempted -- state saved to {p} (checkpoint written before scoring)")
            break

        va = trainer.run_epoch(val_loader, 1, False, split="val")
        sel = trainer.selection_metric(sel_loader)
        # Probe on the selection loader: it carries multi-window targets, and spread is only
        # meaningful a few windows out.
        probe = trainer.anti_collapse_probe(sel_loader, n_windows=min(4, cfg.train.ckpt_windows))

        history["train"].append(tr["loss"])
        history["val"].append(va["loss"])
        history["sel"].append(sel)
        history["probe"].append(probe)

        flag = ""
        if assert_finite(sel, "selection metric") and sel < trainer.best:
            trainer.best = sel
            save_checkpoint(ckpt_path, model, cfg, trainer.opt, epoch=ep, step=trainer.step,
                            metric=sel, extra={"probe": probe, "history": history})
            flag = " *"

        line = (f"epoch {ep + 1:>2}/{cfg.train.epochs}  train {tr['loss']:.5f}  "
                f"val {va['loss']:.5f}  sel {sel:.5f}  ({time.time() - t0:.0f}s){flag}")
        if probe:
            line += (f"\n           spread {probe['spread']:.4f}  spread/rmse "
                     f"{probe['spread_over_rmse']:.3f}  spread-skill {probe['spread_skill']:.3f}  "
                     f"zero-noise gap {probe['zero_noise_gap']:.2e}")
        print(line)

        if tracker is not None:
            tracker.log({"epoch": ep + 1, "train_loss": tr["loss"], "val_loss": va["loss"],
                         "selection": sel, **{f"probe/{k}": v for k, v in probe.items()}})

        if not np.isfinite(tr["loss"]):
            raise RuntimeError("training loss is not finite -- stopping")

        # Phase 3a gate, checked rather than merely documented.
        if probe and ep + 1 == 3 and probe["spread_over_rmse"] < 0.05:
            print("\n  GATE FAILED: spread < 0.05 x RMSE at epoch 3. The noise is not reaching "
                  "the dynamics.\n  Fix the conditioning rather than training through it "
                  "(docs/milestone-2-plan.md, anti-collapse instrumentation).")
            break

        if _PREEMPTED["flag"]:
            p = save_checkpoint(out_dir / "preempted.pt", model, cfg, trainer.opt,
                                epoch=ep, step=trainer.step, metric=sel,
                                extra={"history": history})
            print(f"  preempted -- state saved to {p}")
            break

    return {"history": history, "best": trainer.best, "checkpoint": str(ckpt_path)}
