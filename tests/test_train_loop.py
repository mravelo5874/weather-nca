"""Training-loop contracts: target alignment, conditioning, and the objective's two paths.

The pushforward test exists because the bug it guards against is silent: predictions shift one
window forward while the targets do not, so the model trains against the wrong lead and every
loss curve still looks perfectly reasonable.
"""
import dataclasses

import numpy as np
import pytest
import torch

from wnca.models.nca import build_model
from wnca.train.loop import Batch, Trainer


def _trainer(cfg, mesh, cache, **train_over):
    if train_over:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **train_over))
    torch.manual_seed(0)
    model = build_model(cfg, mesh)
    torch.nn.init.normal_(model.update.head.weight, std=0.01)
    return cfg, Trainer(cfg, model, mesh, cache, device="cpu")


def _batch(cfg, mesh, n_out, B=2):
    N = len(mesh["v"])
    torch.manual_seed(1)
    return Batch(torch.randn(B, N, cfg.c_phys), torch.randn(B, N, cfg.c_phys),
                 torch.randn(B, n_out, N, cfg.c_phys))


def test_no_pushforward_has_zero_offset(tiny_cfg, small_mesh, tiny_cache):
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, pushforward=False)
    _, _, offset = tr._forward(_batch(cfg, small_mesh, 1), 1, 1, train=True)
    assert offset == 0


def test_pushforward_reports_offset_one(tiny_cfg, small_mesh, tiny_cache):
    """Predictions start one window later, so the caller must drop one target."""
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, pushforward=True, noise_std=0.0)
    pred, _, offset = tr._forward(_batch(cfg, small_mesh, 2), 1, 1, train=True)
    assert offset == 1
    assert pred.shape[2] == 1


def test_pushforward_targets_line_up(tiny_cfg, small_mesh, tiny_cache):
    """With offset applied, prediction count must equal the sliced target count."""
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, pushforward=True, noise_std=0.0)
    b = _batch(cfg, small_mesh, 2)
    pred, ovf, offset = tr._forward(b, 1, 1, train=True)
    sliced = b.tgt[:, offset:]
    assert pred.shape[2] == sliced.shape[1], "pushforward misaligns predictions and targets"
    tr._loss(pred, sliced, ovf)  # must not raise


def test_pushforward_is_inactive_at_eval(tiny_cfg, small_mesh, tiny_cache):
    """Validation and the selection metric must not consume a window."""
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, pushforward=True)
    _, _, offset = tr._forward(_batch(cfg, small_mesh, 1), 1, 1, train=False)
    assert offset == 0


def test_pushforward_advances_the_state(tiny_cfg, small_mesh, tiny_cache):
    """It must actually step, not silently no-op."""
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, pushforward=True, noise_std=0.0)
    b = _batch(cfg, small_mesh, 2)
    with_pf, _, _ = tr._forward(b, 1, 1, train=True)
    tr.cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, pushforward=False))
    without, _, _ = tr._forward(b, 1, 1, train=True)
    assert not torch.allclose(with_pf, without, atol=1e-6)


def test_second_order_tendency_is_nonzero_under_pushforward(tiny_cfg, small_mesh, tiny_cache):
    """`prev` must be the field one window before the stepped state.

    Setting it to the stepped state's own physical channels would make the tendency identically
    zero and quietly disable second-order conditioning for every pushforward step.
    """
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, pushforward=True, noise_std=0.0)
    b = _batch(cfg, small_mesh, 2)
    captured = {}
    orig = tr.model._cond

    def spy(state, static, prev_phys):
        out = orig(state, static, prev_phys)
        captured.setdefault("tend", out[..., cfg.state.c_static:].abs().max().item())
        return out

    tr.model._cond = spy
    tr._forward(b, 1, 1, train=True)
    assert captured["tend"] > 0, "second-order tendency collapsed to zero"


def test_deterministic_loss_is_area_weighted_mse(tiny_cfg, small_mesh, tiny_cache):
    from wnca.losses.terms import area_weighted_mse
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache)
    b = _batch(cfg, small_mesh, 1)
    pred = torch.randn(2, 1, 1, len(small_mesh["v"]), cfg.c_phys)
    _, clean, parts = tr._loss(pred, b.tgt, torch.zeros(()))
    want = area_weighted_mse(pred[:, 0], b.tgt, tr.area_w, tr.chan_w).item()
    assert abs(clean - want) < 1e-6 and "field_mse" in parts


def test_stochastic_loss_uses_crps(tiny_cfg, small_mesh, tiny_cache):
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, stochastic=True))
    _, tr = _trainer(cfg, small_mesh, tiny_cache)
    b = _batch(cfg, small_mesh, 1)
    pred = torch.randn(2, 3, 1, len(small_mesh["v"]), cfg.c_phys)
    _, _, parts = tr._loss(pred, b.tgt, torch.zeros(()))
    assert "field_crps" in parts and "field_mse" not in parts


def test_warmup_then_cosine_schedule(tiny_cfg, small_mesh, tiny_cache):
    cfg, tr = _trainer(tiny_cfg, small_mesh, tiny_cache, warmup_steps=10, lr=1e-3)
    assert tr._lr_at(0, 100) < tr._lr_at(5, 100) < tr._lr_at(9, 100)
    assert abs(tr._lr_at(9, 100) - 1e-3) < 2e-4          # peak at end of warmup
    assert tr._lr_at(99, 100) < tr._lr_at(50, 100)        # decaying after
    assert tr._lr_at(99, 100) >= 0


# --- perturbation-growth summary -------------------------------------------------------

def test_settling_transient_is_separated_from_sustained_growth():
    """The phase-0 trace: a 4-window settling transient over a neutrally-stable tail.

    Averaging across the transient reported x1.087 ("AMPLIFYING") for an operator whose
    sustained rate is ~1.03. The transient must not contaminate the verdict.
    """
    from wnca.eval.perturbation import _settled
    ratios = [4.91, 1.86, 1.32, 1.13, 1.04, 1.02, 1.03, 1.06, 1.04, 1.02,
              1.04, 1.05, 1.02, 1.06, 1.02, 1.03, 1.03, 1.01, 1.02]
    out = _settled(ratios, threshold=1.05)
    assert out["transient_windows"] == 4, out
    assert 1.0 < out["sustained_growth"] < 1.05, out["sustained_growth"]


def test_genuinely_amplifying_operator_is_still_flagged():
    """A transient classification must not become a way to hide real amplification."""
    from wnca.eval.perturbation import _settled
    out = _settled([1.4] * 12, threshold=1.05)
    assert out["sustained_growth"] > 1.05
    assert out["transient_windows"] <= 6, "an all-amplifying trace was absorbed as transient"


def test_neutral_operator_has_no_transient():
    from wnca.eval.perturbation import _settled
    out = _settled([1.01] * 10, threshold=1.05)
    assert out["transient_windows"] == 0 and out["sustained_growth"] < 1.05
