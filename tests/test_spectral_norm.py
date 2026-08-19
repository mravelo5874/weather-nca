"""Spectral normalisation of the update MLP -- the structural fix for the phase-2c divergence.

Attempt 5's post-mortem (docs/cloud-compute-incidents.md): the 20-sub-step recurrence has a
stability threshold, and the input layer's weight norm ratchets across it linearly no matter
what the learning rate is. Spectral norm pins every hidden layer's sigma_max to ~1, so the
composed update map's gain is bounded by construction rather than by where training happens
to drift.
"""
import dataclasses

import numpy as np
import torch

from wnca.models.nca import build_model


def _cfg_with_sn(tiny_cfg, on=True):
    return dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(tiny_cfg.model, spectral_norm=on))


def _rand_io(cfg, mesh, B=2):
    N = len(mesh["v"])
    return (torch.randn(B, N, 4 * cfg.c_state), torch.randn(B, N, cfg.c_cond))


def test_spectral_norm_pins_hidden_layer_sigma(tiny_cfg, small_mesh):
    cfg = _cfg_with_sn(tiny_cfg)
    torch.manual_seed(0)
    model = build_model(cfg, small_mesh)
    # Inflate the raw weights so an unwrapped layer would sit far above sigma 1.
    for layer in model.update.layers:
        torch.nn.init.normal_(layer.weight_orig, std=2.0)
    perceived, cond = _rand_io(cfg, small_mesh)
    model.train()
    for _ in range(5):  # the power iteration converges over a few forwards
        model.update(perceived, cond)
    for i, layer in enumerate(model.update.layers):
        sigma = float(torch.linalg.matrix_norm(layer.weight.float(), 2))
        assert sigma <= 1.0 + 0.05, f"layers.{i} sigma_max {sigma} is not pinned"


def test_zero_init_head_stays_unwrapped_and_finite(tiny_cfg, small_mesh):
    """spectral_norm's power iteration divides by the weight's norm, which NaNs on an
    exactly-zero weight (verified empirically, torch 2.6). The head is zero-init by design
    -- the untrained model is exactly the identity map -- so it must NOT be wrapped, and a
    fresh spectral-normed model must still produce finite output from step zero."""
    cfg = _cfg_with_sn(tiny_cfg)
    model = build_model(cfg, small_mesh)
    assert not hasattr(model.update.head, "weight_orig"), "head must not be wrapped"
    assert not hasattr(model.update.film, "weight_orig"), "film must stay free"
    perceived, cond = _rand_io(cfg, small_mesh)
    out = model.update(perceived, cond)
    assert torch.isfinite(out).all()


def test_composed_map_gain_is_bounded(tiny_cfg, small_mesh, forcing_for):
    """The property the fix exists for: with identical inflated raw weights, the plain
    model's forecast step blows up while the spectral-normed one stays bounded."""
    torch.manual_seed(0)
    plain = build_model(_cfg_with_sn(tiny_cfg, False), small_mesh)
    torch.manual_seed(0)
    sn = build_model(_cfg_with_sn(tiny_cfg, True), small_mesh)
    for layer in plain.update.layers:
        torch.nn.init.normal_(layer.weight, std=2.0)
    for layer in sn.update.layers:
        torch.nn.init.normal_(layer.weight_orig, std=2.0)
    torch.nn.init.normal_(plain.update.head.weight, std=0.1)
    torch.nn.init.normal_(sn.update.head.weight, std=0.1)

    cfg = tiny_cfg
    B, N = 2, len(small_mesh["v"])
    # Legacy spectral_norm runs its power iteration in train mode only; converge it on the
    # inflated weights before measuring, exactly as real training would every step.
    perceived, cond = _rand_io(cfg, small_mesh)
    sn.train()
    for _ in range(5):
        sn.update(perceived, cond)

    torch.manual_seed(1)
    phys = torch.randn(B, N, cfg.c_phys)
    static = torch.randn(B, N, cfg.state.c_static)
    forcing = forcing_for(cfg, small_mesh, B, 1)[:, 0]

    plain.eval()
    sn.eval()
    with torch.no_grad():
        out_plain = plain.forecast_step(plain.seed(phys), static, forcing=forcing)
        out_sn = sn.forecast_step(sn.seed(phys), static, forcing=forcing)

    rel_sn = float((out_sn - sn.seed(phys)).norm() / sn.seed(phys).norm())
    assert torch.isfinite(out_sn).all(), "bounded map produced non-finite output"
    assert rel_sn < 1.0, f"spectral-normed map changed the state by {rel_sn:.2f}x"
    rel_plain = float((out_plain - plain.seed(phys)).norm() / plain.seed(phys).norm())
    assert not np.isfinite(rel_plain) or rel_plain > 10.0, (
        f"control did not explode (rel change {rel_plain:.2f}) -- the test is not exercising "
        "the instability it claims to bound")


def test_spectral_norm_changes_arch_hash(tiny_cfg):
    """The flag changes state-dict keys (parametrized weights + power-iteration buffers), so
    checkpoints must not load across it. Follows the solar_forcing pattern: included only
    when enabled, so pre-existing checkpoints stay loadable."""
    assert _cfg_with_sn(tiny_cfg, True).arch_hash() != _cfg_with_sn(tiny_cfg, False).arch_hash()


def test_checkpoint_roundtrip_with_spectral_norm(tiny_cfg, small_mesh, tmp_path):
    from wnca.train.checkpoint import load_checkpoint, save_checkpoint

    cfg = _cfg_with_sn(tiny_cfg)
    model = build_model(cfg, small_mesh)
    # The load path asserts a trained head; a fresh head is exactly zero by design.
    torch.nn.init.normal_(model.update.head.weight, std=0.01)
    p = save_checkpoint(tmp_path / "m.pt", model, cfg)
    fresh = build_model(cfg, small_mesh)
    load_checkpoint(p, fresh, cfg)
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)


def test_training_epoch_runs_with_spectral_norm(tiny_cfg, small_mesh, tiny_cache):
    """Integration: spectral norm + gradient checkpointing + the loss path, one real epoch.
    Grad-checkpointing recomputes the sub-steps in backward, which re-runs the power
    iteration -- the combination must not trip autograd's version counters."""
    from wnca.data.dataset import make_loader
    from wnca.train.loop import Trainer

    cfg = _cfg_with_sn(tiny_cfg)
    assert cfg.model.grad_ckpt, "this test exists to exercise the checkpointing interaction"
    torch.manual_seed(0)
    model = build_model(cfg, small_mesh)
    torch.nn.init.normal_(model.update.head.weight, std=0.01)
    tr = Trainer(cfg, model, small_mesh, tiny_cache, device="cpu")
    out = tr.run_epoch(make_loader(tiny_cache, "train", cfg, n_out=1, shuffle=False),
                       1, True, 10)
    assert np.isfinite(out["loss"])
    assert out["skipped"] == 0 and out["bad_grad"] == 0 and out["bad_loss"] == 0
