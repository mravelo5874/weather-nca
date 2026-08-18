"""Mixed precision, which is dead code locally and load-bearing on the cloud.

`train.amp` was wired up early and never exercised: the local GTX 1660 Ti is Turing, which has
no tensor cores, so AMP buys nothing here and no config ever set it. That made it exactly the
kind of path that works right up until it is needed.

It did not work. `torch.sparse.mm` has no half-precision CUDA kernel --
`RuntimeError: "addmm_sparse_cuda" not implemented for 'Half'` -- so every sparse operator in
the project (mesh perception, the Chebyshev band filters, the WB2 scoring regrid) crashed the
moment autocast was enabled. It would have failed on the first step of the first cloud run.

Perception, band filters and the regrid now force fp32 internally. The cost is small: the
update MLP is ~80% of a sub-step and is dense, so it still gets the tensor cores.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wnca.models.nca import build_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast fp16 needs CUDA")


@CUDA
def test_perception_survives_autocast(tiny_cfg, small_mesh):
    """The regression test for the bug: sparse ops under autocast."""
    from wnca.models.perception import MeshPerception

    p = MeshPerception(small_mesh).cuda()
    x = torch.randn(2, len(small_mesh["v"]), tiny_cfg.c_state, device="cuda")
    with torch.autocast("cuda"):
        out = p(x)
    assert torch.isfinite(out).all()
    assert out.shape == (2, len(small_mesh["v"]), 4 * tiny_cfg.c_state)


@CUDA
def test_band_filters_survive_autocast(small_mesh, tiny_cfg):
    """The phase 3b spectral loss path."""
    from wnca.mesh.operators import laplacian_matrix
    from wnca.mesh.spectral import BandFilters

    bands = BandFilters(laplacian_matrix(small_mesh), small_mesh["area"],
                        n_bands=3, order=8, device="cuda")
    x = torch.randn(2, len(small_mesh["v"]), 1, device="cuda")
    with torch.autocast("cuda"):
        e = bands.log_energies(x)
    assert torch.isfinite(e).all() and e.shape == (2, 3, 1)


@CUDA
def test_wb2_regrid_survives_autocast(small_mesh, tiny_cfg):
    """The scoring path -- would have failed only at the very end of a run."""
    import dataclasses

    from wnca.eval.wb2 import WB2Scorer

    cfg = dataclasses.replace(tiny_cfg, eval=dataclasses.replace(tiny_cfg.eval, wb2_grid="5.625"))
    sc = WB2Scorer(cfg, small_mesh, device="cuda")
    x = torch.randn(1, len(small_mesh["v"]), cfg.c_phys, device="cuda")
    with torch.autocast("cuda"):
        g = sc.to_grid(x)
    assert torch.isfinite(g).all()


@CUDA
def test_full_rollout_under_autocast(tiny_cfg, small_mesh, forcing_for):
    model = build_model(tiny_cfg, small_mesh, device="cuda")
    N = len(small_mesh["v"])
    cur = torch.randn(2, N, tiny_cfg.c_phys, device="cuda")
    st = torch.randn(2, N, tiny_cfg.state.c_static, device="cuda")
    f = forcing_for(tiny_cfg, small_mesh, 2, 2)
    f = f.cuda() if f is not None else None
    with torch.autocast("cuda"):
        out = model.rollout(model.seed(cur), st, 2, forcing=f)
    assert torch.isfinite(out).all()


@CUDA
def test_amp_agrees_with_fp32(tiny_cfg, small_mesh, forcing_for):
    """AMP must be a speed/precision trade, not a different model.

    Loose tolerance on purpose -- fp16 has ~3 decimal digits, and this rolls through several
    sub-steps. The point is agreement to fp16's precision, not bitwise equality.
    """
    torch.manual_seed(0)
    model = build_model(tiny_cfg, small_mesh, device="cuda")
    torch.nn.init.normal_(model.update.head.weight, std=0.01)
    N = len(small_mesh["v"])
    cur = torch.randn(2, N, tiny_cfg.c_phys, device="cuda")
    st = torch.randn(2, N, tiny_cfg.state.c_static, device="cuda")
    f = forcing_for(tiny_cfg, small_mesh, 2, 1)
    f = f.cuda() if f is not None else None

    with torch.no_grad():
        ref = model.rollout(model.seed(cur), st, 1, forcing=f)
        with torch.autocast("cuda"):
            amp = model.rollout(model.seed(cur), st, 1, forcing=f)
    rel = (amp.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    assert rel < 5e-2, f"AMP diverges from fp32 by {rel:.4f} relative -- not a precision trade"


@CUDA
def test_amp_training_step_updates_weights(tiny_cfg, small_mesh, tiny_cache):
    """GradScaler + the finiteness guard must still let a step land."""
    import dataclasses

    from wnca.data.dataset import make_loader
    from wnca.train.loop import Trainer

    cfg = dataclasses.replace(tiny_cfg, train=dataclasses.replace(tiny_cfg.train, amp=True))
    torch.manual_seed(0)
    model = build_model(cfg, small_mesh, device="cuda")
    torch.nn.init.normal_(model.update.head.weight, std=0.01)
    tr = Trainer(cfg, model, small_mesh, tiny_cache, device="cuda")
    # GradScaler exists only to manage fp16 overflow; bf16 has fp32's exponent range and
    # needs none, so the scaler is deliberately absent on Ampere and later.
    native_bf16 = torch.cuda.get_device_capability()[0] >= 8
    assert (tr.scaler is None) == native_bf16, f"scaler={tr.scaler} for dtype={tr.amp_dtype}"
    assert tr.amp_dtype is (torch.bfloat16 if native_bf16 else torch.float16)

    before = model.update.head.weight.detach().clone()
    out = tr.run_epoch(make_loader(tiny_cache, "train", cfg, n_out=1, shuffle=False), 1, True, 10)
    assert out["skipped"] == 0
    assert not torch.equal(before, model.update.head.weight), "no weight update under AMP"


@CUDA
def test_pushforward_under_amp_still_builds_a_graph(tiny_cfg, small_mesh, tiny_cache):
    """Pushforward and AMP each worked alone and broke together.

    Running a module under `no_grad` INSIDE an autocast region populates autocast's weight
    cache with fp16 copies that do not track gradients. The subsequent grad-enabled forward
    reuses those cached weights, so its output has no grad_fn and backward dies with
    "element 0 of tensors does not require grad and does not have a grad_fn".

    This reached a cloud instance and killed phase 2c on its first step -- the feature matrix
    was tested one axis at a time, and the interaction was the thing that mattered.
    """
    import dataclasses

    from wnca.data.dataset import make_loader
    from wnca.train.loop import Trainer

    cfg = dataclasses.replace(
        tiny_cfg,
        train=dataclasses.replace(tiny_cfg.train, amp=True, pushforward=True),
    )
    torch.manual_seed(0)
    model = build_model(cfg, small_mesh, device="cuda")
    torch.nn.init.normal_(model.update.head.weight, std=0.01)
    tr = Trainer(cfg, model, small_mesh, tiny_cache, device="cuda")

    before = model.update.head.weight.detach().clone()
    # n_out+1 targets: pushforward consumes one window before supervision starts.
    out = tr.run_epoch(make_loader(tiny_cache, "train", cfg, n_out=2, shuffle=False), 1, True, 10)

    assert out["skipped"] == 0, "batches were skipped -- loss went non-finite under AMP"
    assert not torch.equal(before, model.update.head.weight), "no weight update"


@CUDA
def test_every_training_feature_combination_runs(tiny_cfg, small_mesh, tiny_cache):
    """The feature matrix, exercised as a matrix.

    amp x pushforward x solar_forcing is 8 combinations; testing them one axis at a time missed
    the one that broke. Cheap insurance against the next interaction.
    """
    import dataclasses
    import itertools

    from wnca.data.dataset import make_loader
    from wnca.train.loop import Trainer

    for amp, pf, solar in itertools.product((False, True), repeat=3):
        cfg = dataclasses.replace(
            tiny_cfg,
            state=dataclasses.replace(tiny_cfg.state, solar_forcing=solar),
            train=dataclasses.replace(tiny_cfg.train, amp=amp, pushforward=pf),
        )
        torch.manual_seed(0)
        model = build_model(cfg, small_mesh, device="cuda")
        torch.nn.init.normal_(model.update.head.weight, std=0.01)
        tr = Trainer(cfg, model, small_mesh, tiny_cache, device="cuda")
        n_out = 2 if pf else 1
        out = tr.run_epoch(make_loader(tiny_cache, "train", cfg, n_out=n_out, shuffle=False),
                           1, True, 10)
        assert out["skipped"] == 0, f"amp={amp} pushforward={pf} solar={solar}: batches skipped"
        assert np.isfinite(out["loss"]), f"amp={amp} pushforward={pf} solar={solar}: non-finite"
