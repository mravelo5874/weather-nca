"""The untrained-model guard.

This is a regression test for the specific incident that wasted three M1 evaluation rounds: a
zero-init identity model reproduced persistence exactly, and the evaluation table looked
plausible until someone noticed the skill column was 0.0% at every lead.

Build, train one step, save, reload, assert the head norm is non-zero and predictions differ
from the identity map. Plus the other three checkpoint incidents: architecture drift,
`best.pt` clobbering, and non-finite selection metrics.
"""
import dataclasses
import time

import numpy as np
import pytest
import torch

from wnca.data.forcing import SolarForcing, synthetic_times
from wnca.models.nca import WeatherNCA, build_model
from wnca.train.checkpoint import (
    assert_finite, latest_checkpoint, load_checkpoint, save_checkpoint, timestamped_path, warm_start,
)


def _forcing(cfg, mesh, B, W, start=3):
    """Solar forcing of the right shape; the model refuses to run without it."""
    if not cfg.state.solar_forcing:
        return None
    return SolarForcing(synthetic_times(start + W + 8), mesh).window(
        torch.arange(start, start + B), W)


def _trained_one_step(cfg, mesh):
    torch.manual_seed(0)
    model = build_model(cfg, mesh)
    N = len(mesh["v"])
    cur = torch.randn(2, N, cfg.c_phys)
    prev = torch.randn(2, N, cfg.c_phys)
    st = torch.randn(2, N, cfg.state.c_static)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    pred = model.rollout(model.seed(cur), st, 1, prev_phys=prev,
                         forcing=_forcing(cfg, mesh, 2, 1))
    (pred - torch.randn_like(pred)).pow(2).mean().backward()
    opt.step()
    return model, opt


def test_untrained_model_is_the_identity_map(tiny_cfg, small_mesh):
    """The premise of the whole guard: a zero-init head really does reproduce persistence."""
    model = build_model(tiny_cfg, small_mesh)
    N = len(small_mesh["v"])
    cur = torch.randn(1, N, tiny_cfg.c_phys)
    with torch.no_grad():
        out = model.rollout(model.seed(cur), torch.randn(1, N, tiny_cfg.state.c_static), 1,
                            forcing=_forcing(tiny_cfg, small_mesh, 1, 1))
    assert torch.allclose(out[:, 0], cur, atol=1e-6), "zero-init model is not the identity map"


def test_save_reload_roundtrip_and_head_norm(tiny_cfg, small_mesh, tmp_path):
    model, opt = _trained_one_step(tiny_cfg, small_mesh)
    assert model.update.head.weight.norm().item() > 0

    path = save_checkpoint(tmp_path / "ck.pt", model, tiny_cfg, opt, epoch=1, step=1, metric=0.5)
    fresh = build_model(tiny_cfg, small_mesh)
    blob = load_checkpoint(path, fresh, tiny_cfg)

    assert blob["epoch"] == 1 and blob["metric"] == 0.5
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)


def test_loading_an_untrained_checkpoint_raises(tiny_cfg, small_mesh, tmp_path):
    """THE test. An untrained model must not load silently into an evaluation path."""
    untrained = build_model(tiny_cfg, small_mesh)
    path = save_checkpoint(tmp_path / "untrained.pt", untrained, tiny_cfg)
    with pytest.raises(RuntimeError, match="untrained"):
        load_checkpoint(path, build_model(tiny_cfg, small_mesh), tiny_cfg)


def test_reloaded_model_differs_from_identity(tiny_cfg, small_mesh, tmp_path):
    """Head norm > 0 is necessary but not sufficient -- check the behaviour, not just the norm."""
    model, opt = _trained_one_step(tiny_cfg, small_mesh)
    path = save_checkpoint(tmp_path / "ck.pt", model, tiny_cfg, opt)
    fresh = build_model(tiny_cfg, small_mesh)
    load_checkpoint(path, fresh, tiny_cfg)

    N = len(small_mesh["v"])
    cur = torch.randn(1, N, tiny_cfg.c_phys)
    with torch.no_grad():
        out = fresh.rollout(fresh.seed(cur), torch.randn(1, N, tiny_cfg.state.c_static), 1,
                            forcing=_forcing(tiny_cfg, small_mesh, 1, 1))
    assert not torch.allclose(out[:, 0], cur, atol=1e-6), "reloaded model still an identity map"


def test_architecture_mismatch_raises(tiny_cfg, small_mesh, tmp_path):
    """M1 incident: evaluating a checkpoint against a config whose architecture had drifted."""
    model, _ = _trained_one_step(tiny_cfg, small_mesh)
    path = save_checkpoint(tmp_path / "ck.pt", model, tiny_cfg)
    other = dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(tiny_cfg.model, hidden_dim=tiny_cfg.model.hidden_dim * 2)
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_checkpoint(path, build_model(other, small_mesh), other)


def test_arch_hash_ignores_non_architectural_fields(tiny_cfg):
    """Changing the learning rate must not invalidate a checkpoint."""
    other = dataclasses.replace(tiny_cfg, train=dataclasses.replace(tiny_cfg.train, lr=123.0))
    assert other.arch_hash() == tiny_cfg.arch_hash()


def test_arch_hash_catches_channel_changes(tiny_cfg):
    other = dataclasses.replace(
        tiny_cfg, variables=dataclasses.replace(tiny_cfg.variables, levels=(500,))
    )
    assert other.arch_hash() != tiny_cfg.arch_hash()


def test_timestamped_paths_do_not_collide(tmp_path):
    """`best.pt` as a single fixed path clobbered comparisons in M1."""
    a = timestamped_path(tmp_path, "p1")
    time.sleep(1.05)
    b = timestamped_path(tmp_path, "p1")
    assert a != b


def test_latest_checkpoint_picks_newest(tiny_cfg, small_mesh, tmp_path):
    model, _ = _trained_one_step(tiny_cfg, small_mesh)
    save_checkpoint(tmp_path / "best_a.pt", model, tiny_cfg)
    time.sleep(0.05)
    newest = save_checkpoint(tmp_path / "best_b.pt", model, tiny_cfg)
    assert latest_checkpoint(tmp_path) == newest


def test_optimizer_and_rng_state_survive(tiny_cfg, small_mesh, tmp_path):
    """Spot resumption needs optimizer state, RNG state, epoch and step -- not just weights."""
    model, opt = _trained_one_step(tiny_cfg, small_mesh)
    path = save_checkpoint(tmp_path / "ck.pt", model, tiny_cfg, opt, epoch=3, step=42)
    fresh = build_model(tiny_cfg, small_mesh)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2)
    blob = load_checkpoint(path, fresh, tiny_cfg, fresh_opt)
    assert blob["step"] == 42
    assert fresh_opt.state_dict()["state"], "optimizer state did not restore"


def test_warm_start_accepts_new_film_parameters(tiny_cfg, small_mesh, tmp_path):
    """Phase 3a warm-starts a stochastic model from a deterministic checkpoint."""
    det, _ = _trained_one_step(tiny_cfg, small_mesh)
    path = save_checkpoint(tmp_path / "det.pt", det, tiny_cfg)
    stoch_cfg = dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(tiny_cfg.model, stochastic=True)
    )
    stoch = build_model(stoch_cfg, small_mesh)
    warm_start(path, stoch, stoch_cfg)
    assert torch.allclose(stoch.update.head.weight, det.update.head.weight)


def test_warm_start_from_untrained_raises(tiny_cfg, small_mesh, tmp_path):
    path = save_checkpoint(tmp_path / "u.pt", build_model(tiny_cfg, small_mesh), tiny_cfg)
    with pytest.raises(RuntimeError, match="untrained"):
        warm_start(path, build_model(tiny_cfg, small_mesh), tiny_cfg)


def test_finiteness_guard():
    assert assert_finite(0.5, "x")
    assert not assert_finite(float("nan"), "x")
    assert not assert_finite(float("inf"), "x")


# --- resume after a crash --------------------------------------------------------------

def test_resume_restores_step_epoch_and_best(tiny_cfg, small_mesh, tmp_path):
    """Restoring `step` matters as much as restoring weights.

    The LR schedule is a function of the step counter, so a resume that forgot it would
    silently restart the cosine decay at full learning rate and undo the run's convergence.
    """
    from wnca.train.loop import Trainer
    model, opt = _trained_one_step(tiny_cfg, small_mesh)
    path = save_checkpoint(tmp_path / "best_x.pt", model, tiny_cfg, opt,
                           epoch=3, step=1234, metric=0.042)

    fresh = build_model(tiny_cfg, small_mesh)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=tiny_cfg.train.lr)
    blob = load_checkpoint(path, fresh, tiny_cfg, fresh_opt)

    assert blob["epoch"] == 3 and blob["step"] == 1234 and blob["metric"] == 0.042

    # Losing `step` would restart the cosine decay. Check the schedule actually depends on it:
    # past warmup it must decay monotonically, so a resumed step sits below an earlier one.
    tr = Trainer(tiny_cfg, fresh, small_mesh, _StubCache(small_mesh, tiny_cfg), device="cpu")
    w = tiny_cfg.train.warmup_steps
    assert tr._lr_at(4000, 5000) < tr._lr_at(1234, 5000) < tr._lr_at(w, 5000)


class _StubCache:
    """Minimal stand-in so Trainer can be constructed without building a real cache."""

    def __init__(self, mesh, cfg):
        self.static = np.zeros((len(mesh["v"]), cfg.state.c_static), dtype=np.float32)

    def times(self, split):
        return synthetic_times(64)


def test_resume_auto_picks_the_newest_checkpoint(tiny_cfg, small_mesh, tmp_path):
    from wnca.train.phases import resolve_resume
    model, _ = _trained_one_step(tiny_cfg, small_mesh)
    save_checkpoint(tmp_path / "best_old.pt", model, tiny_cfg)
    time.sleep(0.05)
    newest = save_checkpoint(tmp_path / "best_new.pt", model, tiny_cfg)
    assert resolve_resume("auto", tmp_path, tiny_cfg) == newest


def test_resume_explicit_path_is_honoured(tiny_cfg, small_mesh, tmp_path):
    from wnca.train.phases import resolve_resume
    model, _ = _trained_one_step(tiny_cfg, small_mesh)
    p = save_checkpoint(tmp_path / "specific.pt", model, tiny_cfg)
    assert resolve_resume(str(p), tmp_path, tiny_cfg) == p


def test_resume_missing_path_raises(tiny_cfg, tmp_path):
    from wnca.train.phases import resolve_resume
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_resume(str(tmp_path / "nope.pt"), tmp_path, tiny_cfg)


def test_resume_auto_with_no_checkpoints_raises(tiny_cfg, tmp_path):
    from wnca.train.phases import resolve_resume
    import dataclasses
    cfg = dataclasses.replace(
        tiny_cfg, tracking=dataclasses.replace(tiny_cfg.tracking, out_dir=str(tmp_path / "empty"))
    )
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        resolve_resume("auto", tmp_path / "empty", cfg)


def test_history_survives_the_checkpoint(tiny_cfg, small_mesh, tmp_path):
    """A resumed run must be able to continue the training curve, not restart it."""
    model, opt = _trained_one_step(tiny_cfg, small_mesh)
    hist = {"train": [0.5, 0.4], "val": [0.6, 0.5], "sel": [0.7, 0.6], "probe": [{}, {}]}
    path = save_checkpoint(tmp_path / "b.pt", model, tiny_cfg, opt, epoch=1, step=10,
                           metric=0.6, extra={"history": hist})
    blob = load_checkpoint(path, build_model(tiny_cfg, small_mesh), tiny_cfg)
    assert blob["extra"]["history"]["sel"] == [0.7, 0.6]

def test_phase3a_can_warm_start_from_phase2c():
    """3a fine-tunes a 2c checkpoint, so its architecture must match 2c's exactly.

    It did not. `phase3a_crps.yaml` set only `stochastic` and `noise_dim`, inheriting
    `spectral_norm: false` from base against 2c's true. A 2c checkpoint carries spectral-norm
    buffers (weight_orig / weight_u / weight_v, 12 keys at n_layers=4), `warm_start` raises on
    unexpected keys, and 3a therefore **could not start at all** -- a guaranteed crash on first
    launch that nothing would have caught before an instance was paid for.

    arch_hash is the right thing to assert: it is exactly the fingerprint `load_checkpoint`
    checks, so if these two ever diverge again the warm start breaks and this fails first.
    """
    from wnca.config import load_config

    nca = load_config("configs/phase2c_full.yaml")
    crps = load_config("configs/phase3a_crps.yaml")
    assert crps.arch_hash() == nca.arch_hash(), (
        f"3a arch {crps.arch_hash()} != 2c arch {nca.arch_hash()} -- warm start will raise. "
        f"spectral_norm 2c={nca.model.spectral_norm} 3a={crps.model.spectral_norm}")


def test_phase3a_keeps_the_weight_decay_that_fixed_phase2c():
    """The silent half of the same bug. `weight_decay` inherits 1e-5 from base, which phase 2c
    measured as ~11,000x too weak to oppose the weight-norm ratchet that destroyed four runs.
    Unlike the arch mismatch this would not crash -- it would quietly undo the stability fix
    partway through a 15-epoch run."""
    from wnca.config import load_config

    nca = load_config("configs/phase2c_full.yaml")
    crps = load_config("configs/phase3a_crps.yaml")
    assert crps.train.weight_decay == nca.train.weight_decay, (
        f"3a weight_decay {crps.train.weight_decay} != 2c's {nca.train.weight_decay}")


def test_phase3a_probe_resolves_onto_the_cache_that_already_exists():
    """The probe's whole point is that it needs no new download.

    `cache_tag()` hashes the years of ALL THREE splits together, so changing any one of them
    invalidates train and val as well -- a ~6.3 GB re-download for a run budgeted at $17. If an
    edit drifts these years, this fails before anyone waits on a cache.
    """
    from wnca.config import load_config
    from wnca.data.cache import cache_tag

    probe = load_config("configs/phase3a_probe.yaml")
    assert tuple(probe.data.train_years) == (2015, 2016)
    assert tuple(probe.data.val_years) == (2017,)
    assert tuple(probe.data.test_years) == (2018,)
    assert cache_tag(probe) == "era5_sub5_c28_2f61cc03ea", (
        "probe no longer points at the cached 2-year split; it would need a fresh download")


def test_phase3a_probe_changes_only_the_split():
    """The probe must differ from its parent in the DATA and nothing else.

    It exists to be a cheap go/no-go for `phase3a_crps.yaml`, which is worthless if the two
    drift apart: a probe that passes on different training decisions than the run it is
    clearing has measured nothing about that run. `extends:` gives inheritance, this asserts it
    was not overridden.
    """
    from wnca.config import load_config

    parent = load_config("configs/phase3a_crps.yaml")
    probe = load_config("configs/phase3a_probe.yaml")

    assert probe.arch_hash() == parent.arch_hash()
    for field in ("epochs", "pushforward", "lr", "weight_decay", "warm_start", "seed"):
        assert getattr(probe.train, field) == getattr(parent.train, field), field
    for field in ("m_train", "m_val", "m_test"):
        assert getattr(probe.ensemble, field) == getattr(parent.ensemble, field), field
    assert probe.data.train_years != parent.data.train_years


def test_phase3a_warm_start_points_at_a_real_checkpoint():
    """`warm_start: null` meant 3a would train from scratch while claiming to fine-tune 2c.

    Asserting the file exists would fail on a fresh clone, so this asserts the config names a
    2c checkpoint -- the thing that was actually missing.
    """
    from wnca.config import load_config

    ws = load_config("configs/phase3a_crps.yaml").train.warm_start
    assert ws, "3a has no warm_start; it would train from scratch, not fine-tune 2c"
    assert "phase2c" in ws and ws.endswith(".pt"), ws


def test_smoke_never_warm_starts():
    """Smoke builds a 64-dim, n_sub=3 model; no full-size checkpoint fits it.

    Without this, `make smoke` on any warm-starting phase (3a, 3b) fails for a reason that has
    nothing to do with the config being wrong -- which defeats the point of smoking before a
    phase goes to cloud.
    """
    from wnca.config import load_config

    assert load_config("configs/phase3a_probe.yaml", smoke=True).train.warm_start is None
    assert load_config("configs/phase3a_crps.yaml", smoke=True).train.warm_start is None
