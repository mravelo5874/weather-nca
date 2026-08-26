"""Regression tests for the per-member band-energy stage.

This stage runs only when `cfg.model.stochastic`, so every phase before 3a skipped it entirely
and no test covered it with solar forcing on. `member_spectra` never passed `forcing=` to the
rollout, and the model refuses to roll out without it -- deliberately, so nothing can silently
train or score on permanent night.

Found by smoking `configs/phase3a_probe.yaml`, which is the first config that is both stochastic
and forced. Left alone it would have crashed phase 3a at EVAL time, i.e. after ~9.5 h of paid
training per seed had already been spent.
"""
import dataclasses

import pytest
import torch

from wnca.eval.spectrum import member_spectra
from wnca.mesh.spectral import build_band_filters
from wnca.models.nca import WeatherNCA
from wnca.models.perception import MeshPerception


def _forced_stochastic(cfg):
    return dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, stochastic=True),
        state=dataclasses.replace(cfg.state, solar_forcing=True),
    )


def _model(cfg, mesh):
    torch.manual_seed(0)
    m = WeatherNCA(cfg, MeshPerception(mesh))
    torch.nn.init.normal_(m.update.head.weight, std=0.01)
    return m.eval()


def test_member_spectra_supplies_solar_forcing(tiny_cfg, small_mesh, tiny_cache):
    """The real regression: with forcing on, this used to raise from inside the rollout."""
    cfg = _forced_stochastic(tiny_cfg)
    bands = build_band_filters(small_mesh, cfg)
    out = member_spectra(_model(cfg, small_mesh), cfg, tiny_cache, bands, split="test",
                         lead_windows=2, n_starts=2, n_members=2, mesh=small_mesh)
    assert out["member_ratio"].shape == out["mean_ratio"].shape
    assert torch.isfinite(torch.from_numpy(out["member_ratio"])).all()


def test_member_spectra_says_what_is_missing_when_mesh_is_omitted(tiny_cfg, small_mesh,
                                                                  tiny_cache):
    """Forcing needs the mesh to build. Fail with the reason, not with a rollout error two
    frames deeper -- the same contract `perturbation_growth` already honours."""
    cfg = _forced_stochastic(tiny_cfg)
    bands = build_band_filters(small_mesh, cfg)
    with pytest.raises(ValueError, match="mesh"):
        member_spectra(_model(cfg, small_mesh), cfg, tiny_cache, bands, split="test",
                       lead_windows=2, n_starts=2, n_members=2)


def test_member_spectra_still_runs_unforced(tiny_cfg, small_mesh, tiny_cache):
    """Forcing off must not require a mesh: the fix must not make the unforced path stricter."""
    cfg = dataclasses.replace(
        tiny_cfg,
        model=dataclasses.replace(tiny_cfg.model, stochastic=True),
        state=dataclasses.replace(tiny_cfg.state, solar_forcing=False),
    )
    bands = build_band_filters(small_mesh, cfg)
    out = member_spectra(_model(cfg, small_mesh), cfg, tiny_cache, bands, split="test",
                         lead_windows=2, n_starts=2, n_members=2)
    assert out["member_ratio"].shape[0] == len(bands.edges) - 1
