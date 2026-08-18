"""WeatherBench-2 scoring: the module that produces externally comparable numbers.

It had no tests, which is the wrong place to have none. Everything else in the project is
scored against persistence and climatology computed by the same code, so a systematic error
would cancel. `eval/wb2.py` is the one path where a bug produces a number that gets compared to
somebody else's published result -- and a biased regrid or a mis-weighted average would look
entirely plausible.

Scored quantities are checked against closed-form answers wherever one exists.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from wnca.eval.wb2 import WB2Scorer, compare_to_reference, lead_to_window
from wnca.mesh.regrid import latlon_cell_area, target_grid


@pytest.fixture(scope="module")
def scorer(small_mesh, tiny_cfg):
    cfg = dataclasses.replace(tiny_cfg, eval=dataclasses.replace(tiny_cfg.eval, wb2_grid="5.625"))
    return WB2Scorer(cfg, small_mesh), cfg


def test_lead_to_window():
    assert lead_to_window(24) == 4 and lead_to_window(6) == 1 and lead_to_window(360) == 60


def test_regrid_preserves_a_constant(scorer, small_mesh):
    """A constant field must survive the regrid exactly, or every RMSE carries a bias."""
    sc, cfg = scorer
    x = torch.full((1, len(small_mesh["v"]), cfg.c_phys), 3.25)
    g = sc.to_grid(x)
    assert torch.allclose(g, torch.full_like(g, 3.25), atol=1e-5)


def test_regrid_output_shape_matches_the_named_grid(scorer, small_mesh):
    sc, cfg = scorer
    lats, lons = target_grid("5.625")
    g = sc.to_grid(torch.randn(2, len(small_mesh["v"]), cfg.c_phys))
    assert g.shape == (2, len(lats) * len(lons), cfg.c_phys)


def test_rmse_of_identical_fields_is_zero(scorer, small_mesh):
    sc, cfg = scorer
    x = torch.randn(1, len(small_mesh["v"]), cfg.c_phys)
    assert float(sc.rmse(x, x).max()) < 1e-5


def test_rmse_matches_a_closed_form_constant_offset(scorer, small_mesh):
    """Offset truth by a known constant: the area-weighted RMSE must be exactly that constant,
    whatever the weights are, because a constant error has no spatial structure to weight."""
    sc, cfg = scorer
    x = torch.randn(1, len(small_mesh["v"]), cfg.c_phys)
    got = sc.rmse(x + 2.5, x)
    assert torch.allclose(got, torch.full_like(got, 2.5), atol=1e-4), got


def test_rmse_is_area_weighted_not_a_plain_mean(scorer, small_mesh):
    """Cosine weighting must actually bite: an error concentrated at the pole should score
    lower than the same error at the equator. Without weighting the two would be equal."""
    sc, cfg = scorer
    lats, lons = target_grid("5.625")
    n = len(small_mesh["v"])
    lat = small_mesh["lat"]
    polar = torch.from_numpy((np.abs(lat) > 70).astype(np.float32)).view(1, n, 1)
    equat = torch.from_numpy((np.abs(lat) < 20).astype(np.float32)).view(1, n, 1)
    zero = torch.zeros(1, n, cfg.c_phys)
    r_pole = float(sc.rmse(polar.expand(-1, -1, cfg.c_phys), zero)[0])
    r_eq = float(sc.rmse(equat.expand(-1, -1, cfg.c_phys), zero)[0])
    assert r_pole < r_eq, f"polar error {r_pole} not down-weighted against equatorial {r_eq}"


def test_cell_weights_are_normalized_and_positive():
    w = latlon_cell_area("5.625")
    assert abs(w.mean() - 1.0) < 1e-9 and w.min() > 0


def test_crps_of_a_perfect_ensemble_is_zero(scorer, small_mesh):
    sc, cfg = scorer
    truth = torch.randn(1, len(small_mesh["v"]), cfg.c_phys)
    members = truth.unsqueeze(1).expand(1, 4, -1, -1).contiguous()
    assert float(sc.crps(members, truth).abs().max()) < 1e-5


def test_crps_penalises_a_biased_ensemble(scorer, small_mesh):
    sc, cfg = scorer
    truth = torch.zeros(1, len(small_mesh["v"]), cfg.c_phys)
    good = 0.1 * torch.randn(1, 4, len(small_mesh["v"]), cfg.c_phys)
    biased = good + 5.0
    assert float(sc.crps(good, truth).mean()) < float(sc.crps(biased, truth).mean())


def test_reference_table_carries_the_ifs_caveat():
    """WB2 scores operational models against IFS analysis and ML models against ERA5. This
    project is on the ERA5 side, so the IFS ENS comparison is slightly favourable to us. That
    caveat must travel with the number, not live only in a docstring."""
    out = compare_to_reference({24: 30.0, 72: 70.0}, key="ifs_ens_crps_z500")
    assert "favourable" in out and "GenCast" in out
    assert "22.4" in out and "30.0" in out


def test_scoring_happens_on_the_grid_not_the_mesh(scorer, small_mesh):
    """The whole point of the module: a mesh-space score and a grid-space score differ, because
    barycentric vertex areas and cos(lat) cells weight the globe differently. If these agreed,
    the regrid would be doing nothing."""
    from wnca.losses.terms import area_weights, per_channel_rmse

    sc, cfg = scorer
    n = len(small_mesh["v"])
    lat = torch.from_numpy(small_mesh["lat"].astype(np.float32)).view(1, n, 1)
    err = (lat / 90.0).expand(1, n, cfg.c_phys).contiguous()  # latitude-dependent error
    zero = torch.zeros_like(err)
    on_grid = float(sc.rmse(err, zero)[0])
    on_mesh = float(per_channel_rmse(err, zero, area_weights(small_mesh["area"]))[0])
    assert abs(on_grid - on_mesh) > 1e-3, "grid and mesh scoring agree -- regrid is a no-op"
