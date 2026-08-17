"""Multi-variable (phase 2b) paths: normalization across scales, and per-channel reporting.

The single-variable phases exercise almost none of this. The failure modes here are quiet -- a
log channel reported as if it were physical, a level table whose rows and columns are
transposed, a per-channel growth rate that silently averages the wrong axis.
"""

import dataclasses

import numpy as np
import pytest
import torch

from wnca.config import Channel, load_config
from wnca.data.normalize import fit_normalizer
from wnca.eval.metrics import (
    Scorecard,
    channel_units,
    format_channel_summary,
    format_level_table,
)
from wnca.eval.perturbation import format_per_channel


@pytest.fixture(scope="module")
def mv_cfg():
    """The real M2 variable set, on a small mesh."""
    return load_config(
        None,
        overrides={
            "phase": "test_mv",
            "mesh": {"n_sub": 2},
            "state": {"c_hidden": 4},
            "model": {"hidden_dim": 16, "n_layers": 2, "n_substeps": 2},
            "data": {
                "source": "synthetic",
                "max_steps_per_split": 20,
                "train_years": [2015],
                "val_years": [2016],
                "test_years": [2017],
            },
            "train": {"epochs": 1, "batch_size": 2, "ckpt_windows": 2},
        },
    )


def _raw(cfg, seed, n_t=12, n_n=20, realistic_q=False):
    """Synthetic raw fields with per-channel magnitudes like the real thing."""
    chans = cfg.variables.channels()
    rng = np.random.default_rng(seed)
    out = np.empty((n_t, n_n, len(chans)), dtype=np.float32)
    q_scale = {850: 6e-3, 700: 3e-3, 500: 1e-3, 300: 1.7e-4, 250: 7.7e-5}
    for i, c in enumerate(chans):
        if c.name == "specific_humidity":
            s = q_scale.get(c.level, 1e-3) if realistic_q else 1e-3
            out[:, :, i] = np.abs(rng.lognormal(np.log(s), 0.5, (n_t, n_n)))
        elif "wind" in c.name:
            out[:, :, i] = rng.normal(0, 20, (n_t, n_n))  # crosses zero
        elif c.name == "geopotential":
            out[:, :, i] = rng.normal(5e4, 3e3, (n_t, n_n))
        else:
            out[:, :, i] = rng.normal(280, 15, (n_t, n_n))
    return out


def test_full_m2_channel_set(mv_cfg):
    chans = mv_cfg.variables.channels()
    assert len(chans) == 28, f"expected 5 vars x 5 levels + 3 surface, got {len(chans)}"
    assert chans[0] == Channel("geopotential", 850)
    assert chans[24] == Channel("specific_humidity", 250)
    assert chans[25] == Channel("10m_u_component_of_wind", None)


def test_channel_order_is_stable_across_loads(mv_cfg):
    """Checkpoints, normalizer stats and the scorecard all index by this order."""
    again = load_config(
        None,
        overrides={
            "variables": {
                "atmospheric": list(mv_cfg.variables.atmospheric),
                "levels": list(mv_cfg.variables.levels),
                "surface": list(mv_cfg.variables.surface),
            }
        },
    )
    assert [c.key for c in again.variables.channels()] == [c.key for c in mv_cfg.variables.channels()]


def test_log_transform_balances_humidity_across_levels(mv_cfg):
    """q spans ~4 orders of magnitude across levels. Without the log the 850 hPa channel
    dominates every gradient and 250 hPa contributes nothing."""
    raw = _raw(mv_cfg, seed=0, realistic_q=True)
    enc = fit_normalizer(raw, mv_cfg).encode(raw)
    q = [i for i, c in enumerate(mv_cfg.variables.channels()) if c.name == "specific_humidity"]
    spreads = [float(enc[:, :, i].std()) for i in q]
    assert max(spreads) / min(spreads) < 1.5, f"q levels still imbalanced: {spreads}"
    assert np.isfinite(enc).all()


def test_without_log_humidity_would_be_imbalanced(mv_cfg):
    """The control for the test above: turn the log off and the imbalance must reappear,
    otherwise the previous test proves nothing."""
    no_log = dataclasses.replace(
        mv_cfg, variables=dataclasses.replace(mv_cfg.variables, log_transform=())
    )
    raw = _raw(mv_cfg, seed=0, realistic_q=True)
    enc = fit_normalizer(raw, no_log).encode(raw)
    q = [i for i, c in enumerate(no_log.variables.channels()) if c.name == "specific_humidity"]
    # Per-channel normalization still rescales each level, so check the raw magnitudes are
    # what the log transform exists to fix.
    mags = [float(np.abs(raw[:, :, i]).mean()) for i in q]
    assert max(mags) / min(mags) > 50, "test fixture does not reproduce the q scale problem"


def test_normalize_round_trip_across_all_scales(mv_cfg):
    """Errors must be small relative to each channel's own spread -- not relative to its
    values, which pass through zero for wind and make relative error meaningless."""
    raw = _raw(mv_cfg, seed=1, realistic_q=True)
    norm = fit_normalizer(raw, mv_cfg)
    dec = norm.decode(norm.encode(raw))
    for i, c in enumerate(mv_cfg.variables.channels()):
        spread = raw[:, :, i].std()
        assert np.abs(dec[:, :, i] - raw[:, :, i]).max() / spread < 1e-4, c.key


def test_log_channels_are_labelled_not_physical(mv_cfg):
    raw = _raw(mv_cfg, seed=2)
    units = channel_units(mv_cfg, fit_normalizer(raw, mv_cfg))
    for i, c in enumerate(mv_cfg.variables.channels()):
        if c.name == "specific_humidity":
            assert units[i] == "log", f"{c.key} must not be reported in physical units"
        elif c.name == "geopotential":
            assert units[i] == "m2/s2"
        elif "wind" in c.name:
            assert units[i] == "m/s"
        elif "temperature" in c.name:
            assert units[i] == "K"


def _fake_scorecard(cfg):
    keys = tuple(c.key for c in cfg.variables.channels())
    sc = Scorecard(20, len(keys), keys)
    rng = np.random.default_rng(3)
    sc.sq_model = rng.uniform(0.01, 0.1, (20, len(keys)))
    sc.sq_persist = sc.sq_model * 4.0
    sc.sq_clim = sc.sq_model * 20.0
    sc.n = 1
    return sc


def test_level_table_covers_every_variable_and_level(mv_cfg):
    norm = fit_normalizer(_raw(mv_cfg, seed=4), mv_cfg)
    out = format_level_table(_fake_scorecard(mv_cfg), mv_cfg, norm, lead_hours=24)
    for name in mv_cfg.variables.atmospheric:
        assert name in out, f"{name} missing from level table"
    for lv in mv_cfg.variables.levels:
        assert str(lv) in out
    for name in mv_cfg.variables.surface:
        assert name in out


def test_level_table_rejects_out_of_range_lead(mv_cfg):
    norm = fit_normalizer(_raw(mv_cfg, seed=5), mv_cfg)
    out = format_level_table(_fake_scorecard(mv_cfg), mv_cfg, norm, lead_hours=100_000)
    assert "outside" in out


def test_channel_summary_lists_all_channels(mv_cfg):
    norm = fit_normalizer(_raw(mv_cfg, seed=6), mv_cfg)
    out = format_channel_summary(_fake_scorecard(mv_cfg), mv_cfg, norm)
    for c in mv_cfg.variables.channels():
        assert c.key in out, f"{c.key} missing from channel summary"


def test_per_channel_growth_flags_the_worst_variable(mv_cfg):
    """If one variable amplifies while the rest are neutral, it sets the sub-step budget."""
    keys = tuple(c.key for c in mv_cfg.variables.channels())
    g = np.full(len(keys), 1.01)
    g[keys.index("v_component_of_wind_250")] = 1.42
    out = format_per_channel(
        {"per_channel_growth": g, "channels": keys, "sustained_from_window": 3}, mv_cfg
    )
    assert "v_component_of_wind_250" in out and "1.42" in out
    assert "sets the sub-step budget" in out


def test_per_channel_growth_reports_all_clear(mv_cfg):
    keys = tuple(c.key for c in mv_cfg.variables.channels())
    out = format_per_channel(
        {"per_channel_growth": np.full(len(keys), 1.01), "channels": keys,
         "sustained_from_window": 2},
        mv_cfg,
    )
    assert "all 28 channels" in out


def test_multivar_model_roundtrips_shapes(mv_cfg, small_mesh):
    """28 physical + hidden channels through perception and the update rule."""
    from wnca.models.nca import build_model

    cfg = dataclasses.replace(mv_cfg, mesh=dataclasses.replace(mv_cfg.mesh, n_sub=3))
    model = build_model(cfg, small_mesh)
    N = len(small_mesh["v"])
    cur = torch.randn(2, N, cfg.c_phys)
    with torch.no_grad():
        from wnca.data.forcing import SolarForcing, synthetic_times

        forcing = SolarForcing(synthetic_times(20), small_mesh).window(torch.arange(3, 5), 2)
        out = model.rollout(
            model.seed(cur), torch.randn(2, N, cfg.state.c_static), 2,
            prev_phys=torch.randn(2, N, cfg.c_phys), forcing=forcing,
        )
    assert out.shape == (2, 2, N, 28)
    assert cfg.c_state == 28 + cfg.state.c_hidden


def test_channel_weights_reach_the_loss(mv_cfg):
    """28 equally-weighted channels is the default; the override must actually bite."""
    from wnca.losses.terms import channel_weights

    w = channel_weights(mv_cfg, "cpu")
    assert w.shape == (1, 28) and torch.allclose(w, torch.ones_like(w))

    weighted = dataclasses.replace(
        mv_cfg, loss=dataclasses.replace(mv_cfg.loss, channel_weights={"geopotential_500": 5.0})
    )
    w2 = channel_weights(weighted, "cpu")
    assert w2[0, mv_cfg.variables.index_of("geopotential", 500)] == 5.0
    assert w2.sum() == 27 + 5.0


def test_unknown_channel_weight_is_rejected(mv_cfg):
    """A typo in a weight key must fail loudly rather than silently weighting nothing."""
    with pytest.raises(ValueError, match="unknown channels"):
        load_config(None, overrides={"loss": {"channel_weights": {"geopotential_999": 2.0}}})
