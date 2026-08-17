"""Tensor shape contracts across the mesh / data / model boundaries.

Cheap tests, but they cover the seam where an ensemble axis, a channel axis and a lead-time
axis all meet -- which is exactly where a silent transposition would hide.
"""
import dataclasses

import numpy as np
import pytest
import torch

from wnca.config import Channel, load_config
from wnca.data.dataset import WeatherSeq
from wnca.models.control_gnn import ControlGNN
from wnca.models.nca import build_model
from wnca.models.perception import MeshPerception


def test_channel_order_is_variable_major(tiny_cfg):
    """Channel order is frozen by the ordering rule, never by zarr order. Normalization stats,
    the scorecard and every checkpoint depend on it."""
    chans = tiny_cfg.variables.channels()
    assert chans[0] == Channel("geopotential", 500)
    assert chans[1] == Channel("geopotential", 850)
    assert chans[-1] == Channel("2m_temperature", None)
    assert len(chans) == 2 * 2 + 1


def test_derived_dimensions(tiny_cfg):
    assert tiny_cfg.c_phys == 5
    assert tiny_cfg.c_state == tiny_cfg.c_phys + tiny_cfg.state.c_hidden
    assert tiny_cfg.c_cond == tiny_cfg.state.c_static + tiny_cfg.c_phys  # second_order


def test_perception_output_width(tiny_cfg, small_mesh):
    N = len(small_mesh["v"])
    p = MeshPerception(small_mesh)
    out = p(torch.randn(3, N, tiny_cfg.c_state))
    assert out.shape == (3, N, 4 * tiny_cfg.c_state)


def test_perception_identity_block_is_first(tiny_cfg, small_mesh):
    """The layout is [identity | grad_x | grad_y | laplacian]; downstream code slices on it."""
    N = len(small_mesh["v"])
    x = torch.randn(2, N, tiny_cfg.c_state)
    assert torch.allclose(MeshPerception(small_mesh)(x)[..., : tiny_cfg.c_state], x)


@pytest.mark.parametrize("kind", ["nca", "control_gnn"])
def test_rollout_shapes(tiny_cfg, small_mesh, kind):
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, kind=kind))
    model = build_model(cfg, small_mesh)
    N = len(small_mesh["v"])
    cur = torch.randn(2, N, cfg.c_phys)
    st = torch.randn(2, N, cfg.state.c_static)
    with torch.no_grad():
        out = model.rollout(model.seed(cur), st, 3, prev_phys=torch.randn(2, N, cfg.c_phys))
    assert out.shape == (2, 3, N, cfg.c_phys)


@pytest.mark.parametrize("kind", ["nca", "control_gnn"])
def test_ensemble_rollout_shapes(tiny_cfg, small_mesh, kind):
    cfg = dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(tiny_cfg.model, kind=kind, stochastic=True)
    )
    model = build_model(cfg, small_mesh)
    N = len(small_mesh["v"])
    cur = torch.randn(2, N, cfg.c_phys)
    with torch.no_grad():
        out = model.rollout_ensemble(model.seed(cur), torch.randn(2, N, cfg.state.c_static),
                                     3, n_members=5)
    assert out.shape == (2, 5, 3, N, cfg.c_phys)


def test_ensemble_member_axis_unflattens_correctly(tiny_cfg, small_mesh):
    """Members are folded into the batch as [B*M]; `.view(B, M, ...)` must invert that. If the
    two disagreed, member 0 of batch 1 would silently become member 1 of batch 0."""
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, stochastic=True))
    model = build_model(cfg, small_mesh)
    N = len(small_mesh["v"])
    cur = torch.stack([torch.full((N, cfg.c_phys), float(b)) for b in range(3)])
    with torch.no_grad():
        out = model.rollout_ensemble(model.seed(cur), torch.zeros(3, N, cfg.state.c_static),
                                     1, n_members=4)
    for b in range(3):
        assert torch.allclose(out[b].mean(), torch.tensor(float(b)), atol=1e-3), \
            f"batch element {b} leaked across the member axis"


def test_seed_zeroes_hidden_channels(tiny_cfg, small_mesh):
    model = build_model(tiny_cfg, small_mesh)
    N = len(small_mesh["v"])
    s = model.seed(torch.randn(2, N, tiny_cfg.c_phys))
    assert s.shape == (2, N, tiny_cfg.c_state)
    assert s[..., tiny_cfg.c_phys :].abs().max() == 0


def test_dataset_triple_shapes(small_mesh):
    N, C = len(small_mesh["v"]), 3
    ds = WeatherSeq(np.random.randn(20, N, C).astype(np.float32), n_out=4)
    prev, cur, tgt = ds[0]
    assert prev.shape == (N, C) and cur.shape == (N, C) and tgt.shape == (4, N, C)
    assert len(ds) == 20 - 4 - 1


def test_dataset_targets_follow_current():
    """(x_{t-1}, x_t) -> x_{t+1..}: an off-by-one here trains the model on the wrong lead."""
    x = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    prev, cur, tgt = WeatherSeq(x, n_out=2)[3]
    assert prev.item() == 3 and cur.item() == 4 and tgt.flatten().tolist() == [5, 6]


def test_dataset_rejects_too_short_a_split():
    with pytest.raises(ValueError, match="at least"):
        WeatherSeq(np.zeros((3, 5, 1), dtype=np.float32), n_out=8)


def test_config_rejects_overlapping_splits():
    with pytest.raises(ValueError, match="overlap"):
        load_config(None, overrides={"data": {"train_years": [2015], "val_years": [2015]}})


def test_config_rejects_rollout_curriculum():
    """M1 ran it twice; it contributed nothing and destroyed the long-lead metric both times."""
    with pytest.raises(ValueError, match="rollout_epochs"):
        load_config(None, overrides={"train": {"rollout_epochs": 2}})


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(None, overrides={"train": {"lr_": 1e-3}})


def test_phase_configs_all_load_and_validate():
    """Every shipped config must parse and pass schema validation."""
    from pathlib import Path
    from wnca.config import CONFIG_DIR
    for p in sorted(Path(CONFIG_DIR).glob("*.yaml")):
        cfg = load_config(p)
        assert cfg.c_phys > 0, p.name
