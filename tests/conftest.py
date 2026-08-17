"""Shared fixtures. Everything here runs on a tiny mesh so `make test` stays under 60s."""
import numpy as np
import pytest

from wnca.config import load_config
from wnca.mesh.icosphere import edges_from_faces, icosphere
from wnca.mesh.operators import build_perception_coeffs


@pytest.fixture(scope="session")
def small_mesh():
    """n_sub=3 (642 nodes): big enough for the operators to be meaningful, small enough to be fast."""
    v, faces = icosphere(3)
    edges = edges_from_faces(faces)
    gx, gy, lap, area = build_perception_coeffs(v, faces, edges)
    return dict(
        v=v, faces=faces, edges=edges, gx=gx, gy=gy, lap=lap, area=area,
        lat=np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))),
        lon=np.degrees(np.arctan2(v[:, 1], v[:, 0])),
    )


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """Synthetic-data config on the small mesh: no network, no ERA5, deterministic."""
    return load_config(
        None,
        overrides={
            "phase": "test",
            "mesh": {"n_sub": 3},
            "variables": {
                "atmospheric": ["geopotential", "temperature"],
                "levels": [500, 850],
                "surface": ["2m_temperature"],
                "log_transform": [],
            },
            "state": {"c_hidden": 4},
            "model": {"hidden_dim": 32, "n_layers": 2, "n_substeps": 3, "noise_dim": 8},
            "data": {
                "source": "synthetic",
                "cache_dir": str(tmp_path_factory.mktemp("cache")),
                "max_steps_per_split": 40,
                "train_years": [2015], "val_years": [2016], "test_years": [2017],
            },
            "train": {"epochs": 1, "batch_size": 2, "ckpt_windows": 2, "warmup_steps": 2},
            "ensemble": {"m_train": 3, "m_val": 3, "m_test": 3},
        },
    )


@pytest.fixture(scope="session")
def tiny_cache(tiny_cfg, small_mesh):
    from wnca.data.cache import build_cache
    return build_cache(tiny_cfg, small_mesh, verbose=False)


@pytest.fixture
def forcing_for():
    """Build solar forcing of the right shape for a direct model call.

    Tests that exercise shapes, noise or checkpointing do not care what the sun is doing, but
    the model refuses to run without forcing when `solar_forcing` is on -- deliberately, so a
    broken pipeline cannot silently train on permanent night.
    """
    import numpy as np
    import torch

    from wnca.data.forcing import SolarForcing, synthetic_times

    def make(cfg, mesh, B, n_windows, start=3):
        if not cfg.state.solar_forcing:
            return None
        sf = SolarForcing(synthetic_times(start + n_windows + 8), mesh)
        return sf.window(torch.arange(start, start + B), n_windows)

    return make
