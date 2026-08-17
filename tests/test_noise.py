"""zero-noise == deterministic; nonzero noise => spread.

These are the two halves of "is the noise real?". Together with the zero-noise ablation logged
every validation pass in phase 3a, they are what stands between a working ensemble and a
decorative one that scores well and means nothing.
"""
import pytest
import torch

from wnca.data.forcing import SolarForcing, synthetic_times
from wnca.models.nca import WeatherNCA
from wnca.models.perception import MeshPerception


def _forcing(cfg, mesh, B, W, start=3):
    """Solar forcing of the right shape; the model refuses to run without it."""
    if not cfg.state.solar_forcing:
        return None
    return SolarForcing(synthetic_times(start + W + 8), mesh).window(
        torch.arange(start, start + B), W)


def _model(cfg, mesh, stochastic=True):
    import dataclasses
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, stochastic=stochastic))
    torch.manual_seed(0)
    m = WeatherNCA(cfg, MeshPerception(mesh))
    # Untrained models are the identity map, which would make every test below trivially pass.
    torch.nn.init.normal_(m.update.head.weight, std=0.01)
    torch.nn.init.normal_(m.update.film.weight, std=0.05)
    return cfg, m


def _inputs(cfg, mesh, B=2):
    N = len(mesh["v"])
    torch.manual_seed(1)
    cur = torch.randn(B, N, cfg.c_phys)
    return cur, torch.randn(B, N, cfg.c_phys), torch.randn(B, N, cfg.state.c_static)


def test_film_is_identity_at_init(tiny_cfg, small_mesh):
    """Zero-init FiLM means a freshly stochastic model is numerically identical to the
    deterministic checkpoint it warm-starts from. That is what makes 2c -> 3a safe."""
    import dataclasses
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, stochastic=True))
    torch.manual_seed(0)
    m = WeatherNCA(cfg, MeshPerception(small_mesh))
    torch.nn.init.normal_(m.update.head.weight, std=0.01)
    cur, prev, st = _inputs(cfg, small_mesh)
    with torch.no_grad():
        a = m.forecast_step(m.seed(cur), st, prev, torch.randn(2, cfg.model.noise_dim), _forcing(cfg, small_mesh, 2, 1)[:, 0] if cfg.state.solar_forcing else None)
        b = m.forecast_step(m.seed(cur), st, prev, torch.zeros(2, cfg.model.noise_dim), _forcing(cfg, small_mesh, 2, 1)[:, 0] if cfg.state.solar_forcing else None)
    assert torch.allclose(a, b, atol=1e-6), "FiLM is not identity at init"


def test_zero_noise_equals_deterministic_path(tiny_cfg, small_mesh):
    """z=0 must reproduce the deterministic forward exactly, at any weights."""
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=True)
    cur, prev, st = _inputs(cfg, small_mesh)
    with torch.no_grad():
        stoch_zero = m.forecast_step(m.seed(cur), st, prev, torch.zeros(2, cfg.model.noise_dim), _forcing(cfg, small_mesh, 2, 1)[:, 0] if cfg.state.solar_forcing else None)
        m.update.stochastic = False
        deterministic = m.forecast_step(m.seed(cur), st, prev, None, _forcing(cfg, small_mesh, 2, 1)[:, 0] if cfg.state.solar_forcing else None)
    assert torch.allclose(stoch_zero, deterministic, atol=1e-6)


def test_nonzero_noise_produces_spread(tiny_cfg, small_mesh):
    """The load-bearing test. If this fails the noise is decorative and phase 3a is dead."""
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=True)
    cur, prev, st = _inputs(cfg, small_mesh)
    with torch.no_grad():
        pred = m.rollout_ensemble(m.seed(cur), st, 2, prev_phys=prev, n_members=8, forcing=_forcing(cfg, small_mesh, cur.shape[0], 2))
    spread = pred.std(dim=1).mean().item()
    assert spread > 1e-6, f"ensemble collapsed at init: spread {spread}"


def test_members_are_distinct(tiny_cfg, small_mesh):
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=True)
    cur, prev, st = _inputs(cfg, small_mesh, B=1)
    with torch.no_grad():
        pred = m.rollout_ensemble(m.seed(cur), st, 1, prev_phys=prev, n_members=4, forcing=_forcing(cfg, small_mesh, cur.shape[0], 1))[0, :, 0]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(pred[i], pred[j], atol=1e-7), f"members {i},{j} identical"


def test_same_z_gives_same_member(tiny_cfg, small_mesh):
    """Member identity must be a deterministic function of z -- otherwise spread is just RNG."""
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=True)
    cur, prev, st = _inputs(cfg, small_mesh, B=1)
    z = torch.randn(1, 2, cfg.model.noise_dim)
    with torch.no_grad():
        a = m.rollout_ensemble(m.seed(cur), st, 2, prev_phys=prev, n_members=2, z=z, forcing=_forcing(cfg, small_mesh, cur.shape[0], 2))
        b = m.rollout_ensemble(m.seed(cur), st, 2, prev_phys=prev, n_members=2, z=z, forcing=_forcing(cfg, small_mesh, cur.shape[0], 2))
    assert torch.allclose(a, b, atol=1e-6)


def test_reseed_hidden_does_not_touch_noise(tiny_cfg, small_mesh):
    """`reseed_hidden` must zero hidden channels and nothing else. If it re-drew z, member
    identity would dissolve and spread would stop meaning anything at long leads."""
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=True)
    N = len(small_mesh["v"])
    state = torch.randn(2, N, cfg.c_state)
    out = m.reseed_hidden(state)
    assert torch.allclose(out[..., : cfg.c_phys], state[..., : cfg.c_phys])
    assert out[..., cfg.c_phys :].abs().max() == 0


def test_noise_is_constant_across_cells(tiny_cfg, small_mesh):
    """One z per member, broadcast over ALL nodes. A per-cell perturbation is exactly what the
    smoothing operator damps for free -- the whole design rests on this being global."""
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=True)
    N = len(small_mesh["v"])
    torch.manual_seed(3)
    perceived = torch.randn(1, N, 4 * cfg.c_state)
    cond = torch.randn(1, N, cfg.c_cond)
    z = torch.randn(1, cfg.model.noise_dim)
    with torch.no_grad():
        delta = m.update(perceived, cond, z) - m.update(perceived, cond, torch.zeros_like(z))
    # A uniform input under a global modulation must give a uniform output difference.
    with torch.no_grad():
        flat = torch.ones(1, N, 4 * cfg.c_state)
        fcond = torch.ones(1, N, cfg.c_cond)
        d2 = m.update(flat, fcond, z) - m.update(flat, fcond, torch.zeros_like(z))
    assert d2.std(dim=1).max().item() < 1e-6, "noise effect varies across cells"
    assert delta.abs().max() > 0, "noise had no effect at all"


def test_deterministic_model_ignores_z(tiny_cfg, small_mesh):
    cfg, m = _model(tiny_cfg, small_mesh, stochastic=False)
    cur, prev, st = _inputs(cfg, small_mesh)
    with torch.no_grad():
        a = m.forecast_step(m.seed(cur), st, prev, torch.randn(2, cfg.model.noise_dim), _forcing(cfg, small_mesh, 2, 1)[:, 0] if cfg.state.solar_forcing else None)
        b = m.forecast_step(m.seed(cur), st, prev, None, _forcing(cfg, small_mesh, 2, 1)[:, 0] if cfg.state.solar_forcing else None)
    assert torch.allclose(a, b, atol=1e-7)


def test_film_gradients_are_nonzero(tiny_cfg, small_mesh):
    """Zero-init weights still receive gradient (d/dW = dL/dmod * z^T), so the pathway can
    actually learn. A zero-init that killed the gradient would silently never train."""
    import dataclasses
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, stochastic=True))
    torch.manual_seed(0)
    m = WeatherNCA(cfg, MeshPerception(small_mesh))
    torch.nn.init.normal_(m.update.head.weight, std=0.01)
    cur, prev, st = _inputs(cfg, small_mesh)
    pred = m.rollout_ensemble(m.seed(cur), st, 1, prev_phys=prev, n_members=3, forcing=_forcing(cfg, small_mesh, cur.shape[0], 1))
    pred.pow(2).mean().backward()
    assert m.update.film.weight.grad.abs().max() > 0, "FiLM pathway receives no gradient"
