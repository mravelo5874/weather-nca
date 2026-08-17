"""Solar forcing: does it actually encode the sun?

Phase 2b showed the model cannot represent the diurnal cycle without this -- 2m temperature
scored -48% skill at +24 h because persistence gets the day/night swing for free at exactly
one diurnal period. Shape tests alone would not catch a forcing that is subtly wrong (an hour
angle with the sign flipped, a declination six months out of phase), and a wrong forcing is
worse than none: it would train the model on a fictitious sun.

So these check the physics against facts that are true of the real Earth.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wnca.data.forcing import (
    N_SOLAR_CHANNELS,
    SolarForcing,
    day_of_year_angle,
    synthetic_times,
    timestep_solar_params,
)


def _times(*iso):
    return np.array([np.datetime64(t, "s") for t in iso])


def test_declination_is_positive_in_northern_summer():
    """June solstice: the sun is over the northern hemisphere."""
    p = timestep_solar_params(_times("2020-06-21T12:00:00"))
    assert p[0, 0] > 0.35, f"sin(declination) = {p[0, 0]}, expected ~+0.4 in June"


def test_declination_is_negative_in_northern_winter():
    p = timestep_solar_params(_times("2020-12-21T12:00:00"))
    assert p[0, 0] < -0.35, f"sin(declination) = {p[0, 0]}, expected ~-0.4 in December"


def test_declination_is_near_zero_at_the_equinoxes():
    for t in ("2020-03-20T12:00:00", "2020-09-22T12:00:00"):
        p = timestep_solar_params(_times(t))
        assert abs(p[0, 0]) < 0.09, f"{t}: sin(dec) = {p[0, 0]}, expected ~0"


def test_declination_magnitude_matches_axial_tilt():
    """Peak declination is the Earth's 23.44 degree obliquity, not something else."""
    days = _times(*[f"2020-{m:02d}-21T00:00:00" for m in range(1, 13)])
    peak = np.abs(np.arcsin(timestep_solar_params(days)[:, 0])).max()
    assert 0.36 < peak < 0.43, f"peak declination {np.degrees(peak):.1f} deg, expected ~23.4"


@pytest.fixture(scope="module")
def eq_mesh():
    """Four points on the equator, 90 degrees apart in longitude."""
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    return {"lat": np.zeros(4), "lon": lon}


def test_noon_is_where_the_sun_is_overhead(eq_mesh):
    """At 12:00 UTC on an equinox, the subsolar point is near 0 deg longitude, so the node at
    lon 0 must be the brightest and the node at lon 180 must be in darkness."""
    sf = SolarForcing(_times("2020-03-20T12:00:00"), eq_mesh)
    cos_z = sf.at(torch.tensor([0]))[0, :, 0].numpy()
    assert cos_z[0] > 0.95, f"lon 0 should be near-overhead at 12Z, got {cos_z[0]}"
    assert cos_z[2] == 0.0, f"lon 180 should be in night at 12Z, got {cos_z[2]}"
    assert cos_z.argmax() == 0


def test_midnight_flips_the_lit_hemisphere(eq_mesh):
    sf = SolarForcing(_times("2020-03-20T00:00:00"), eq_mesh)
    cos_z = sf.at(torch.tensor([0]))[0, :, 0].numpy()
    assert cos_z[2] > 0.95, "lon 180 should be lit at 00Z"
    assert cos_z[0] == 0.0, "lon 0 should be dark at 00Z"


def test_insolation_is_never_negative(eq_mesh):
    """Night contributes no insolation; a negative cos(zenith) would be unphysical input."""
    t = synthetic_times(64, start="2020-01-01")
    sf = SolarForcing(t, eq_mesh)
    cos_z = sf.at(torch.arange(len(t)))[..., 0]
    assert (cos_z >= 0).all() and cos_z.max() > 0.5


def test_polar_night_and_midnight_sun():
    """The strongest test of the geometry: in December the north pole gets nothing and the
    south pole is lit around the clock."""
    mesh = {"lat": np.array([89.0, -89.0]), "lon": np.array([0.0, 0.0])}
    t = synthetic_times(8, start="2020-12-21")  # two full days, 6-hourly
    sf = SolarForcing(t, mesh)
    cos_z = sf.at(torch.arange(len(t)))[..., 0].numpy()
    assert cos_z[:, 0].max() == 0.0, "north pole should be in polar night in December"
    assert cos_z[:, 1].min() > 0.0, "south pole should have midnight sun in December"


def test_diurnal_cycle_is_present_at_midlatitudes():
    """The whole point: a fixed point must see day and night over 24 h."""
    mesh = {"lat": np.array([45.0]), "lon": np.array([0.0])}
    t = synthetic_times(4, start="2020-06-21")  # one day, 6-hourly
    cos_z = SolarForcing(t, mesh).at(torch.arange(4))[:, 0, 0].numpy()
    assert cos_z.max() > 0.5 and cos_z.min() < 0.1, f"no diurnal swing: {cos_z}"


def test_annual_cycle_channels_are_periodic():
    a = day_of_year_angle(_times("2020-01-01T00:00:00"))
    b = day_of_year_angle(_times("2020-12-31T00:00:00"))
    assert np.allclose(a, b, atol=0.03), f"annual cycle not periodic: {a} vs {b}"


def test_window_targets_the_following_timesteps(eq_mesh):
    """Window k must be evaluated at start + 1 + k, not at the seed time."""
    t = synthetic_times(32, start="2020-03-20")
    sf = SolarForcing(t, eq_mesh)
    start = torch.tensor([4])
    win = sf.window(start, 3)
    assert win.shape == (1, 3, 4, N_SOLAR_CHANNELS)
    for k in range(3):
        assert torch.allclose(win[:, k], sf.at(start + 1 + k))


def test_indices_past_the_end_are_clamped_not_wrapped(eq_mesh):
    """A rollout may run past the last timestamp. Wrapping would jump discontinuously back to
    January mid-forecast; clamping degrades to a fixed solar state instead."""
    t = synthetic_times(10, start="2020-06-01")
    sf = SolarForcing(t, eq_mesh)
    assert torch.allclose(sf.at(torch.tensor([50])), sf.at(torch.tensor([9])))


def test_batch_shapes_round_trip(eq_mesh):
    sf = SolarForcing(synthetic_times(40), eq_mesh)
    out = sf.window(torch.arange(5), 6)
    assert out.shape == (5, 6, 4, N_SOLAR_CHANNELS)
    assert torch.isfinite(out).all()


def test_model_refuses_to_run_without_forcing(tiny_cfg, small_mesh):
    """Defaulting to zeros would mean permanent polar night, and the model would train on it
    silently. It must fail loudly instead."""
    from wnca.models.nca import build_model

    model = build_model(tiny_cfg, small_mesh)
    N = len(small_mesh["v"])
    with pytest.raises(ValueError, match="solar_forcing"):
        model.rollout(model.seed(torch.randn(1, N, tiny_cfg.c_phys)),
                      torch.zeros(1, N, tiny_cfg.state.c_static), 1)


def test_forcing_changes_the_prediction(tiny_cfg, small_mesh, forcing_for):
    """If day and night produced identical output, the pathway would be decorative."""
    from wnca.models.nca import build_model

    torch.manual_seed(0)
    model = build_model(tiny_cfg, small_mesh)
    torch.nn.init.normal_(model.update.head.weight, std=0.02)
    N = len(small_mesh["v"])
    cur = torch.randn(1, N, tiny_cfg.c_phys)
    st = torch.zeros(1, N, tiny_cfg.state.c_static)
    seed = model.seed(cur)
    f = forcing_for(tiny_cfg, small_mesh, 1, 1)
    with torch.no_grad():
        lit = model.rollout(seed, st, 1, forcing=f)
        dark = model.rollout(seed, st, 1, forcing=torch.zeros_like(f))
    assert not torch.allclose(lit, dark, atol=1e-7), "solar forcing has no effect on the output"
