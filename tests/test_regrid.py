"""Regrid: round-trip and conservation.

The mesh -> lat-lon operator is what scores against WeatherBench-2's 1.5 degree leaderboard.
If it is not conservative the RMSE is biased, and the bias is invisible -- the numbers stay
plausible, they are just not comparable to the published ones.
"""
import numpy as np
import pytest

from wnca.mesh.regrid import (
    latlon_cell_area, latlon_to_mesh_weights, mesh_to_latlon_matrix, regrid_to_mesh, target_grid,
)


@pytest.fixture(scope="module")
def A(small_mesh):
    return mesh_to_latlon_matrix(small_mesh, grid="5.625", n_quad=3)


def test_rows_sum_to_one(A):
    """Row-stochastic is what makes the operator conservative: it cannot invent or lose mass."""
    rs = np.asarray(A.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0, atol=1e-10), f"row sums range {rs.min()} to {rs.max()}"


def test_constant_field_is_exact(A, small_mesh):
    """A constant must map to itself exactly. Nearest-neighbour passes this too -- it is the
    floor, not the goal."""
    out = A @ np.full(len(small_mesh["v"]), 7.25)
    assert np.abs(out - 7.25).max() < 1e-9


def test_weights_are_nonnegative(A):
    """Negative interpolation weights would let the regrid overshoot into new extrema."""
    assert A.data.min() >= -1e-12


def test_smooth_field_survives_the_regrid(A, small_mesh):
    """A degree-1 harmonic is resolved by both grids, so the regrid should be near-exact."""
    lats, lons = target_grid("5.625")
    f = small_mesh["v"][:, 2]  # z = sin(lat)
    got = (A @ f).reshape(len(lats), len(lons))
    want = np.sin(np.radians(lats))[:, None] * np.ones((1, len(lons)))
    assert np.abs(got - want).max() < 0.05, f"max error {np.abs(got - want).max()}"


def test_global_mean_is_preserved(A, small_mesh):
    """The property that actually matters for RMSE comparability: the area-weighted global
    mean must survive the regrid."""
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(3)
    f = small_mesh["v"] @ coeffs  # a smooth degree-1 field
    mesh_mean = np.average(f, weights=small_mesh["area"])
    grid_mean = np.average(A @ f, weights=latlon_cell_area("5.625"))
    assert abs(mesh_mean - grid_mean) < 0.02, f"mesh {mesh_mean:.5f} vs grid {grid_mean:.5f}"


def test_round_trip_latlon_mesh_latlon(small_mesh):
    """lat-lon -> mesh -> lat-lon on a field both grids resolve."""
    lats, lons = target_grid("5.625")
    grid = np.sin(np.radians(lats))[:, None] * np.ones((1, len(lons)))
    idx, w = latlon_to_mesh_weights(lats, lons, small_mesh["lat"], small_mesh["lon"])
    on_mesh = regrid_to_mesh(grid[None], idx, w)[0]
    back = (mesh_to_latlon_matrix(small_mesh, grid="5.625", n_quad=3) @ on_mesh).reshape(grid.shape)
    assert np.abs(back - grid).max() < 0.1, f"round-trip error {np.abs(back - grid).max()}"


def test_latlon_to_mesh_reproduces_constants(small_mesh):
    lats, lons = target_grid("5.625")
    idx, w = latlon_to_mesh_weights(lats, lons, small_mesh["lat"], small_mesh["lon"])
    out = regrid_to_mesh(np.full((1, len(lats), len(lons)), 3.5), idx, w)[0]
    assert np.abs(out - 3.5).max() < 1e-5


def test_cell_areas_normalized():
    w = latlon_cell_area("5.625")
    assert abs(w.mean() - 1.0) < 1e-9 and w.min() > 0
