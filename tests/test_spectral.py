"""Band energies of a known field.

On the unit sphere a degree-l spherical harmonic has Laplacian eigenvalue -l(l+1), so a band
covering [l_lo, l_hi] must capture a Y_l with l in that range and reject one well outside it.
That makes this a real check on the Chebyshev construction rather than a self-consistency test.
"""
import numpy as np
import pytest
import torch

from wnca.mesh.operators import laplacian_matrix, spectral_radius
from wnca.mesh.spectral import BandFilters, band_edges, chebyshev_coeffs


@pytest.fixture(scope="module")
def bands(small_mesh):
    return BandFilters(laplacian_matrix(small_mesh), small_mesh["area"], n_bands=4, order=32)


def _sph_harm(mesh, l, m):
    from scipy.special import sph_harm_y
    theta = np.arccos(np.clip(mesh["v"][:, 2], -1, 1))
    phi = np.arctan2(mesh["v"][:, 1], mesh["v"][:, 0])
    return torch.from_numpy(np.real(sph_harm_y(l, m, theta, phi)).astype(np.float32)).view(1, -1, 1)


def test_lambda_max_is_positive(bands):
    assert bands.lambda_max > 0


def test_low_harmonic_lands_in_the_lowest_band(bands, small_mesh):
    e = bands.energies(_sph_harm(small_mesh, 2, 1))[0, :, 0].numpy()
    assert e.argmax() == 0, f"l=2 peaked in band {e.argmax()}, fractions {e / e.sum()}"


def test_band_index_increases_with_wavenumber(bands, small_mesh):
    """The ordering property is the one that matters: higher l must not land in a lower band."""
    peaks = [bands.energies(_sph_harm(small_mesh, l, min(l, 2)))[0, :, 0].argmax().item()
             for l in (1, 4, 10, 18)]
    assert peaks == sorted(peaks), f"band peaks not monotone in l: {peaks}"
    assert peaks[-1] > peaks[0], f"all harmonics landed in the same band: {peaks}"


def test_bands_partition_energy(bands, small_mesh):
    """Raised-cosine windows sum to ~1, so band energies must roughly total the field's energy
    rather than double-counting or dropping a range."""
    x = _sph_harm(small_mesh, 6, 2)
    total = float((bands.area_w * x.pow(2)).mean())
    assert 0.7 * total < float(bands.energies(x).sum()) < 1.4 * total


def test_constant_field_has_energy_only_in_band_zero(bands, small_mesh):
    x = torch.ones(1, len(small_mesh["v"]), 1)
    e = bands.energies(x)[0, :, 0].numpy()
    assert e[0] > 0.9 * e.sum(), f"constant leaked out of band 0: {e / e.sum()}"


def test_log_energies_finite_on_zero_field(bands, small_mesh):
    x = torch.zeros(1, len(small_mesh["v"]), 1)
    assert torch.isfinite(bands.log_energies(x)).all()


def test_chebyshev_coeffs_shape():
    c = chebyshev_coeffs(100.0, n_bands=4, order=16)
    assert c.shape == (4, 17) and np.isfinite(c).all()


def test_band_edges_increase():
    e = band_edges(500.0, 5)
    assert len(e) == 6 and np.all(np.diff(e) > 0) and e[0] == 0


def test_spectral_radius_matches_dense(small_mesh):
    L = laplacian_matrix(small_mesh)
    dense = np.abs(np.linalg.eigvals((-L).toarray())).max()
    assert abs(spectral_radius((-L).tocsr()) - dense) / dense < 1e-3


def test_gradients_flow_through_filters(bands, small_mesh):
    x = torch.randn(1, len(small_mesh["v"]), 1, requires_grad=True)
    bands.log_energies(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
