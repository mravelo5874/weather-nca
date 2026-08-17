"""Analytic convergence check for the mesh operators, from M1 section 1.

Re-run whenever mesh code is touched. This is the test that caught the flat-space Laplacian
failure -- a naive finite difference does not converge on a curved surface, and the only way
that shows up is errors that stop shrinking as the mesh refines.

Degree-1 spherical harmonics f = x, y, z have surface gradient  a_hat - f r_hat  and
Laplace-Beltrami exactly -l(l+1) f = -2f.
"""
import numpy as np
import pytest

from wnca.mesh.icosphere import edges_from_faces, icosphere, tangent_basis
from wnca.mesh.operators import apply_edge_op, build_perception_coeffs, edge_op_matrix


def _errors(n_sub):
    v, faces = icosphere(n_sub)
    edges = edges_from_faces(faces)
    N = len(v)
    gx, gy, lap, _ = build_perception_coeffs(v, faces, edges)
    east, north = tangent_basis(v)
    ge, le = [], []
    for ax in range(3):
        a = np.zeros(3); a[ax] = 1.0
        f = v[:, ax]
        gt = a[None, :] - f[:, None] * v
        tx, ty = np.sum(gt * east, 1), np.sum(gt * north, 1)
        gxn = apply_edge_op(f, edges, gx, N)
        gyn = apply_edge_op(f, edges, gy, N)
        ge.append(np.sqrt(np.sum((gxn - tx) ** 2 + (gyn - ty) ** 2)) / np.sqrt(np.sum(tx**2 + ty**2)))
        le.append(np.linalg.norm(apply_edge_op(f, edges, lap, N) + 2.0 * f) / np.linalg.norm(2.0 * f))
    return float(np.mean(ge)), float(np.mean(le))


def test_operators_converge_under_refinement():
    """Both errors must shrink monotonically. If they grow, the operator is wrong."""
    errs = [_errors(ns) for ns in (2, 3, 4)]
    grads = [e[0] for e in errs]
    laps = [e[1] for e in errs]
    assert grads == sorted(grads, reverse=True), f"gradient error not shrinking: {grads}"
    assert laps == sorted(laps, reverse=True), f"laplacian error not shrinking: {laps}"
    assert grads[-1] < 5e-3 and laps[-1] < 2e-2, f"errors too large at n_sub=4: {grads[-1]}, {laps[-1]}"


def test_convergence_is_first_order_or_better():
    """A wrong operator can still shrink; it just shrinks too slowly. Check the rate."""
    e2, e3, e4 = (_errors(ns)[0] for ns in (2, 3, 4))
    assert e2 / e3 > 1.7 and e3 / e4 > 1.7, f"gradient convergence too slow: {e2}, {e3}, {e4}"


@pytest.mark.parametrize("op", ["gx", "gy", "lap"])
def test_sparse_matches_scatter_add(small_mesh, op):
    """The sparse assembly the model uses must equal the scatter-add definition."""
    N = len(small_mesh["v"])
    A = edge_op_matrix(small_mesh["edges"], small_mesh[op], N)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(N)
    assert np.allclose(A @ f, apply_edge_op(f, small_mesh["edges"], small_mesh[op], N), atol=1e-10)


def test_operators_annihilate_constants(small_mesh):
    """Every operator is a difference operator: a constant field has zero gradient and zero
    Laplacian. A non-zero result here means the diagonal was assembled wrong."""
    N = len(small_mesh["v"])
    ones = np.ones(N)
    for op in ("gx", "gy", "lap"):
        A = edge_op_matrix(small_mesh["edges"], small_mesh[op], N)
        assert np.abs(A @ ones).max() < 1e-10, f"{op} does not annihilate constants"


def test_vertex_areas_sum_to_sphere(small_mesh):
    """Barycentric vertex areas must tile the (polyhedral) sphere."""
    total = small_mesh["area"].sum()
    assert 0.90 * 4 * np.pi < total < 4 * np.pi, f"vertex areas sum to {total}, expected ~{4 * np.pi:.3f}"
