"""Cotangent Laplacian, least-squares tangent gradient, and their sparse-matrix form.

Perception is the mesh analogue of the `[identity, Sobel_x, Sobel_y, Laplacian]` filter bank
image NCAs use:

    grad_x f_i = sum_{j in N(i)} gx_ij (f_j - f_i)
    lap     f_i = sum_{j in N(i)} l_ij  (f_j - f_i)

The Laplacian is the cotangent Laplace-Beltrami operator from discrete differential geometry.
A naive flat-space finite difference does *not* converge on a curved surface -- M1 tried it and
`test_operators.py` is the check that caught it.

Every one of these is a linear operator on the field, so each is assembled once as a sparse
[N, N] matrix (see `edge_op_matrix`). That form is what the model uses: the scatter-add
formulation M1 used materializes a [B, E, C] index tensor, which at M2's 60 state channels and
an ensemble folded into the batch is hundreds of MB of int64 per perception call.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_perception_coeffs(
    v: np.ndarray, faces: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-edge (gx, gy, lap) coefficients + per-node barycentric area.

    Edge k carries j=src -> i=dst, so all three coefficients are indexed by destination.
    """
    from .icosphere import tangent_basis

    N, E = len(v), len(edges)
    src, dst = edges[:, 0], edges[:, 1]
    east, north = tangent_basis(v)

    # --- least-squares tangent-plane gradient (rows of the pseudoinverse) ---
    d = v[src] - v[dst]
    d = d - np.sum(d * v[dst], axis=1, keepdims=True) * v[dst]
    dx = np.sum(d * east[dst], axis=1)
    dy = np.sum(d * north[dst], axis=1)
    gx = np.zeros(E)
    gy = np.zeros(E)
    order = np.argsort(dst, kind="stable")
    bounds = np.searchsorted(dst[order], np.arange(N + 1))
    for i in range(N):
        idx = order[bounds[i] : bounds[i + 1]]
        W = np.linalg.pinv(np.stack([dx[idx], dy[idx]], axis=1))  # [2, k]
        gx[idx], gy[idx] = W[0], W[1]

    # --- cotangent Laplace-Beltrami + barycentric vertex areas ---
    cot: dict[tuple[int, int], float] = {}
    area = np.zeros(N)
    for a, b, c in faces:
        p = v[[a, b, c]]
        tri_area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        area[[a, b, c]] += tri_area / 3.0
        for i0, i1, iop in ((a, b, c), (b, c, a), (c, a, b)):
            u, w = v[i0] - v[iop], v[i1] - v[iop]
            ct = np.dot(u, w) / max(np.linalg.norm(np.cross(u, w)), 1e-16)
            cot[(i0, i1)] = cot.get((i0, i1), 0.0) + ct
            cot[(i1, i0)] = cot.get((i1, i0), 0.0) + ct
    lap = np.array([cot.get((int(j), int(i)), 0.0) for j, i in edges]) / (2.0 * area[dst])
    return gx, gy, lap, area


def apply_edge_op(f: np.ndarray, edges: np.ndarray, coeff: np.ndarray, N: int) -> np.ndarray:
    """Reference scatter-add form: out[i] = sum_j coeff_ij (f_j - f_i).

    Kept as the definition the sparse assembly is tested against.
    """
    src, dst = edges[:, 0], edges[:, 1]
    out = np.zeros(N)
    np.add.at(out, dst, coeff * (f[src] - f[dst]))
    return out


def edge_op_matrix(edges: np.ndarray, coeff: np.ndarray, N: int) -> sp.csr_matrix:
    """Assemble `out[i] = sum_j coeff_ij (f_j - f_i)` as a sparse [N, N] matrix.

        A[i, j] = coeff_ij   (j != i)
        A[i, i] = -sum_j coeff_ij

    so that (A f)[i] = sum_j coeff_ij f_j - (sum_j coeff_ij) f_i, which is the identity above.
    """
    src, dst = edges[:, 0], edges[:, 1]
    off = sp.coo_matrix((coeff, (dst, src)), shape=(N, N))
    diag = sp.diags(np.asarray(off.sum(axis=1)).ravel())
    return (off - diag).tocsr()


def laplacian_matrix(mesh: dict[str, np.ndarray]) -> sp.csr_matrix:
    """The cotangent Laplace-Beltrami operator as a sparse matrix.

    Negative semi-definite by construction: on the unit sphere a degree-l spherical harmonic
    has eigenvalue -l(l+1). `mesh/spectral.py` negates it before the Chebyshev recursion.
    """
    return edge_op_matrix(mesh["edges"], mesh["lap"], len(mesh["v"]))


def spectral_radius(A: sp.spmatrix, n_iter: int = 200, tol: float = 1e-7, seed: int = 0) -> float:
    """Largest-magnitude eigenvalue by power iteration.

    Deliberately not a finite-difference estimate: M1's finite-difference spectral radius sat
    on the float32 noise floor and reported pure noise (findings section 5).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[0])
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(n_iter):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny < 1e-30:
            return 0.0
        x = y / ny
        if abs(ny - lam) < tol * max(ny, 1.0):
            return float(ny)
        lam = ny
    return float(lam)
