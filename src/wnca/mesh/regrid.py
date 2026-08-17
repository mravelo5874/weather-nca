"""Mesh <-> lat-lon sparse regrid matrices, cached.

Two directions, and they are not symmetric:

* **lat-lon -> mesh** (`latlon_to_mesh_weights`): bilinear interpolation. Used to ingest ERA5.
* **mesh -> lat-lon** (`mesh_to_latlon_matrix`): area-weighted average of the piecewise-linear
  (P1) mesh interpolant over each target cell. Used to score against WeatherBench-2, whose
  probabilistic numbers are published at 1.5 degrees.

The scoring direction must be conservative, not nearest-neighbour, or the RMSE is biased.
Both operators are row-stochastic, so both reproduce a constant field exactly -- that is the
property `tests/test_regrid.py` checks, along with a round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

# Named lat-lon targets. "1.5" is the WeatherBench-2 probabilistic leaderboard grid.
TARGET_GRIDS = {
    "1.5": (240, 121),
    "2.8125": (128, 64),
    "5.625": (64, 32),
}


def target_grid(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(lats, lons) in degrees for a named equiangular-with-poles grid."""
    if name not in TARGET_GRIDS:
        raise ValueError(f"unknown target grid {name!r}; have {sorted(TARGET_GRIDS)}")
    n_lon, n_lat = TARGET_GRIDS[name]
    lats = np.linspace(-90.0, 90.0, n_lat)
    lons = np.arange(n_lon) * (360.0 / n_lon)
    return lats, lons


# --------------------------------------------------------------------------------------
# lat-lon -> mesh (ingest)
# --------------------------------------------------------------------------------------


def latlon_to_mesh_weights(
    lats: np.ndarray, lons: np.ndarray, mesh_lat: np.ndarray, mesh_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear weights from a regular lat-lon grid onto mesh nodes.

    Returns (idx[N, 4], w[N, 4]) indexing the FLATTENED [n_lat * n_lon] grid, row-major as
    lat * n_lon + lon. `data/era5.py` transposes every field to (time, latitude, longitude)
    before calling this so the contract is unconditional rather than store-dependent.
    """
    n_lat, n_lon = len(lats), len(lons)
    asc = lats[0] < lats[-1]
    la = lats if asc else lats[::-1]
    ml = np.clip(mesh_lat, la[0], la[-1])
    i1 = np.clip(np.searchsorted(la, ml), 1, n_lat - 1)
    i0 = i1 - 1
    ty = (ml - la[i0]) / np.maximum(la[i1] - la[i0], 1e-12)
    if not asc:
        i0, i1 = n_lat - 1 - i0, n_lat - 1 - i1

    lo = np.mod(mesh_lon - lons[0], 360.0)
    dlon = (lons[-1] - lons[0]) / (n_lon - 1)
    j0 = np.floor(lo / dlon).astype(int) % n_lon
    j1 = (j0 + 1) % n_lon
    tx = lo / dlon - np.floor(lo / dlon)

    idx = np.stack([i0 * n_lon + j0, i0 * n_lon + j1, i1 * n_lon + j0, i1 * n_lon + j1], axis=1)
    w = np.stack([(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx], axis=1)
    return idx.astype(np.int64), w


def regrid_to_mesh(field_stack: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """[T, n_lat, n_lon] -> [T, N_nodes]."""
    flat = field_stack.reshape(len(field_stack), -1)
    return np.einsum("tnk,nk->tn", flat[:, idx], w).astype(np.float32)


# --------------------------------------------------------------------------------------
# mesh -> lat-lon (scoring)
# --------------------------------------------------------------------------------------


def _barycentric(v: np.ndarray, faces: np.ndarray, pts: np.ndarray, cand: np.ndarray):
    """Locate each point in a candidate face and return (face_id, weights).

    A point p on the unit sphere lies in the spherical triangle (a, b, c) iff the solution of
    [a b c] w = p has all components non-negative. Solving that 3x3 system gives the P1
    barycentric weights directly.
    """
    Q, K = cand.shape
    best_face = np.full(Q, -1, dtype=np.int64)
    best_w = np.zeros((Q, 3))
    todo = np.arange(Q)

    for k in range(K):
        if todo.size == 0:
            break
        f = faces[cand[todo, k]]  # [q, 3]
        M = v[f].transpose(0, 2, 1)  # [q, 3, 3]: columns are the triangle vertices
        try:
            w = np.linalg.solve(M, pts[todo][..., None])[..., 0]  # [q, 3]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate face
            continue
        s = w.sum(axis=1, keepdims=True)
        w = w / np.where(np.abs(s) < 1e-30, 1.0, s)
        hit = (w >= -1e-9).all(axis=1)
        rows = todo[hit]
        best_face[rows] = cand[rows, k]
        best_w[rows] = np.clip(w[hit], 0.0, None)
        todo = todo[~hit]

    # Points that fell through every candidate (numerically on an edge): use the nearest face
    # and clamp. On a uniform icosphere with k=8 candidates this is empty in practice.
    if todo.size:
        f = faces[cand[todo, 0]]
        M = v[f].transpose(0, 2, 1)
        w = np.clip(np.linalg.solve(M, pts[todo][..., None])[..., 0], 0.0, None)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
        best_face[todo] = cand[todo, 0]
        best_w[todo] = w
    return best_face, best_w


def mesh_to_latlon_matrix(
    mesh: dict[str, np.ndarray],
    grid: str = "1.5",
    n_quad: int = 3,
    n_cand: int = 8,
) -> sp.csr_matrix:
    """Area-weighted (conservative) mesh -> lat-lon operator, shape [n_lat * n_lon, N_nodes].

    Each target cell is sampled on an `n_quad x n_quad` grid of its interior, each sample is
    evaluated by P1 interpolation on the containing mesh triangle, and the samples are averaged
    with cos(lat) area weights. Rows sum to 1, so a constant field maps to itself exactly and
    the operator does not bias the mean.
    """
    v, faces = mesh["v"], mesh["faces"]
    lats, lons = target_grid(grid)
    n_lat, n_lon = len(lats), len(lons)
    dlat = 180.0 / (n_lat - 1)
    dlon = 360.0 / n_lon

    # Quadrature offsets: midpoints of n_quad equal sub-intervals of the cell.
    off = (np.arange(n_quad) + 0.5) / n_quad - 0.5
    qlat = np.clip(lats[:, None] + off[None, :] * dlat, -90.0, 90.0)  # [n_lat, n_quad]
    qlon = lons[:, None] + off[None, :] * dlon  # [n_lon, n_quad]

    # Full quadrature point set, ordered [lat, lon, qi, qj] to match the row-major flattening.
    LA = np.radians(qlat)[:, None, :, None]
    LO = np.radians(qlon)[None, :, None, :]
    LA, LO = np.broadcast_arrays(LA, LO)
    pts = np.stack(
        [np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], axis=-1
    ).reshape(-1, 3)
    aw = np.broadcast_to(np.cos(LA), LA.shape).reshape(-1)  # cos(lat) area element

    centroids = v[faces].mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    cand = cKDTree(centroids).query(pts, k=min(n_cand, len(faces)))[1]
    if cand.ndim == 1:
        cand = cand[:, None]
    fid, bw = _barycentric(v, faces, pts, cand)

    n_cells = n_lat * n_lon
    per_cell = n_quad * n_quad
    row = np.repeat(np.arange(n_cells), per_cell * 3)
    col = faces[fid].reshape(-1)
    val = (bw * aw[:, None]).reshape(-1)

    A = sp.coo_matrix((val, (row, col)), shape=(n_cells, len(v))).tocsr()
    rs = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(rs, 1e-30)) @ A


def latlon_cell_area(grid: str = "1.5") -> np.ndarray:
    """cos(lat) area weights for a named grid, flattened row-major and normalized to mean 1."""
    lats, lons = target_grid(grid)
    w = np.cos(np.radians(lats))
    # Pole rows are half-cells.
    w[0] *= 0.5
    w[-1] *= 0.5
    full = np.repeat(w[:, None], len(lons), axis=1).ravel()
    return full / full.mean()


def cached_mesh_to_latlon(
    mesh: dict[str, np.ndarray], cache_dir: str | Path, n_sub: int, grid: str = "1.5"
) -> sp.csr_matrix:
    """Build once, reuse. The regrid is pure geometry, so it caches like the mesh does."""
    path = Path(cache_dir) / f"regrid_sub{n_sub}_to_{grid.replace('.', 'p')}.npz"
    if path.exists():
        return sp.load_npz(path)
    A = mesh_to_latlon_matrix(mesh, grid=grid)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, A)
    return A


# --------------------------------------------------------------------------------------
# display only
# --------------------------------------------------------------------------------------


def nearest_grid_index(mesh: dict[str, np.ndarray], n_lat: int = 180, n_lon: int = 360):
    """Nearest-neighbour scatter back to a regular grid. Plotting only -- never for scoring."""
    glat = np.linspace(-89.5, 89.5, n_lat)
    glon = np.linspace(-180, 179, n_lon)
    mlon, mlat = np.meshgrid(np.radians(glon), np.radians(glat))
    gxyz = np.stack(
        [np.cos(mlat) * np.cos(mlon), np.cos(mlat) * np.sin(mlon), np.sin(mlat)], axis=-1
    ).reshape(-1, 3)
    nn = cKDTree(mesh["v"]).query(gxyz)[1].reshape(n_lat, n_lon)
    return nn, glat, glon
