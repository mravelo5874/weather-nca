"""Icosahedral mesh construction, refinement, vertex areas.

Ported verbatim in behaviour from the M1 notebook. Square lat-lon grids distort badly toward
the poles; cells live on a subdivided icosahedron instead.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..config import R_EARTH, Config
from .operators import build_perception_coeffs


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return v, f


def icosphere(n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Recursively subdivide an icosahedron, re-projecting to the unit sphere."""
    v, f = icosahedron()
    v = list(v)
    for _ in range(n_sub):
        cache: dict[tuple[int, int], int] = {}
        nf: list[list[int]] = []

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (v[a] + v[b]) / 2.0
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.array(nf, dtype=np.int64)
    return np.array(v), f


def edges_from_faces(f: np.ndarray) -> np.ndarray:
    """Unique directed edges (both directions), self-loops removed."""
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.unique(np.concatenate([e, e[:, ::-1]], axis=0), axis=0)
    return e[e[:, 0] != e[:, 1]]


def tangent_basis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local east / north orthonormal frame at each node."""
    zc = np.broadcast_to(np.array([0.0, 0.0, 1.0]), v.shape)
    east = np.cross(zc, v)
    n = np.linalg.norm(east, axis=1, keepdims=True)
    east = np.where(n < 1e-8, np.array([1.0, 0.0, 0.0]), east / np.maximum(n, 1e-30))
    east /= np.linalg.norm(east, axis=1, keepdims=True)
    north = np.cross(v, east)
    north /= np.linalg.norm(north, axis=1, keepdims=True)
    return east, north


def mesh_path(cfg: Config) -> Path:
    return cfg.cache_dir / f"mesh_sub{cfg.mesh.n_sub}.npz"


def build_mesh(cfg: Config, verbose: bool = True) -> dict[str, np.ndarray]:
    """Build (or load) the mesh and its perception coefficients.

    Returns v, faces, edges, gx, gy, lap, area, lat, lon. The coefficients are geometry, not
    parameters: computed once, never trained.
    """
    path = mesh_path(cfg)
    if path.exists():
        m = dict(np.load(path))
        if verbose:
            print(f"mesh loaded from cache: N={len(m['v'])}  ({path})")
        return m

    t0 = time.time()
    v, faces = icosphere(cfg.mesh.n_sub)
    edges = edges_from_faces(faces)
    gx, gy, lap, area = build_perception_coeffs(v, faces, edges)
    m = dict(
        v=v,
        faces=faces,
        edges=edges,
        gx=gx,
        gy=gy,
        lap=lap,
        area=area,
        lat=np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))),
        lon=np.degrees(np.arctan2(v[:, 1], v[:, 0])),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **m)
    if verbose:
        print(f"mesh built in {time.time() - t0:.1f}s: N={len(v)}  E={len(edges)}  -> {path}")
    return m


def mean_spacing_km(n_nodes: int) -> float:
    return math.sqrt(4 * math.pi * R_EARTH**2 / n_nodes) / 1e3
