"""Chebyshev band-pass filters on the mesh Laplacian.

The cotangent Laplace-Beltrami operator is already built and already validated against
spherical harmonics (M1 section 1), so band-pass filters come free via Chebyshev polynomials
of the scaled Laplacian -- no eigendecomposition, just a few sparse matvecs per band.

    L~ = 2 (-L) / lambda_max - I        (spectrum in [-1, 1])
    T_0 = x,  T_1 = L~ x,  T_k = 2 L~ T_{k-1} - T_{k-2}

A band-pass filter is a fixed linear combination of the T_k. On the unit sphere a degree-l
spherical harmonic has Laplacian eigenvalue -l(l+1), so band b covers a contiguous range of
total wavenumber -- which is what makes `tests/test_spectral.py` a real check rather than a
tautology.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .operators import laplacian_matrix, spectral_radius


def band_edges(lambda_max: float, n_bands: int) -> np.ndarray:
    """Band boundaries, equally spaced in sqrt(lambda) ~ wavenumber.

    Equal spacing in lambda itself would put almost every resolved scale in the top band,
    because lambda ~ l^2.
    """
    return np.linspace(0.0, np.sqrt(lambda_max), n_bands + 1) ** 2


def _band_response(lam: np.ndarray, edges: np.ndarray, b: int) -> np.ndarray:
    """Raised-cosine window for band `b`, in sqrt(lambda) space.

    Adjacent windows overlap on a half-cosine and sum to 1 everywhere, so the bands partition
    the field's energy rather than double-counting or dropping a range.
    """
    s = np.sqrt(np.maximum(lam, 0.0))
    e = np.sqrt(edges)
    lo, hi = e[b], e[b + 1]
    out = np.ones_like(s)

    if b > 0:  # rising edge shared with band b-1
        prev = e[b - 1]
        mid = 0.5 * (prev + lo)
        rise = (s - mid) / max(lo - mid, 1e-30)
        out = np.where(s < mid, 0.0, np.where(s < lo, 0.5 * (1 - np.cos(np.pi * np.clip(rise, 0, 1))), out))
    if b < len(edges) - 2:  # falling edge shared with band b+1
        nxt = e[b + 2]
        mid = 0.5 * (hi + nxt)
        fall = (s - hi) / max(mid - hi, 1e-30)
        out = np.where(s > mid, 0.0, np.where(s > hi, 0.5 * (1 + np.cos(np.pi * np.clip(fall, 0, 1))), out))
    return out


def chebyshev_coeffs(lambda_max: float, n_bands: int, order: int) -> np.ndarray:
    """Chebyshev coefficients [n_bands, order + 1] for the band responses.

    Chebyshev-Gauss quadrature of  c_k = (2/pi) int h(lambda(y)) T_k(y) / sqrt(1 - y^2) dy,
    with c_0 halved so the series is  sum_k c_k T_k.
    """
    edges = band_edges(lambda_max, n_bands)
    M = max(4 * (order + 1), 128)
    theta = np.pi * (np.arange(M) + 0.5) / M
    y = np.cos(theta)
    lam = 0.5 * lambda_max * (y + 1.0)  # invert L~ = 2 lambda / lambda_max - 1

    c = np.zeros((n_bands, order + 1))
    for b in range(n_bands):
        h = _band_response(lam, edges, b)
        for k in range(order + 1):
            c[b, k] = (2.0 / M) * np.sum(h * np.cos(k * theta))
    c[:, 0] *= 0.5
    return c


class BandFilters:
    """Chebyshev band-pass filters, held as a torch sparse operator plus fixed coefficients.

    `energies(x)` returns per-band area-weighted mean square. `log_energies(x)` is what the
    spectral loss term scores, so that a band two orders of magnitude weaker than another
    still contributes to the gradient.
    """

    def __init__(
        self,
        laplacian: sp.spmatrix,
        area: np.ndarray,
        n_bands: int = 5,
        order: int = 24,
        lambda_max: float | None = None,
        device: str | torch.device = "cpu",
    ):
        self.n_bands, self.order = n_bands, order
        # -L is positive semi-definite; power iteration on it gives the spectral radius.
        neg = (-laplacian).tocsr()
        self.lambda_max = float(lambda_max if lambda_max is not None else spectral_radius(neg))
        self.edges = band_edges(self.lambda_max, n_bands)

        # L~ = 2 (-L) / lambda_max - I
        N = neg.shape[0]
        scaled = (neg * (2.0 / self.lambda_max) - sp.eye(N, format="csr")).tocoo()
        idx = torch.from_numpy(np.stack([scaled.row, scaled.col])).long()
        val = torch.from_numpy(scaled.data).float()
        self.L = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce().to(device)

        c = chebyshev_coeffs(self.lambda_max, n_bands, order)
        self.coeffs = torch.from_numpy(c).float().to(device)  # [n_bands, order+1]
        w = torch.from_numpy(np.asarray(area, dtype=np.float32)).to(device)
        self.area_w = (w / w.mean()).view(1, -1, 1)  # [1, N, 1], mean 1

    def to(self, device: str | torch.device) -> "BandFilters":
        self.L = self.L.to(device)
        self.coeffs = self.coeffs.to(device)
        self.area_w = self.area_w.to(device)
        return self

    def _spmm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L~ to [B, N, C] by folding batch and channel into the dense side."""
        B, N, C = x.shape
        flat = x.permute(1, 0, 2).reshape(N, B * C)
        out = torch.sparse.mm(self.L, flat)
        return out.reshape(N, B, C).permute(1, 0, 2)

    def filter_bands(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, C] -> [B, n_bands, N, C] via the Chebyshev recursion (one pass, all bands)."""
        t_prev, t_cur = x, self._spmm(x)
        out = self.coeffs[:, 0].view(-1, 1, 1, 1) * t_prev.unsqueeze(0)
        out = out + self.coeffs[:, 1].view(-1, 1, 1, 1) * t_cur.unsqueeze(0)
        for k in range(2, self.order + 1):
            t_prev, t_cur = t_cur, 2.0 * self._spmm(t_cur) - t_prev
            out = out + self.coeffs[:, k].view(-1, 1, 1, 1) * t_cur.unsqueeze(0)
        return out.permute(1, 0, 2, 3)

    def energies(self, x: torch.Tensor) -> torch.Tensor:
        """Area-weighted mean square per band: [B, N, C] -> [B, n_bands, C]."""
        f = self.filter_bands(x)
        return (self.area_w.unsqueeze(1) * f.pow(2)).mean(dim=2)

    def log_energies(self, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return torch.log(self.energies(x) + eps)


def build_band_filters(mesh: dict[str, np.ndarray], cfg, device="cpu", cache_dir=None) -> BandFilters:
    """Build the filters, caching the (expensive-once) lambda_max power iteration."""
    L = laplacian_matrix(mesh)
    lam = None
    if cache_dir is not None:
        p = Path(cache_dir) / f"lambda_max_sub{cfg.mesh.n_sub}.npy"
        if p.exists():
            lam = float(np.load(p))
        else:
            lam = spectral_radius((-L).tocsr())
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.asarray(lam))
    return BandFilters(
        L,
        mesh["area"],
        n_bands=cfg.loss.n_bands,
        order=cfg.loss.cheby_order,
        lambda_max=lam,
        device=device,
    )
