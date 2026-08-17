"""Band-energy CRPS -- the spectral term.

FourCastNet 3's finding: pointwise CRPS training alone does not produce correct spectra. Prior
CRPS models scored well but had spatially uncorrelated, spectrally wrong members. The fix is a
composite loss with a spectral term.

Scoring per-member **log** band energies against truth's with the same fair CRPS keeps the whole
objective inside one proper-scoring framework, rather than bolting an MSE term onto a CRPS
loss. Logs matter: a band two orders of magnitude weaker than another would otherwise
contribute nothing to the gradient.

Whether this is needed *for an NCA* is genuinely unknown -- the local-rule inductive bias may
already preserve the spectrum, as M1's 100.9% retained power at 24 h weakly suggests. Phase 3b
runs it as an ablation, and the answer is worth writing down either way.
"""

from __future__ import annotations

import torch

from ..mesh.spectral import BandFilters
from .crps import fair_crps_pointwise


def band_energy_crps(members: torch.Tensor, truth: torch.Tensor, bands: BandFilters,
                     chan_w: torch.Tensor | None = None, alpha: float = 1.0) -> torch.Tensor:
    """Fair CRPS over per-member log band energies.

    members [B, M, N, C] | truth [B, N, C] -> scalar.

    Band energies are a global reduction, so there is no spatial axis left to area-weight --
    the weighting already happened inside `BandFilters.energies`.
    """
    B, M = members.shape[:2]
    flat = members.reshape(B * M, *members.shape[2:])
    e_mem = bands.log_energies(flat).view(B, M, bands.n_bands, -1)  # [B, M, n_bands, C]
    e_truth = bands.log_energies(truth)  # [B, n_bands, C]

    pw = fair_crps_pointwise(e_mem, e_truth, alpha)  # [B, n_bands, C]
    if chan_w is not None:
        w = chan_w.view(1, 1, -1)
        return (pw * w).sum() / (torch.ones_like(pw) * w).sum()
    return pw.mean()


def band_energy_ratio(pred: torch.Tensor, truth: torch.Tensor, bands: BandFilters) -> torch.Tensor:
    """Per-band energy of `pred` relative to `truth`: [B, N, C] -> [n_bands, C].

    The diagnostic form. Exit criterion 4 wants individual members within 20% of ERA5 at 72 h,
    so this is called on members, never on the ensemble mean -- the mean is smooth by
    construction and would pass a check the members fail.
    """
    ep = bands.energies(pred).mean(dim=0)
    et = bands.energies(truth).mean(dim=0)
    return ep / et.clamp_min(1e-30)
