"""Band-energy spectrum evaluation.

Two cautions carried from M1, both of which cost real time:

* The original spectrum check was **one-sided**: it read a 14x excess of high-wavenumber power
  as if it were over-smoothing. Report the ratio and name both failure directions.
* A matched spectrum means the right *amount* of variance per scale, not the right
  *placement*. M1's 100.9% retained power at 24 h sat alongside real phase error along the
  fronts. Band energies are a necessary check, not a sufficient one -- reliability and CRPS are
  the calibration evidence.

Exit criterion 4 is about **individual members**, never the ensemble mean. The mean is smooth
by construction and would pass a check the members fail, which is precisely the FourCastNet 3
finding that put the spectral term in the loss.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from ..losses.spectral import band_energy_ratio


@torch.no_grad()
def member_spectra(
    model,
    cfg: Config,
    cache,
    bands,
    split: str = "test",
    lead_windows: int = 12,
    n_starts: int = 8,
    n_members: int | None = None,
    device: str = "cpu",
    mesh=None,
) -> dict:
    """Per-band energy of individual members relative to ERA5, at one lead time.

    Returns ratios [n_bands, C] for members and, separately, for the ensemble mean -- the gap
    between the two is the quantity FCN3 says pointwise CRPS gets wrong.
    """
    model.eval()
    series = cache.split(split).array
    M = n_members if n_members is not None else (cfg.ensemble.m_test if cfg.model.stochastic else 1)
    st = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)

    # This stage is reached only when cfg.model.stochastic, so phase 3a is the first config to
    # run it with solar forcing ON -- and without this the rollout raises. Found by smoking
    # phase3a_probe; it would otherwise have crashed a paid run at eval time, after training.
    solar = None
    if cfg.state.solar_forcing:
        if mesh is None:
            raise ValueError("member_spectra needs `mesh` when state.solar_forcing is on")
        from ..data.forcing import SolarForcing

        solar = SolarForcing(cache.times(split), mesh, device)

    usable = len(series) - lead_windows - 2
    starts = np.unique(np.linspace(1, max(usable, 1), min(n_starts, max(usable, 1))).astype(int))

    mem_acc, mean_acc = [], []
    for s0 in starts:
        prev = torch.from_numpy(np.array(series[s0 - 1], dtype=np.float32)).unsqueeze(0).to(device)
        cur = torch.from_numpy(np.array(series[s0], dtype=np.float32)).unsqueeze(0).to(device)
        truth = torch.from_numpy(
            np.array(series[s0 + lead_windows], dtype=np.float32)
        ).unsqueeze(0).to(device)

        fw = solar.window(torch.tensor([int(s0)], device=device), lead_windows) if solar else None
        pred = model.rollout_ensemble(model.seed(cur), st, lead_windows, prev_phys=prev,
                                      n_members=M, forcing=fw)
        last = pred[:, :, -1]  # [1, M, N, C]

        members = last.reshape(M, *last.shape[2:])
        mem_acc.append(band_energy_ratio(members, truth.expand_as(members), bands).cpu().numpy())
        mean_acc.append(band_energy_ratio(last.mean(dim=1), truth, bands).cpu().numpy())

    return {
        "member_ratio": np.mean(mem_acc, axis=0),  # [n_bands, C]
        "mean_ratio": np.mean(mean_acc, axis=0),
        "band_edges": bands.edges,
        "band_l": _edges_as_wavenumber(bands.edges),
        "lead_hours": lead_windows * 6,
        "channels": tuple(c.key for c in cfg.variables.channels()),
    }


def _edges_as_wavenumber(edges: np.ndarray) -> np.ndarray:
    """Laplacian eigenvalue lambda = l(l+1) on the unit sphere -> total wavenumber l."""
    return (-1 + np.sqrt(1 + 4 * np.asarray(edges))) / 2


def verdict(ratio: float, low: float = 0.8, high: float = 1.2) -> str:
    """Name both failure directions, so the M1 one-sided read cannot recur."""
    if ratio < low:
        return "over-smooth / blurring"
    if ratio > high:
        return "grid-scale noise or amplitude blow-up"
    return "comparable"


def summarize(spec: dict, channel: str = "geopotential_500") -> str:
    ci = spec["channels"].index(channel) if channel in spec["channels"] else 0
    lines = [f"band energy vs ERA5 at +{spec['lead_hours']}h, channel {spec['channels'][ci]}",
             f"{'band (l)':>14} {'members':>10} {'ens mean':>10}  verdict"]
    l = spec["band_l"]
    for b in range(len(spec["member_ratio"])):
        mr = spec["member_ratio"][b, ci]
        er = spec["mean_ratio"][b, ci]
        lines.append(f"{l[b]:>6.0f}-{l[b + 1]:<7.0f} {mr:>10.3f} {er:>10.3f}  {verdict(mr)}")
    return "\n".join(lines)
