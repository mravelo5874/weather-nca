"""Time-varying external forcing: top-of-atmosphere solar geometry.

Phase 2b measured the gap this fills. `2m_temperature` scored **-48% skill at +24 h** -- much
worse than doing nothing -- because 24 h is exactly one diurnal cycle, so persistence compares
the same local solar time and gets the day/night swing for free. The model's static
conditioning was orography, land-sea mask, sin(lat) and cos(lat): nothing time-varying, so it
had no way to know the hour and smoothed the swing away instead. Skill dipped at 24 h and 48 h
and was positive in between -- a sawtooth, which is the diurnal signature. `temperature_850`,
a free-atmosphere control, showed no such structure.

Three channels, all cheap analytic functions of (time, lat, lon):

    cos_zenith   max(0, cos of the solar zenith angle) -- a TOA insolation proxy, and the
                 quantity that actually drives surface heating. Zero on the night side.
    sin_doy      annual cycle, so the model can distinguish January from July at the same
    cos_doy      solar elevation.

These are held constant across the sub-steps of a forecast window, exactly like the rest of the
conditioning, and evaluated at the **target** time of each window -- the model is told what the
sun is doing over the interval it is integrating into.

Per-timestep quantities are reduced to three scalars (sin/cos declination and the UTC hour
angle) so the per-node expansion is a couple of multiplies on GPU. Nothing is cached: the
forcing is a pure function of the timestamps the cache already stores, so adding it does not
invalidate an existing cache.
"""

from __future__ import annotations

import numpy as np
import torch

N_SOLAR_CHANNELS = 3
_DAYS_PER_YEAR = 365.2422


def _as_datetime64(times: np.ndarray) -> np.ndarray:
    return np.asarray(times).astype("datetime64[s]")


def timestep_solar_params(times: np.ndarray) -> np.ndarray:
    """Per-timestep solar scalars: [T, 3] = (sin declination, cos declination, UTC hour angle).

    Declination uses the standard first-order series; it is accurate to a few tenths of a degree,
    which is far below the resolution at which any of this matters for a 6 h forecast window.
    """
    t = _as_datetime64(times)
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    seconds_into_year = (t - year_start).astype("float64")
    day_of_year = seconds_into_year / 86400.0

    # Declination: 23.44 deg tilt, zero near the March equinox (~day 80).
    gamma = 2.0 * np.pi * (day_of_year - 80.0) / _DAYS_PER_YEAR
    dec = np.radians(23.44) * np.sin(gamma)

    # UTC hour angle: 0 at midnight UTC, so local hour angle is this plus longitude.
    frac_day = (seconds_into_year % 86400.0) / 86400.0
    utc_angle = 2.0 * np.pi * frac_day - np.pi

    return np.stack([np.sin(dec), np.cos(dec), utc_angle], axis=-1).astype(np.float32)


def day_of_year_angle(times: np.ndarray) -> np.ndarray:
    """[T, 2] = (sin, cos) of the annual cycle."""
    t = _as_datetime64(times)
    year_start = t.astype("datetime64[Y]").astype("datetime64[s]")
    doy = (t - year_start).astype("float64") / 86400.0
    a = 2.0 * np.pi * doy / _DAYS_PER_YEAR
    return np.stack([np.sin(a), np.cos(a)], axis=-1).astype(np.float32)


def synthetic_times(n: int, start: str = "2015-01-01", step_hours: int = 6) -> np.ndarray:
    """6-hourly timestamps, for synthetic data and tests where no real times exist."""
    return np.datetime64(start, "s") + np.arange(n) * np.timedelta64(step_hours * 3600, "s")


class SolarForcing:
    """Expands per-timestep solar scalars to per-node forcing channels on the fly.

    Materializing [T, N, 3] would be 2.3 GB per channel for the 39-year split; computing it
    per batch is a handful of multiplies and costs nothing.
    """

    def __init__(self, times: np.ndarray, mesh: dict, device: str | torch.device = "cpu"):
        self.n_times = len(times)
        p = timestep_solar_params(times)  # [T, 3]
        doy = day_of_year_angle(times)  # [T, 2]
        self.params = torch.from_numpy(p).to(device)
        self.doy = torch.from_numpy(doy).to(device)
        lat = torch.from_numpy(np.radians(mesh["lat"]).astype(np.float32)).to(device)
        lon = torch.from_numpy(np.radians(mesh["lon"]).astype(np.float32)).to(device)
        self.sin_lat, self.cos_lat, self.lon = torch.sin(lat), torch.cos(lat), lon

    def to(self, device) -> "SolarForcing":
        for a in ("params", "doy", "sin_lat", "cos_lat", "lon"):
            setattr(self, a, getattr(self, a).to(device))
        return self

    def at(self, idx: torch.Tensor) -> torch.Tensor:
        """Forcing at absolute time indices `idx` [...] -> [..., N, 3].

        Indices past the end of the series are clamped rather than wrapped: a rollout may run
        past the last timestamp of a split, and clamping degrades to a fixed solar state instead
        of jumping discontinuously back to January.
        """
        idx = torch.as_tensor(idx, device=self.params.device).long().clamp(0, self.n_times - 1)
        p = self.params[idx]  # [..., 3]
        sin_dec, cos_dec, utc = p[..., 0:1], p[..., 1:2], p[..., 2:3]

        # cos(zenith) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(hour angle)
        hour_angle = utc + self.lon  # broadcast [..., 1] + [N] -> [..., N]
        cos_z = self.sin_lat * sin_dec + self.cos_lat * cos_dec * torch.cos(hour_angle)
        cos_z = cos_z.clamp_min(0.0)  # night contributes no insolation

        d = self.doy[idx]  # [..., 2]
        n = cos_z.shape[-1]
        sin_doy = d[..., 0:1].expand(*d.shape[:-1], n)
        cos_doy = d[..., 1:2].expand(*d.shape[:-1], n)
        return torch.stack([cos_z, sin_doy, cos_doy], dim=-1)

    def window(self, start_idx: torch.Tensor, n_windows: int) -> torch.Tensor:
        """Forcing for the TARGET time of each of `n_windows` windows: [B, W, N, 3].

        `start_idx` is the index of the seed field, so window k targets `start_idx + 1 + k`.
        """
        offsets = torch.arange(1, n_windows + 1, device=self.params.device)
        return self.at(start_idx.view(-1, 1) + offsets.view(1, -1))
