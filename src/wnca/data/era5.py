"""WeatherBench-2 zarr -> mesh tensors. The axis-order transpose fix lives here.

M1 lost a 45-minute training run to a silently transposed ERA5 store. `regrid_to_mesh` and
`latlon_to_mesh_weights` both assume (..., latitude, longitude) flattened row-major. WB2 zarr
does NOT guarantee that axis order -- several stores are (time, level, longitude, latitude) --
and reading one layout as the other shears each row by a constant offset, turning real z500
into diagonal stripes that a model will happily learn to fit.

So every field is explicitly transposed here. That makes the contract with the regridder
unconditional rather than store-dependent, which is the whole point.
"""

from __future__ import annotations

import numpy as np

from ..config import WB2_PATHS, Config
from ..mesh.regrid import latlon_to_mesh_weights, regrid_to_mesh

STATIC_FIELDS = ("geopotential_at_surface", "land_sea_mask")


def open_wb2(cfg: Config):
    import xarray as xr

    return xr.open_zarr(WB2_PATHS[cfg.data.wb2_res], chunks=None, storage_options={"token": "anon"})


def mesh_weights(cfg: Config, mesh: dict[str, np.ndarray], ds):
    """Bilinear lat-lon -> mesh weights for the store's native grid. Computed once per run."""
    return latlon_to_mesh_weights(ds.latitude.values, ds.longitude.values, mesh["lat"], mesh["lon"])


def select_split(cfg: Config, ds, years: tuple[int, ...]):
    """Lazily subset the store to `years` (and to `max_steps_per_split`, for smoke runs)."""
    sub = ds.sel(time=ds.time.dt.year.isin(list(years)))
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"no ERA5 timesteps found for years {years}")
    if cfg.data.max_steps_per_split:
        sub = sub.isel(time=slice(0, min(sub.sizes["time"], cfg.data.max_steps_per_split)))
    return sub


def regrid_chunk(
    cfg: Config, sub, mesh: dict[str, np.ndarray], idx: np.ndarray, w: np.ndarray,
    t0: int, t1: int, out: np.ndarray,
) -> None:
    """Regrid timesteps [t0, t1) of `sub` into `out[t0:t1]`, one channel at a time.

    Chunked so a preempted cache build resumes from a timestep boundary rather than restarting.
    `out` is typically a memmap.
    """
    window = sub.isel(time=slice(t0, t1))
    for ci, ch in enumerate(cfg.variables.channels()):
        if ch.name not in window:
            raise KeyError(f"variable {ch.name!r} not in the WB2 store at {cfg.data.wb2_res}")
        da = window[ch.name]
        if ch.level is not None:
            da = da.sel(level=ch.level)
        # The transpose that the corrupted-ERA5 run existed to teach us about.
        da = da.transpose("time", "latitude", "longitude")
        out[t0:t1, :, ci] = regrid_to_mesh(np.asarray(da.values), idx, w)


def load_era5(
    cfg: Config,
    mesh: dict[str, np.ndarray],
    years: tuple[int, ...],
    ds=None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-split convenience loader. Returns (fields[T, N, C], times[T]).

    Channels are in `cfg.variables.channels()` order; values are raw physical units.
    Normalization happens in `data/normalize.py`, from the train split only.
    """
    ds = ds if ds is not None else open_wb2(cfg)
    sub = select_split(cfg, ds, years)
    idx, w = mesh_weights(cfg, mesh, ds)
    T = sub.sizes["time"]
    out = np.empty((T, len(mesh["v"]), cfg.c_phys), dtype=np.float32)
    if verbose:
        print(f"  streaming {T} timesteps x {cfg.c_phys} channels at {cfg.data.wb2_res} ...")
    regrid_chunk(cfg, sub, mesh, idx, w, 0, T, out)
    return out, sub.time.values


def load_static(cfg: Config, mesh: dict[str, np.ndarray], ds=None) -> np.ndarray:
    """Static conditioning: normalized orography, land-sea mask, sin(lat), cos(lat) -> [N, 4]."""
    n = len(mesh["v"])
    lat_r = np.radians(mesh["lat"])

    orog = np.zeros(n, dtype=np.float32)
    lsm = np.zeros(n, dtype=np.float32)
    if cfg.data.source == "era5":
        ds = ds if ds is not None else open_wb2(cfg)
        idx, w = latlon_to_mesh_weights(ds.latitude.values, ds.longitude.values, mesh["lat"], mesh["lon"])
        for name, dst in (("geopotential_at_surface", "orog"), ("land_sea_mask", "lsm")):
            if name in ds:
                arr = np.asarray(ds[name].transpose("latitude", "longitude").values)
                vals = regrid_to_mesh(arr[None], idx, w)[0]
                if dst == "orog":
                    orog = vals
                else:
                    lsm = vals

    static = np.stack(
        [
            (orog - orog.mean()) / (orog.std() + 1e-8),
            lsm,
            np.sin(lat_r),
            np.cos(lat_r),
        ],
        axis=-1,
    ).astype(np.float32)
    if static.shape[1] != cfg.state.c_static:
        raise ValueError(f"built {static.shape[1]} static channels, config says {cfg.state.c_static}")
    return static


def plot_raw_frame(cfg: Config, ds=None, var: str = "geopotential", level: int | None = 500):
    """Plot one raw native-grid frame before regridding.

    Ten seconds of insurance against the failure mode that voided an entire M1 run: expect a
    few large mid-latitude troughs and ridges, NOT regular diagonal stripes.
    """
    import matplotlib.pyplot as plt

    ds = ds if ds is not None else open_wb2(cfg)
    da = ds[var]
    if level is not None and "level" in da.dims:
        da = da.sel(level=level)
    da = da.transpose("time", "latitude", "longitude").isel(time=0)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.pcolormesh(ds.longitude.values, ds.latitude.values, np.asarray(da.values),
                       cmap="RdBu_r", shading="auto")
    ax.set_title(f"raw ERA5 {var}{'' if level is None else f' {level}'}, first frame")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# synthetic fallback -- same interface, so the whole pipeline runs with no network
# --------------------------------------------------------------------------------------


def make_synthetic(cfg: Config, mesh: dict[str, np.ndarray], n_steps: int, seed: int = 0):
    """Advecting vortices + a stationary orographic wave, broadcast across channels.

    Deterministic, periodic, band-limited, zero Lyapunov exponent: its verdicts on *stability*
    are meaningful, its numbers for *skill* are not. Used by `make smoke` and by tests.
    """
    rng = np.random.default_rng(seed)
    n = len(mesh["v"])
    lat, lon = np.radians(mesh["lat"]), np.radians(mesh["lon"])
    channels = cfg.variables.channels()

    orog = (
        np.exp(-((lat - 0.6) ** 2 + (np.mod(lon + 2.0, 2 * np.pi) - np.pi) ** 2) / 0.25)
        + 0.6 * np.exp(-((lat + 0.5) ** 2 + (np.mod(lon - 1.0, 2 * np.pi) - np.pi) ** 2) / 0.30)
    ).astype(np.float32)

    n_vort = 6
    lat0 = rng.uniform(-1.0, 1.0, n_vort)
    lon0 = rng.uniform(0, 2 * np.pi, n_vort)
    amp = rng.uniform(0.5, 1.5, n_vort) * rng.choice([-1, 1], n_vort)
    width = rng.uniform(0.25, 0.5, n_vort)
    U = 0.18  # rad per 6 h window (~30 m/s at midlatitudes)

    base = np.zeros((n_steps, n), dtype=np.float32)
    for t in range(n_steps):
        f = 0.8 * orog
        for k in range(n_vort):
            lonk = lon0[k] + U * t / np.maximum(np.cos(lat0[k]), 0.3)
            dlon = np.arctan2(np.sin(lon - lonk), np.cos(lon - lonk))
            ang = np.arccos(
                np.clip(np.sin(lat) * np.sin(lat0[k]) + np.cos(lat) * np.cos(lat0[k]) * np.cos(dlon), -1, 1)
            )
            f = f + amp[k] * np.exp(-((ang / width[k]) ** 2))
        base[t] = f

    # Give each channel its own offset, scale and phase lag so the coupling machinery has
    # something non-degenerate to chew on.
    out = np.empty((n_steps, n, len(channels)), dtype=np.float32)
    for ci, ch in enumerate(channels):
        lag = ci % 3
        scale = 1.0 + 0.3 * ci
        offset = 10.0 * ci
        shifted = np.roll(base, lag, axis=0)
        vals = shifted * scale + offset
        if ch.name in cfg.variables.log_transform:
            vals = np.abs(vals) * 1e-5 + 1e-8  # plausible specific-humidity magnitudes
        out[:, :, ci] = vals
    return out
