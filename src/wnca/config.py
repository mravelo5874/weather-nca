"""Config dataclasses, YAML load, schema validation.

One `base.yaml` plus per-phase overrides, loaded into frozen dataclasses and validated at
startup. Every run writes its fully-resolved config next to its checkpoints, so reproducing a
run means pointing at that file rather than remembering what was edited.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

# WeatherBench-2 ERA5 zarr stores, verified against the WB2 data guide.
WB2_PATHS = {
    "64x32": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
    "240x121": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
    "1440x721": "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
}

R_EARTH = 6.371e6
HOURS_PER_WINDOW = 6


# --------------------------------------------------------------------------------------
# channel spec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Channel:
    """One physical prognostic channel. `level` is None for surface variables."""

    name: str
    level: int | None = None

    @property
    def key(self) -> str:
        return self.name if self.level is None else f"{self.name}_{self.level}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.key


@dataclass(frozen=True)
class VariablesConfig:
    """Which ERA5 fields make up the physical state, and in what channel order.

    Channel order is `atmospheric x levels` (variable-major) followed by `surface`, and it is
    frozen by this ordering rule — never by the order things happen to appear in the zarr.
    Normalization stats, the scorecard, and every checkpoint depend on it.
    """

    atmospheric: tuple[str, ...] = (
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
    )
    levels: tuple[int, ...] = (850, 700, 500, 300, 250)
    surface: tuple[str, ...] = (
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
    )
    # Spans orders of magnitude across levels; without the log the 850 hPa channel dominates
    # every gradient. See docs/milestone-2-plan.md, Data.
    log_transform: tuple[str, ...] = ("specific_humidity",)
    log_offset: float = 1e-9

    def channels(self) -> tuple[Channel, ...]:
        out = [Channel(v, lv) for v in self.atmospheric for lv in self.levels]
        out += [Channel(v, None) for v in self.surface]
        return tuple(out)

    @property
    def c_phys(self) -> int:
        return len(self.channels())

    def index_of(self, name: str, level: int | None = None) -> int:
        return self.channels().index(Channel(name, level))


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshConfig:
    n_sub: int = 5  # 3->642, 4->2562, 5->10242 (~223 km), 6->40962


@dataclass(frozen=True)
class StateConfig:
    """Cell state is `[physical | hidden]`; static channels ride alongside as conditioning."""

    c_hidden: int = 32
    c_static: int = 4  # orography, land-sea mask, sin(lat), cos(lat)
    second_order: bool = True  # feed the tendency (x_t - x_{t-1}) as conditioning
    # Top-of-atmosphere solar geometry (cos zenith + annual cycle) as time-varying
    # conditioning. Without it the diurnal cycle is unrepresentable and 2m_temperature scores
    # WORSE than persistence at 24 h. Measured in phase 2b; see docs/milestone-2-findings.md.
    solar_forcing: bool = True


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "nca"  # "nca" | "control_gnn"
    hidden_dim: int = 512
    n_layers: int = 4

    # --- time integration: sub-steps are PDE sub-steps, not forecast steps ---
    n_substeps: int = 20  # THE CFL BUDGET. Validated at n_sub=5 on z500 in M1.
    dt: float = 0.05

    # --- probabilistic head ---
    noise_dim: int = 16  # FiLM noise vector z, drawn once per member per trajectory
    stochastic: bool = False  # phases 3a+ only; 2a-2d are deterministic

    # --- rollout semantics ---
    reseed_hidden: bool = True  # zero hidden channels between forecast windows

    # Spectral-normalize the update MLP's hidden layers, so the composed sub-step map's gain
    # is bounded by construction instead of depending on where the weight norm happens to
    # drift. Phase 2c's divergence was exactly that drift crossing the recurrence's stability
    # threshold (docs/cloud-compute-incidents.md, attempt 5). The head is NOT wrapped: it is
    # zero-init, and spectral_norm's power iteration NaNs on an exactly-zero weight
    # (verified empirically, torch 2.6). The head's norm is asserted on checkpoint load.
    spectral_norm: bool = False

    grad_ckpt: bool = True  # recompute sub-steps in backward

    # --- control GNN only ---
    gnn_hops: int = 4
    gnn_hidden: int = 256


@dataclass(frozen=True)
class DataConfig:
    source: str = "era5"  # "era5" | "synthetic"
    wb2_res: str = "240x121"
    train_years: tuple[int, ...] = tuple(range(1979, 2018))
    val_years: tuple[int, ...] = (2018,)
    test_years: tuple[int, ...] = (2020,)  # matches the WB2 probabilistic leaderboard
    cache_dir: str = "./wnca_cache"
    cache_dtype: str = "float32"  # "float16" halves a 39-year multi-variable cache
    max_steps_per_split: int | None = None  # smoke / debugging truncation
    plot_raw_era5: bool = False


@dataclass(frozen=True)
class LossConfig:
    w_field: float = 1.0
    w_spec: float = 0.0  # phase 3b turns this on
    w_over: float = 1e-2  # overflow penalty on |hidden| > 1, carried from M1
    crps_alpha: float = 1.0  # 1.0 = fully fair; AIFS uses ~0.95 for gradient stability
    n_bands: int = 5
    cheby_order: int = 24
    # Per-channel loss weights, keyed by Channel.key. Unlisted channels get 1.0.
    channel_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleConfig:
    m_train: int = 4
    m_val: int = 16
    m_test: int = 50


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: bool = False

    # M1 showed the rollout curriculum contributed nothing across two attempts and destroyed
    # the long-lead metric twice. Do not re-enable without a measurement. See CLAUDE.md.
    rollout_epochs: int = 0
    rollout_windows: tuple[int, ...] = (2, 4, 8)
    rollout_lr_scale: float = 0.03

    # Sanchez-Gonzalez-style input noise (M1). Distinct from `pushforward` below.
    noise_std: float = 0.05
    # Brandstetter pushforward: unroll one window under no_grad, then take the step from
    # that state. ~1.5x the cost of single-step, none of the instability of backprop
    # through 160 sub-steps.
    pushforward: bool = False

    ckpt_windows: int = 8  # ONE fixed selection metric, every phase, never compared across
    # Fraction of validation start times the selection metric uses. An 8-window rollout over
    # the full split costs more than the training pass itself at M2 sizes. The subset is
    # evenly spaced and FIXED for the whole phase -- a resampled subset would make epoch-to-
    # epoch comparison meaningless, which is M1 incident 2 in a new costume.
    ckpt_subsample: float = 1.0
    ckpt_every_steps: int = 0  # 0 = epoch boundaries only
    warm_start: str | None = None  # checkpoint path to initialize from
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    max_windows: int = 60  # 15 days at 6 h
    n_starts: int = 32
    lead_hours: tuple[int, ...] = (6, 12, 24, 48, 72, 120, 168, 240, 360)
    wb2_grid: str = "1.5"  # regrid target for leaderboard comparison


@dataclass(frozen=True)
class TrackingConfig:
    wandb: bool = False
    project: str = "weather-nca"
    run_name: str | None = None
    out_dir: str = "./runs"


@dataclass(frozen=True)
class Config:
    phase: str = "base"
    mesh: MeshConfig = field(default_factory=MeshConfig)
    variables: VariablesConfig = field(default_factory=VariablesConfig)
    state: StateConfig = field(default_factory=StateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # ---- derived ----
    @property
    def c_phys(self) -> int:
        return self.variables.c_phys

    @property
    def c_state(self) -> int:
        return self.c_phys + self.state.c_hidden

    @property
    def c_forcing(self) -> int:
        from .data.forcing import N_SOLAR_CHANNELS

        return N_SOLAR_CHANNELS if self.state.solar_forcing else 0

    @property
    def c_cond(self) -> int:
        return (self.state.c_static + self.c_forcing
                + (self.c_phys if self.state.second_order else 0))

    @property
    def cache_dir(self) -> Path:
        return Path(self.data.cache_dir)

    def arch_hash(self) -> str:
        """Fingerprint of everything that fixes a parameter shape.

        Checkpoint load asserts against this. M1 ran three evaluation rounds on a model whose
        architecture had silently drifted from the one that produced the checkpoint.
        """
        payload = {
            "n_sub": self.mesh.n_sub,
            "channels": [c.key for c in self.variables.channels()],
            "c_hidden": self.state.c_hidden,
            "c_static": self.state.c_static,
            "second_order": self.state.second_order,
            "kind": self.model.kind,
            "hidden_dim": self.model.hidden_dim,
            "n_layers": self.model.n_layers,
            "noise_dim": self.model.noise_dim,
            "gnn_hops": self.model.gnn_hops,
            "gnn_hidden": self.model.gnn_hidden,
        }
        # Optional features are included only when ENABLED, so turning one on changes the hash
        # (it changes the parameter shapes) while leaving checkpoints made before the feature
        # existed still loadable. Adding a field unconditionally would invalidate every prior
        # checkpoint in the project.
        if self.state.solar_forcing:
            payload["solar_forcing"] = True
        if self.model.spectral_norm:
            # Spectral norm reparametrizes `weight` (extra buffers, moved parameter), so the
            # state-dict keys differ from a plain model even though the shapes do not.
            payload["spectral_norm"] = True
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    # ---- validation ----
    def validate(self) -> Config:
        v, m, t, ls = self.variables, self.model, self.train, self.loss
        if self.mesh.n_sub < 1:
            raise ValueError(f"mesh.n_sub must be >= 1, got {self.mesh.n_sub}")
        if not v.channels():
            raise ValueError("variables resolve to zero physical channels")
        if len(set(c.key for c in v.channels())) != len(v.channels()):
            raise ValueError("duplicate channel in variables spec")
        for name in v.log_transform:
            if name not in v.atmospheric and name not in v.surface:
                raise ValueError(f"log_transform names unknown variable {name!r}")
        if m.kind not in ("nca", "control_gnn"):
            raise ValueError(f"unknown model.kind {m.kind!r}")
        if m.n_substeps < 1:
            raise ValueError("model.n_substeps must be >= 1")
        if m.stochastic and m.noise_dim < 1:
            raise ValueError("model.stochastic requires noise_dim >= 1")
        if self.data.source == "era5" and self.data.wb2_res not in WB2_PATHS:
            raise ValueError(f"unknown data.wb2_res {self.data.wb2_res!r}")
        if self.data.cache_dtype not in ("float32", "float16"):
            raise ValueError("data.cache_dtype must be float32 or float16")
        if set(self.data.train_years) & set(self.data.val_years):
            raise ValueError("train_years and val_years overlap")
        if set(self.data.train_years) & set(self.data.test_years):
            raise ValueError("train_years and test_years overlap")
        if not 0.0 <= ls.crps_alpha <= 1.0:
            raise ValueError("loss.crps_alpha must be in [0, 1]")
        if ls.w_spec > 0 and ls.n_bands < 2:
            raise ValueError("spectral term needs loss.n_bands >= 2")
        unknown = set(ls.channel_weights) - {c.key for c in v.channels()}
        if unknown:
            raise ValueError(f"loss.channel_weights names unknown channels: {sorted(unknown)}")
        if self.ensemble.m_train < 2 and m.stochastic:
            raise ValueError("fair CRPS needs ensemble.m_train >= 2")
        if t.rollout_epochs > 0 and not _ROLLOUT_OVERRIDE:
            raise ValueError(
                "train.rollout_epochs > 0 is disabled: M1 ran the curriculum twice, it "
                "contributed nothing and destroyed the long-lead metric both times. "
                "Set WNCA_ALLOW_ROLLOUT_CURRICULUM=1 and write down the measurement first."
            )
        if t.ckpt_windows < 1:
            raise ValueError("train.ckpt_windows must be >= 1")
        return self


_ROLLOUT_OVERRIDE = bool(__import__("os").environ.get("WNCA_ALLOW_ROLLOUT_CURRICULUM"))


# --------------------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------------------


def _coerce(tp: Any, val: Any) -> Any:
    """Coerce a YAML scalar/sequence to the annotated type."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or str(origin) == "typing.Union" or origin is type(int | None):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return None if val is None else _coerce(args[0], val)
    if origin is tuple:
        (inner,) = [a for a in typing.get_args(tp) if a is not Ellipsis][:1] or [Any]
        return tuple(_coerce(inner, x) for x in _as_sequence(val))
    if origin is dict:
        return dict(val or {})
    if is_dataclass(tp):
        return _build(tp, val or {})
    if tp is bool:
        return bool(val)
    if tp is int:
        return int(val)
    if tp is float:
        return float(val)
    if tp is str:
        return str(val)
    return val


def _as_sequence(val: Any) -> list:
    """Accept `[1979, 1980]`, `1979-2017`, or a bare scalar for tuple fields."""
    if isinstance(val, str) and "-" in val.strip() and val.strip()[0].isdigit():
        lo, hi = (int(p) for p in val.split("-", 1))
        return list(range(lo, hi + 1))
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _build(cls: Any, d: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {k: _coerce(hints[k], v) for k, v in d.items()}
    return cls(**kwargs)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# n_sub=3 (642 nodes) + one month of data; `make smoke` must finish in under two minutes.
SMOKE_OVERRIDES: dict[str, Any] = {
    "phase": "smoke",
    "mesh": {"n_sub": 3},
    "model": {"hidden_dim": 64, "n_layers": 2, "n_substeps": 4},
    "state": {"c_hidden": 8},
    "data": {"max_steps_per_split": 120, "train_years": [2015], "val_years": [2016], "test_years": [2017]},
    "train": {"epochs": 1, "batch_size": 2, "ckpt_windows": 2, "warmup_steps": 5},
    "ensemble": {"m_train": 2, "m_val": 2, "m_test": 2},
    "eval": {"max_windows": 4, "n_starts": 2, "lead_hours": [6, 12, 24]},
    "tracking": {"wandb": False},
}


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    smoke: bool = False,
) -> Config:
    """Load `configs/base.yaml`, overlay a phase config, then any explicit overrides.

    A phase config may set `extends:` to chain from something other than base.yaml.
    """
    raw: dict[str, Any] = {}
    chain: list[Path] = []
    if path is not None:
        p = Path(path)
        if not p.exists() and (CONFIG_DIR / p).exists():
            p = CONFIG_DIR / p
        chain.append(p)
        seen = {p.resolve()}
        while True:
            parent = _read_yaml(chain[0]).get("extends", "base.yaml" if chain[0].name != "base.yaml" else None)
            if not parent:
                break
            pp = (chain[0].parent / parent).resolve()
            if pp in seen:
                raise ValueError(f"circular `extends` at {pp}")
            seen.add(pp)
            chain.insert(0, pp)
    else:
        chain = [CONFIG_DIR / "base.yaml"]

    for p in chain:
        d = _read_yaml(p)
        d.pop("extends", None)
        raw = _deep_merge(raw, d)

    if smoke:
        raw = _deep_merge(raw, SMOKE_OVERRIDES)
    if overrides:
        raw = _deep_merge(raw, overrides)

    return _build(Config, raw).validate()


def pick_device(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)
    return "mps" if mps() else "cpu"
