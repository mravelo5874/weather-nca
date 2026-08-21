"""WeatherNCA -- perception + update rule, composed over sub-steps.

Three time granularities, and keeping them straight is the whole architecture:

| level            | what it is                          | method          |
|------------------|-------------------------------------|-----------------|
| **sub-step**     | one residual PDE update; the atomic op | `nca_step`      |
| **forecast step**| `n_substeps` sub-steps = one 6 h window | `forecast_step` |
| **rollout**      | repeated forecast steps, autoregressive | `rollout`       |

Confusing them is the most common error in this codebase.

M2 adds an ensemble axis. Members are folded into the batch dimension for every tensor op
(`[B, M, N, C] -> [B*M, N, C]`), so perception and the update rule stay unchanged and the
memory cost is linear in M rather than requiring a second code path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ..config import Config
from .perception import MeshPerception
from .update import UpdateRule


class ForecastModel(nn.Module):
    """Everything that is *not* the update operator: seeding, conditioning, rollout, ensembles.

    The NCA and the control GNN share all of it and differ only in `forecast_step`. That is
    what makes phase 2d a controlled experiment rather than two loosely related runs -- same
    state layout, same rollout semantics, same conditioning, same loss; only locality differs.
    """

    cfg: Config

    def forecast_step(self, state, static, prev_phys=None, z=None, forcing=None):  # pragma: no cover
        raise NotImplementedError

    def reseed_hidden(self, state: torch.Tensor) -> torch.Tensor:
        """Zero the hidden channels, keeping the physical ones.

        Single-step training never presents a non-zero hidden state at the start of a forecast
        step, so carrying it forward at inference feeds the MLP channels it has never been
        trained on. That mismatch -- not blurring, not CFL -- was the M1 blow-up.

        This does NOT touch z. Re-drawing the noise vector here would dissolve member identity
        and make spread meaningless at long leads.
        """
        c = self.cfg.c_phys
        return torch.cat([state[..., :c], torch.zeros_like(state[..., c:])], dim=-1)

    def _cond(self, state, static, prev_phys, forcing=None):
        """[static | solar forcing | tendency] -- held fixed across the window's sub-steps.

        `forcing` is evaluated at the window's TARGET time: the model is told what the sun is
        doing over the interval it is integrating into.
        """
        parts = [static]
        if self.cfg.state.solar_forcing:
            if forcing is None:
                # Defaulting to zeros here would mean permanent polar night, and the model
                # would train on it without complaint. Fail loudly instead.
                raise ValueError(
                    "state.solar_forcing is enabled but no forcing was supplied to the rollout. "
                    "Pass forcing=... (see data/forcing.SolarForcing.window), or disable "
                    "state.solar_forcing."
                )
            parts.append(forcing)
        if self.cfg.state.second_order:
            cur = state[..., : self.cfg.c_phys]
            parts.append(cur - prev_phys if prev_phys is not None else torch.zeros_like(cur))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    # ---- rollout ----
    def rollout(self, state, static, n_windows, prev_phys=None, z=None, return_aux=False,
                forcing=None):
        """Physical field at every window: [B, n_windows, N, c_phys].

        With `return_aux=True` also returns mean hidden-channel overflow, measured BEFORE any
        re-seeding so the penalty still bites when re-seeding is on.
        """
        c = self.cfg.c_phys
        cur, outs = state[..., :c], []
        ovf = state.new_zeros(())
        for w in range(n_windows):
            fw = forcing[:, w] if forcing is not None else None
            state = self.forecast_step(state, static, prev_phys, z, fw)
            prev_phys, cur = cur, state[..., :c]
            outs.append(cur)
            ovf = ovf + torch.relu(state[..., c:].abs() - 1.0).mean()
            if self.cfg.model.reseed_hidden:
                state = self.reseed_hidden(state)
        out = torch.stack(outs, dim=1)
        return (out, ovf / n_windows) if return_aux else out

    # ---- ensemble ----
    def rollout_ensemble(self, state, static, n_windows, prev_phys=None, n_members=1,
                         z=None, generator=None, return_aux=False, forcing=None):
        """Run `n_members` members and return [B, M, n_windows, N, c_phys].

        One z per member per trajectory, drawn here unless supplied, then held constant for
        every cell and every sub-step of the whole rollout.
        """
        B = state.shape[0]
        M = n_members
        dev, dtype = state.device, state.dtype

        if self.cfg.model.stochastic:
            if z is None:
                z = torch.randn(B, M, self.cfg.model.noise_dim, device=dev, dtype=dtype,
                                generator=generator)
            z_flat = z.reshape(B * M, -1)
        else:
            z_flat = None

        state_f = _expand(state, M)
        static_f = _expand(static, M)
        prev_f = _expand(prev_phys, M) if prev_phys is not None else None
        # [B, W, N, K] -> [B*M, W, N, K], members adjacent to match `_expand`.
        forcing_f = None
        if forcing is not None:
            B0, W, N, K = forcing.shape
            forcing_f = forcing.unsqueeze(1).expand(B0, M, W, N, K).reshape(B0 * M, W, N, K)

        out = self.rollout(state_f, static_f, n_windows, prev_f, z_flat, return_aux=return_aux,
                           forcing=forcing_f)
        pred, aux = out if return_aux else (out, None)
        pred = pred.view(B, M, *pred.shape[1:])
        return (pred, aux) if return_aux else pred

    # ---- seed ----
    def seed(self, phys0: torch.Tensor) -> torch.Tensor:
        """Observed field -> full cell state; hidden channels start at zero."""
        B, N, _ = phys0.shape
        return torch.cat([phys0, phys0.new_zeros(B, N, self.cfg.state.c_hidden)], dim=-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class WeatherNCA(ForecastModel):
    """The local rule: `n_substeps` residual PDE updates per 6 h window."""

    def __init__(self, cfg: Config, perception: MeshPerception):
        super().__init__()
        self.cfg = cfg
        self.perceive = perception
        self.update = UpdateRule(cfg)

    # ---- one PDE sub-step (deterministic mask: every cell, every step) ----
    def nca_step(self, state: torch.Tensor, cond: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        return state + self.cfg.model.dt * self.update(self.perceive(state), cond, z)

    # ---- one 6 h window = n_substeps sub-steps (the CFL budget) ----
    def forecast_step(self, state, static, prev_phys=None, z=None, forcing=None):
        cond = self._cond(state, static, prev_phys, forcing)  # held fixed across sub-steps
        ckpt = self.cfg.model.grad_ckpt and self.training and torch.is_grad_enabled()
        for _ in range(self.cfg.model.n_substeps):
            if ckpt:
                # Store only sub-step boundaries; recompute the interior in backward.
                state = torch.utils.checkpoint.checkpoint(
                    self.nca_step, state, cond, z, use_reentrant=False
                )
            else:
                state = self.nca_step(state, cond, z)
        return state


def _expand(x: torch.Tensor, M: int) -> torch.Tensor:
    """[B, N, C] -> [B*M, N, C], members adjacent so `.view(B, M, ...)` inverts it."""
    B, N, C = x.shape
    return x.unsqueeze(1).expand(B, M, N, C).reshape(B * M, N, C)


def build_model(cfg: Config, mesh, device: str = "cpu") -> nn.Module:
    """Construct the model named by `cfg.model.kind`."""
    if cfg.model.kind == "nca":
        return WeatherNCA(cfg, MeshPerception(mesh, cfg.model.perception_dilation)).to(device)
    if cfg.model.kind == "control_gnn":
        from .control_gnn import ControlGNN

        return ControlGNN(cfg, mesh).to(device)
    raise ValueError(f"unknown model.kind {cfg.model.kind!r}")
