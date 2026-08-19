"""UpdateRule + FiLM noise conditioning.

The per-cell MLP that maps a perceived neighbourhood plus conditioning to a residual delta.

**Why the noise perturbs the rule and not the seed.** The NCA applies `n_substeps` rounds of a
smoothing local operator per forecast window. Any perturbation placed in the seed field is
exactly the thing that operator is built to damp -- the network does not have to learn to
ignore it, the dynamics erase it for free. Ensemble collapse would be the *expected* outcome of
seed noise here, not a risk. Perturbing the rule cannot be damped, because it changes the
operator rather than the state it acts on.

A single low-dimensional vector z (16-32 dims for an entire global field) FiLM-modulates every
hidden layer. The low dimensionality is load-bearing: it forces the perturbation to express
itself as coherent, physically plausible spatial structure rather than per-cell noise. This is
the FGN / WeatherNext 2 design.

Two invariants the rest of the code depends on:

* z is drawn **once per member per trajectory** and held constant across all cells and all
  sub-steps. `reseed_hidden` must never re-draw it, or member identity dissolves.
* The FiLM projection is **zero-initialized**, so gamma = 1 and beta = 0 and a freshly
  stochastic model is numerically identical to the deterministic checkpoint it warm-starts
  from. That is what makes the phase 2c -> 3a transition safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class UpdateRule(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.stochastic = cfg.model.stochastic
        in_dim = 4 * cfg.c_state + cfg.c_cond
        h, n = cfg.model.hidden_dim, cfg.model.n_layers

        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, h)] + [nn.Linear(h, h) for _ in range(n - 1)]
        )
        # One (scale, shift) pair per hidden layer, produced from the noise vector.
        self.film = nn.Linear(cfg.model.noise_dim, 2 * h * n)
        nn.init.zeros_(self.film.weight)  # gamma = 1, beta = 0 -> exact deterministic model
        nn.init.zeros_(self.film.bias)

        self.head = nn.Linear(h, cfg.c_state)
        nn.init.zeros_(self.head.weight)  # untrained model is exactly the identity map
        nn.init.zeros_(self.head.bias)

        if cfg.model.spectral_norm:
            # Bound the Lipschitz constant of the update map. The 2c divergence was the
            # weight norm ratcheting across the 20-sub-step stability threshold; with every
            # hidden layer's sigma_max pinned near 1, the per-window gain is bounded by
            # ~(1 + dt * sigma_head)^n_substeps -- note sigma_head is NOT pinned, so this
            # bounds the hidden-layer contribution only; the head remains a free
            # amplification path held by weight_decay, not by construction.
            # Hidden layers ONLY: the head is zero-init and spectral_norm's power iteration
            # divides by the weight's norm, which NaNs on an exactly-zero weight (verified
            # on torch 2.6). film is left free too -- capping it would cap ensemble spread.
            self.layers = nn.ModuleList(
                [nn.utils.spectral_norm(layer) for layer in self.layers]
            )

    def forward(self, perceived: torch.Tensor, cond: torch.Tensor,
                z: torch.Tensor | None = None) -> torch.Tensor:
        """perceived [B, N, 4C] | cond [B, N, c_cond] | z [B, noise_dim] or None.

        B here is the *flattened* batch-and-member axis, so one z per row broadcasts over all
        nodes -- which is exactly the "constant across cells" invariant.
        """
        h = torch.cat([perceived, cond], dim=-1)

        if z is None or not self.stochastic:
            for layer in self.layers:
                h = F.gelu(layer(h))
            return self.head(h)

        mod = self.film(z).view(z.shape[0], len(self.layers), 2, -1)
        for i, layer in enumerate(self.layers):
            gamma = 1.0 + mod[:, i, 0].unsqueeze(1)  # [B, 1, hidden]
            beta = mod[:, i, 1].unsqueeze(1)
            h = F.gelu(gamma * layer(h) + beta)
        return self.head(h)

    def film_parameters(self):
        """The noise pathway, so its gradient norm can be logged separately (phase 3a gate)."""
        return list(self.film.parameters())
