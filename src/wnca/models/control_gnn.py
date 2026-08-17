"""Same-budget non-local baseline: a multi-scale message-passing GNN over the same mesh.

This is the model that turns the milestone's central question from "how far are we from
DeepMind" -- unanswerable on four spot GPUs -- into "does restricting the update to a strictly
local rule cost accuracy, holding everything else fixed," which is answerable and is the actual
thesis under test.

Matched to the NCA on mesh, variables, data, optimizer, and (by tuning `gnn_hidden` /
`gnn_hops`) parameter count and wall-clock. It inherits seeding, conditioning, rollout and
ensemble machinery from `ForecastModel`, so the *only* thing that differs is the update
operator: one non-local message-passing pass per 6 h window instead of `n_substeps` strictly
local ones.

Non-locality comes from the nested icosphere. `icosphere(k)` keeps the vertices of
`icosphere(k-1)` as its first N_{k-1} entries, so the coarse levels' edge lists are valid
long-range edges on the fine mesh -- one hop at level `n_sub - 3` spans roughly eight fine
cells. Layers cycle through the levels coarse-to-fine, which is the multi-scale part.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..mesh.icosphere import edges_from_faces, icosphere
from .nca import ForecastModel


def multiscale_edges(n_sub: int, n_levels: int) -> list[np.ndarray]:
    """Edge lists for levels n_sub, n_sub-1, ..., valid as indices into the fine mesh."""
    out = []
    for lvl in range(n_sub, max(n_sub - n_levels, 0), -1):
        _, faces = icosphere(lvl)
        out.append(edges_from_faces(faces))
    return out


class MessagePassingLayer(nn.Module):
    """One interaction-network round on a fixed edge set."""

    def __init__(self, hidden: int, edges: np.ndarray, n_nodes: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.register_buffer("src", torch.from_numpy(np.ascontiguousarray(edges[:, 0])).long())
        self.register_buffer("dst", torch.from_numpy(np.ascontiguousarray(edges[:, 1])).long())
        # Mean aggregation: degree varies (12 pentagons among the hexagons), and sum
        # aggregation would make those nodes systematically louder.
        deg = np.bincount(edges[:, 1], minlength=n_nodes).astype(np.float32)
        self.register_buffer("inv_deg", torch.from_numpy(1.0 / np.maximum(deg, 1.0)).view(1, -1, 1))

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # [B, N, H]
        m = self.edge_mlp(torch.cat([h[:, self.src], h[:, self.dst]], dim=-1))  # [B, E, H]
        agg = torch.zeros_like(h).index_add_(1, self.dst, m) * self.inv_deg
        return self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))


class ControlGNN(ForecastModel):
    def __init__(self, cfg: Config, mesh: dict[str, np.ndarray]):
        super().__init__()
        self.cfg = cfg
        n_nodes = len(mesh["v"])
        hidden = cfg.model.gnn_hidden
        levels = multiscale_edges(cfg.mesh.n_sub, n_levels=3)

        self.encoder = nn.Sequential(
            nn.Linear(cfg.c_state + cfg.c_cond, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        # Coarse-to-fine, cycling if there are more hops than levels.
        self.layers = nn.ModuleList(
            [MessagePassingLayer(hidden, levels[::-1][i % len(levels)], n_nodes)
             for i in range(cfg.model.gnn_hops)]
        )
        self.decoder = nn.Linear(hidden, cfg.c_state)
        nn.init.zeros_(self.decoder.weight)  # identity at init, same trick as the NCA head
        nn.init.zeros_(self.decoder.bias)

        # Same FiLM noise pathway as the NCA, so a probabilistic control is a config change.
        self.film = nn.Linear(cfg.model.noise_dim, 2 * hidden * len(self.layers))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forecast_step(self, state, static, prev_phys=None, z=None):
        """One non-local pass = one 6 h window. No sub-steps: that is the whole point."""
        cond = self._cond(state, static, prev_phys)
        h = self.encoder(torch.cat([state, cond], dim=-1))

        mod = None
        if z is not None and self.cfg.model.stochastic:
            mod = self.film(z).view(z.shape[0], len(self.layers), 2, -1)

        for i, layer in enumerate(self.layers):
            if self.cfg.model.grad_ckpt and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
            if mod is not None:
                h = (1.0 + mod[:, i, 0].unsqueeze(1)) * h + mod[:, i, 1].unsqueeze(1)
        return state + self.decoder(F.gelu(h))

    def film_parameters(self):
        return list(self.film.parameters())
