"""1.5 degree regrid + WeatherBench-2 leaderboard comparison.

Two comparability requirements, both work items rather than footnotes:

**The regrid.** WB2 probabilistic numbers are published at 1.5 degrees lat-lon. Scoring the
mesh against them requires an explicit regrid, and the operator must be conservative
(area-weighted), not nearest-neighbour, or the RMSE is biased. `mesh/regrid.py` builds it once,
`tests/test_regrid.py` round-trips it, and it is cached and reused.

**Which truth.** WB2 scores operational models against IFS analysis and ML models against ERA5.
This project is on the ERA5 side. So:

    - the GenCast comparison is like-for-like;
    - the IFS ENS comparison is slightly FAVOURABLE to us, because ERA5 is our training
      target and is not what IFS ENS was scored against.

Do not quote the IFS ENS row as a clean win. This comment exists so nobody has to rediscover
that, which is exactly what the plan asked for.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from ..config import HOURS_PER_WINDOW, Config
from ..mesh.regrid import cached_mesh_to_latlon, latlon_cell_area

# Published WB2 reference numbers, z500 RMSE / CRPS in m^2/s^2 at 1.5 degrees.
# Source: docs/milestone-2-plan.md, Evaluation. Scored against ERA5 unless noted.
WB2_REFERENCE = {
    "ifs_ens_crps_z500": {24: 22.4, 72: 58.3, 240: 262.0},  # scored vs IFS analysis, see above
    "deterministic_rmse_z500_frontier": {24: 55.0, 72: 250.0},  # GraphCast-class, approximate
}


class WB2Scorer:
    """Regrids mesh fields to the WB2 grid and scores them there, not on the mesh.

    Scoring on the mesh and comparing to a 1.5 degree leaderboard number is not a comparison,
    it is two different metrics with the same name.
    """

    def __init__(self, cfg: Config, mesh, device: str = "cpu"):
        self.cfg = cfg
        A = cached_mesh_to_latlon(mesh, cfg.data.cache_dir, cfg.mesh.n_sub, cfg.eval.wb2_grid)
        self.A = _to_torch_sparse(A).to(device)
        w = latlon_cell_area(cfg.eval.wb2_grid)
        self.cell_w = torch.from_numpy(w.astype(np.float32)).to(device).view(-1, 1)
        self.device = device

    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """[..., N, C] -> [..., n_cells, C], area-weighted conservative."""
        # fp32: no half kernel for sparse.mm, and this is the scoring path where precision
        # matters more than speed.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            shape = x.shape
            flat = x.reshape(-1, shape[-2], shape[-1])
            B, N, C = flat.shape
            dense = flat.permute(1, 0, 2).reshape(N, B * C)
            out = torch.sparse.mm(self.A, dense)
            out = out.reshape(-1, B, C).permute(1, 0, 2)
            return out.reshape(*shape[:-2], -1, C)

    def rmse(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Area-weighted RMSE on the WB2 grid, per channel. Squares pooled, root last."""
        p, t = self.to_grid(pred), self.to_grid(truth)
        sq = (p - t) ** 2
        dims = tuple(range(sq.ndim - 1))
        num = (sq * self.cell_w).sum(dim=dims)
        den = (torch.ones_like(sq) * self.cell_w).sum(dim=dims)
        return torch.sqrt(num / den)

    def crps(self, members: torch.Tensor, truth: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Fair CRPS on the WB2 grid, per channel. members [B, M, N, C]."""
        from ..losses.crps import fair_crps_pointwise

        B, M = members.shape[:2]
        g = self.to_grid(members.reshape(B * M, *members.shape[2:]))
        g = g.view(B, M, *g.shape[1:])
        pw = fair_crps_pointwise(g, self.to_grid(truth), alpha)
        dims = tuple(range(pw.ndim - 1))
        num = (pw * self.cell_w).sum(dim=dims)
        den = (torch.ones_like(pw) * self.cell_w).sum(dim=dims)
        return num / den


def _to_torch_sparse(A: sp.spmatrix) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.from_numpy(np.stack([A.row, A.col])).long()
    return torch.sparse_coo_tensor(idx, torch.from_numpy(A.data).float(), A.shape).coalesce()


def compare_to_reference(crps_by_lead: dict[int, float], key: str = "ifs_ens_crps_z500") -> str:
    """Format our z500 CRPS against a published reference, with the caveat attached."""
    ref = WB2_REFERENCE[key]
    lines = [f"z500 CRPS vs {key} (m^2/s^2)",
             f"{'lead':>7} {'ours':>10} {'reference':>10} {'ratio':>8}"]
    for h in sorted(ref):
        if h in crps_by_lead:
            r = crps_by_lead[h] / ref[h]
            lines.append(f"{h:>5}h  {crps_by_lead[h]:>10.1f} {ref[h]:>10.1f} {r:>7.2f}x")
    if key == "ifs_ens_crps_z500":
        lines.append("  note: IFS ENS is scored against IFS analysis, we are scored against")
        lines.append("  ERA5. This comparison is slightly favourable to us -- do not quote it")
        lines.append("  as a clean win. The GenCast comparison is the like-for-like one.")
    return "\n".join(lines)


def lead_to_window(hours: int) -> int:
    return hours // HOURS_PER_WINDOW
