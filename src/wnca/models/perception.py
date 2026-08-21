"""MeshPerception -- the NCA "sensor", unchanged in behaviour from M1.

`[identity | grad_x | grad_y | laplacian]` per node -> 4C channels, the mesh analogue of the
filter bank image NCAs use. The coefficients are geometry, not parameters: computed once by
`mesh/operators.py`, registered as buffers, never trained.

Implementation note: M1 aggregated with `scatter_add_` over a [B, E, C] index tensor. At M2's
60 state channels, ~61k edges and an ensemble folded into the batch that index alone is
hundreds of MB of int64 per call. Each operator is linear in the field, so all three are
assembled once as sparse [N, N] matrices instead -- same arithmetic, no index materialization.
`tests/test_operators.py` checks the two forms agree.
"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn

from ..mesh.operators import dilated_laplacian, edge_op_matrix


def _to_torch_sparse(A) -> torch.Tensor:
    A = A.tocoo()
    idx = torch.from_numpy(np.stack([A.row, A.col])).long()
    val = torch.from_numpy(A.data).float()
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


class MeshPerception(nn.Module):
    """Measured note: stacking the three operators into one [3N, N] sparse matrix to save two
    kernel launches was tried and was *slower* (33 ms vs 20 ms at B*C = 480) -- permuting the
    3x-wider result costs more memory traffic than the launches save. Three separate calls it
    is. Direct measurement over clever measurement, per the standing methodology note.
    """

    def __init__(self, mesh: dict[str, np.ndarray], dilation: int = 0, fanout: int = 6):
        super().__init__()
        n = len(mesh["v"])
        self.n_nodes = n
        self.dilation = int(dilation)
        for name in ("gx", "gy", "lap"):
            self.register_buffer(name, _to_torch_sparse(edge_op_matrix(mesh["edges"], mesh[name], n)))
        # Optional fifth group: mean over the ring at exactly `dilation` hops, minus the
        # centre. Same 6-neighbour fan-out as the local stencil, so it costs one more sparse
        # matmul of the same size regardless of radius -- and it applies uniformly to every
        # node, unlike coarse-icosphere shortcuts which only connect the nodes that exist at
        # the coarse level.
        self.register_buffer(
            "dil",
            _to_torch_sparse(dilated_laplacian(mesh["edges"], mesh["v"], self.dilation, fanout))
            if self.dilation else None)

    def _apply_op(self, op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply an [N, N] sparse operator to [B, N, C] by folding B and C into the dense side."""
        B, N, C = x.shape
        flat = x.permute(1, 0, 2).reshape(N, B * C)
        out = torch.sparse.mm(op, flat)
        return out.reshape(N, B, C).permute(1, 0, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, C] -> [B, N, 4C]
        # `torch.sparse.mm` has no half-precision CUDA kernel ("addmm_sparse_cuda not
        # implemented for 'Half'"), so perception is forced to fp32 under AMP. That costs
        # little: perception is ~20% of a sub-step and is memory-bound, while the update MLP
        # -- the other ~80%, and dense -- still gets the tensor cores.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            groups = [xf, self._apply_op(self.gx, xf), self._apply_op(self.gy, xf),
                      self._apply_op(self.lap, xf)]
            if self.dil is not None:
                groups.append(self._apply_op(self.dil, xf))
            return torch.cat(groups, dim=-1)
