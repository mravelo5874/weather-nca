"""The dilated perception ring -- the minimal-delta locality control.

Phase 2d's GNN control turned out to be a confounded one: it changes locality, iteration count,
weight sharing and input featurisation all at once, and -- measured -- it reaches *less* far per
window than the strictly local model it was meant to control for (9 hops against 20).

This is the single-variable alternative. Everything about the phase-2c model is held fixed and
one operator is added to perception: the mean over the ring at exactly `perception_dilation`
hops, minus the centre. Same 6-neighbour fan-out as the local stencil, so it costs one more
sparse matmul of constant size whatever the radius, and it applies to every node uniformly.

The two arms that matter:

    perception_dilation = 1   five groups, but the fifth carries NO extra reach
    perception_dilation = 8   five groups, each sub-step now reaches 8 hops

They have **identical parameter counts**, so a difference between them is reach and nothing
else. `= 0` is the phase-2c architecture and must stay bit-compatible with its checkpoints.
"""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np
import pytest
import torch

from wnca.mesh.icosphere import edges_from_faces, icosphere
from wnca.mesh.operators import dilated_laplacian, dilated_ring_edges
from wnca.models.nca import build_model


def _dil(cfg, d, **model_over):
    return dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, perception_dilation=d, **model_over))


def _params(cfg, mesh):
    return sum(p.numel() for p in build_model(cfg, mesh, "cpu").parameters())


# ---------------------------------------------------------------------------- the operator ---
def test_ring_is_a_proper_averaging_laplacian(small_mesh):
    """Every row must be mean(ring) - centre: off-diagonal summing to +1, diagonal -1. An
    earlier version normalised by the neighbour's IN-degree (which varies 0-48) instead of the
    centre's fan-out, producing a valid-looking but wrongly weighted operator."""
    n = len(small_mesh["v"])
    for hops in (1, 2, 3):
        L = dilated_laplacian(small_mesh["edges"], n, hops).tocsr()
        off = L.copy()
        off.setdiag(0)
        off.eliminate_zeros()
        rows = np.asarray(off.sum(1)).ravel()
        assert np.allclose(rows, 1.0), f"hops={hops}: off-diagonal rows sum to {rows.min()}"
        assert np.allclose(L.diagonal(), -1.0)


def test_ring_fanout_is_constant_and_uniform_across_nodes(small_mesh):
    """The whole point against the coarse-icosphere control: cost independent of radius, and
    every node treated alike. A coarse level's edges only connect nodes that exist at that
    level, which is why stacking them reaches less far than the local model."""
    n = len(small_mesh["v"])
    for hops in (1, 2, 3):
        ring = dilated_ring_edges(small_mesh["edges"], n, hops, fanout=6)
        fan = np.bincount(ring[:, 1], minlength=n)
        assert fan.max() <= 6
        assert fan.min() >= 5, f"hops={hops}: some node has fan-out {fan.min()}"


def test_ring_members_sit_at_exactly_the_requested_distance(small_mesh):
    n = len(small_mesh["v"])
    _, faces = icosphere(3)
    adj = [[] for _ in range(n)]
    for a, b in edges_from_faces(faces):
        adj[a].append(b)
        adj[b].append(a)

    def bfs(src):
        d = np.full(n, -1)
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if d[w] < 0:
                    d[w] = d[u] + 1
                    q.append(w)
        return d

    for hops in (2, 3):
        ring = dilated_ring_edges(small_mesh["edges"], n, hops, fanout=6)
        for centre in (0, 17, 100):
            members = ring[ring[:, 1] == centre][:, 0]
            d = bfs(centre)
            assert all(d[j] == hops for j in members), \
                f"hops={hops}, centre={centre}: got distances {sorted({d[j] for j in members})}"


# --------------------------------------------------------------------------------- the model ---
def test_dilation_zero_is_bit_compatible_with_phase_2c(tiny_cfg, small_mesh):
    """`0` must leave the architecture and its hash exactly as they were, or every existing
    checkpoint stops loading."""
    base = _dil(tiny_cfg, 0)
    assert base.arch_hash() == tiny_cfg.arch_hash()
    assert base.n_perception_groups == 4
    assert _params(base, small_mesh) == _params(tiny_cfg, small_mesh)


def test_the_two_control_arms_are_exactly_parameter_matched(tiny_cfg, small_mesh):
    """THE property the experiment rests on: dilation 1 and 8 differ only in reach."""
    a, b = _dil(tiny_cfg, 1), _dil(tiny_cfg, 8)
    assert _params(a, small_mesh) == _params(b, small_mesh)
    assert a.n_perception_groups == b.n_perception_groups == 5
    assert a.arch_hash() != b.arch_hash(), "the radius must change the hash: buffers differ"


def test_enabling_dilation_widens_the_update_mlp(tiny_cfg, small_mesh):
    assert _params(_dil(tiny_cfg, 8), small_mesh) > _params(_dil(tiny_cfg, 0), small_mesh)


@pytest.mark.parametrize("dilation", [0, 1, 3])
def test_forward_is_finite_and_correctly_shaped(tiny_cfg, small_mesh, dilation):
    from wnca.models.perception import MeshPerception

    cfg = _dil(tiny_cfg, dilation)
    p = MeshPerception(small_mesh, dilation)
    x = torch.randn(2, len(small_mesh["v"]), cfg.c_state)
    out = p(x)
    assert out.shape == (2, len(small_mesh["v"]), cfg.n_perception_groups * cfg.c_state)
    assert torch.isfinite(out).all()


def test_perception_survives_autocast_with_dilation(tiny_cfg, small_mesh):
    """Sparse ops have no half kernel; perception forces fp32 internally. The extra operator
    goes through the same path and must not break it."""
    from wnca.models.perception import MeshPerception

    p = MeshPerception(small_mesh, 3)
    x = torch.randn(2, len(small_mesh["v"]), tiny_cfg.c_state)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = p(x)
    assert torch.isfinite(out).all()


def test_dilation_actually_increases_the_receptive_field(tiny_cfg, small_mesh, forcing_for):
    """Measured, not assumed -- by gradient support, the same way the GNN control's reach was
    measured. Two sub-steps keep the 20-sub-step gradient blow-up out of the way; reach scales
    linearly with sub-step count.

    `dilation=1` must match `dilation=0`: that is what makes it the no-extra-reach control.
    """
    n = len(small_mesh["v"])
    _, faces = icosphere(3)
    adj = [[] for _ in range(n)]
    for a, b in edges_from_faces(faces):
        adj[a].append(b)
        adj[b].append(a)
    probe = 100
    d = np.full(n, -1)
    d[probe] = 0
    q = deque([probe])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if d[w] < 0:
                d[w] = d[u] + 1
                q.append(w)

    def reach(dilation):
        cfg = _dil(tiny_cfg, dilation, n_substeps=2, dt=0.5)
        torch.manual_seed(0)
        m = build_model(cfg, small_mesh, "cpu").eval()
        torch.nn.init.normal_(m.update.head.weight, std=1e-3)
        torch.manual_seed(1)
        phys = torch.randn(1, n, cfg.c_phys)
        static = torch.randn(1, n, cfg.state.c_static)
        forcing = forcing_for(cfg, small_mesh, 1, 1)[:, 0]
        s0 = m.seed(phys).detach().requires_grad_(True)
        m.forecast_step(s0, static, forcing=forcing)[0, probe].sum().backward()
        g = s0.grad[0].abs().sum(-1).numpy()
        assert np.isfinite(g).all()
        return int(d[g > g.max() * 1e-9].max())

    r0, r1, r3 = reach(0), reach(1), reach(3)
    assert r1 == r0, f"dilation=1 must add no reach (got {r1} vs {r0}) -- it is the control"
    assert r3 > r0, f"dilation=3 must reach further than local (got {r3} vs {r0})"
    assert r3 >= r0 + 2, f"dilation=3 barely widened anything: {r3} vs {r0}"
    # Clean linear scaling (reach = n_substeps x dilation) is NOT asserted here: this mesh has
    # 642 nodes, so a 3-hop ring already spans a large fraction of the sphere and two of them
    # wrap rather than compose -- measured 4 hops where 6 would be the flat-space answer. On
    # the real n_sub=5 mesh it does scale: at 2 sub-steps, dilation 4 -> 8 hops and dilation
    # 8 -> 16, i.e. exactly n_substeps x dilation.
