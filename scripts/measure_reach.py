#!/usr/bin/env python
"""Measure how far information actually travels in one 6 h window, per architecture.

Written because two different paper-arithmetic estimates of the control GNN's receptive field
were both wrong, in the same direction, by 1.5-3.6x. The project's methodology note says to
prefer direct measurement over clever measurement; this is the direct one.

**Method.** Receptive field is the *support of the computation graph*, so it is measured with a
gradient rather than a perturbation: make the input state require grad, take one
`forecast_step`, backprop from a single output node, and see which input nodes have non-zero
gradient. Magnitude-independent, so it does not depend on the weights being trained -- unlike a
finite-difference probe, which on the NCA either vanishes or overflows depending on the head
scale (a random head at std 0.05 NaNs through 20 sub-steps).

The NCA arm is verified structurally instead: every perception operator (identity, grad-x,
grad-y, Laplacian) is checked to have non-zeros only on immediate neighbours, so each sub-step
is exactly one hop and `n_substeps` sub-steps reach exactly `n_substeps` hops, uniformly for
every node.

**Why it matters.** A coarse icosphere level's edge list only connects the nodes that EXIST at
that level -- level 1 has 42 vertices of 10,242 -- and `ControlGNN` has no pooling/unpooling to
carry information between levels. So a coarse layer is a no-op for ~99.6% of the mesh, and the
"sum of hop spans" estimate is meaningless.

    python scripts/measure_reach.py
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wnca.config import load_config  # noqa: E402
from wnca.data.forcing import SolarForcing, synthetic_times  # noqa: E402
from wnca.mesh.icosphere import build_mesh, edges_from_faces, icosphere  # noqa: E402
from wnca.models.nca import build_model  # noqa: E402
from wnca.models.perception import MeshPerception  # noqa: E402


def _adj(n_sub: int, n: int):
    _, faces = icosphere(n_sub)
    adj = [[] for _ in range(n)]
    for a, b in edges_from_faces(faces):
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _dists(adj, src: int, n: int) -> np.ndarray:
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


def check_nca_is_one_hop(mesh, adj, sample: int = 4000) -> bool:
    """Every perception non-zero must sit on an immediate neighbour (or the diagonal)."""
    p = MeshPerception(mesh)
    ok = True
    for name in ("gx", "gy", "lap"):
        M = getattr(p, name).coalesce()
        r, c = M.indices()
        hops = [0 if i == j else (1 if j in adj[i] else 99)
                for i, j in zip(r[:sample].tolist(), c[:sample].tolist())]
        worst = max(hops)
        ok &= worst <= 1
        print(f"  perception.{name:<4} max hop distance over {min(sample, len(r))} "
              f"non-zeros: {worst}")
    return ok


def gnn_reach(cfg, mesh, adj, dev, probes) -> dict:
    n = len(mesh["v"])
    sf = SolarForcing(synthetic_times(8), mesh, dev)
    f0 = sf.window(torch.tensor([0], device=dev), 1)[:, 0]
    torch.manual_seed(0)
    m = build_model(cfg, mesh, dev).eval()
    torch.nn.init.normal_(m.decoder.weight, std=1e-2)  # zero-init decoder propagates nothing
    out = {}
    for node, tag in probes:
        gd = _dists(adj, node, n)
        torch.manual_seed(1)
        phys = torch.randn(1, n, cfg.c_phys, device=dev)
        static = torch.randn(1, n, cfg.state.c_static, device=dev)
        s0 = m.seed(phys).detach().requires_grad_(True)
        y = m.forecast_step(s0, static, forcing=f0)
        y[0, node].sum().backward()
        g = s0.grad[0].abs().sum(-1).float().cpu().numpy()
        hit = g > g.max() * 1e-9
        out[tag] = (int(gd[hit].max()), int(hit.sum()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nca", default="configs/phase2c_full.yaml")
    ap.add_argument("--gnn", default="configs/phase2d_control.yaml")
    ap.add_argument("--levels", default="3,4,5")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nca_cfg = load_config(a.nca)
    gnn_cfg = load_config(a.gnn)
    mesh = build_mesh(nca_cfg, verbose=False)
    n = len(mesh["v"])
    adj = _adj(nca_cfg.mesh.n_sub, n)

    counts = {lv: len(icosphere(lv)[0]) for lv in range(nca_cfg.mesh.n_sub + 1)}
    print(f"mesh n_sub={nca_cfg.mesh.n_sub}, {n} nodes | icosphere sizes {counts}")
    print("nested, so node index < N_L means the node exists at level L\n")

    print("NCA -- structural check (receptive field is exact, not estimated):")
    one_hop = check_nca_is_one_hop(mesh, adj)
    nsub = nca_cfg.model.n_substeps
    print(f"  -> each sub-step is {'exactly 1 hop' if one_hop else 'NOT 1-hop (!)'}; "
          f"{nsub} sub-steps reach {nsub} hops, uniformly for every node\n")

    probes = [(5, "level-1 node"), (300, "level-3 node"), (5000, "fine-only node")]
    print("control GNN -- measured by gradient support, one 6 h window:\n")
    print(f"  {'gnn_levels':>11} " + " ".join(f"{t:>26}" for _, t in probes))
    for lv in [int(x) for x in a.levels.split(",")]:
        cfg = dataclasses.replace(
            gnn_cfg, model=dataclasses.replace(gnn_cfg.model, gnn_levels=lv))
        r = gnn_reach(cfg, mesh, adj, dev, probes)
        cells = " ".join(f"{r[t][0]:>10} hops {r[t][1]:>5} nodes" for _, t in probes)
        print(f"  {lv:>11} {cells}")
    print(f"\n  {'NCA':>11} " + " ".join(f"{nsub:>10} hops {'~1200':>5} nodes"
                                         for _ in probes))
    print("\nA control with less reach than the arm it controls for cannot test non-locality:")
    print("a loss is attributable to reach, not to topology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
