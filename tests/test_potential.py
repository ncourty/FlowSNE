"""The exact gradient-drift mode (``drift_mode="potential"``): distances,
backends, the Lipschitz potential network and the affinities built on them.

Run: conda run -n dev python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, FixedPotential, RandersAffinity, potential_flow, randers_distance,
    randers_distance_indexed, randers_distance_lazy, randers_knn,
)
from flowsne.distances import randers_distance_pairs  # noqa: E402

DT = torch.float64


def _cloud(n=200, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, d, generator=g, dtype=DT)
    phi = 0.3 * torch.sin(Z[:, 0]) + 0.2 * Z[:, 1]           # 0.36-Lipschitz
    return Z, phi


def test_linear_potential_equals_constant_drift():
    """phi = a.x is the gradient of the constant field a: the exact distance
    must coincide with both closed forms (exact for a constant drift)."""
    Z, _ = _cloud()
    a = torch.tensor([0.3, -0.2, 0.1, 0.4], dtype=DT)
    W = a.expand_as(Z).clone()
    D = randers_distance(Z, Z @ a, mode="potential")
    for mode in ("frozen", "averaged"):
        torch.testing.assert_close(D, randers_distance(Z, W, mode=mode))


def test_potential_distance_definition_and_shapes():
    Z, phi = _cloud()
    D = randers_distance(Z, phi, mode="potential")
    ref = torch.cdist(Z, Z) + phi[None, :] - phi[:, None]
    ref.fill_diagonal_(0.0)
    torch.testing.assert_close(D, ref.clamp_min(0), atol=1e-9, rtol=0)
    torch.testing.assert_close(D, randers_distance(Z, phi[:, None], mode="potential"))
    idx = torch.randint(0, Z.shape[0], (Z.shape[0], 7))
    torch.testing.assert_close(randers_distance_indexed(Z, phi, idx, mode="potential", chunk_elems=50),
                               D.gather(1, idx), atol=1e-9, rtol=0)
    r, c = idx[:, 0], idx[:, 1]
    torch.testing.assert_close(randers_distance_pairs(Z, phi, r, c, mode="potential"), D[r, c],
                               atol=1e-9, rtol=0)


def test_potential_lazy_and_knn_backends():
    pytest.importorskip("pykeops")
    Z, phi = _cloud()
    D = randers_distance(Z, phi, mode="potential")
    L = randers_distance_lazy(Z, phi, mode="potential")
    v = torch.randn(Z.shape[0], 1, dtype=DT)
    torch.testing.assert_close(L @ v, D @ v)
    d0, i0 = randers_knn(Z, phi, 10, mode="potential")
    d1, i1 = randers_knn(Z, phi, 10, mode="potential", backend="keops")
    torch.testing.assert_close(d0, d1)


def test_potential_flow_is_lipschitz_and_starts_at_zero():
    torch.manual_seed(0)
    net = potential_flow(3).to(DT)
    z = torch.randn(500, 3, dtype=DT)
    assert net(z).abs().max() == 0
    for p in net.parameters():                          # arbitrary, large weights
        p.data = 5 * torch.randn_like(p)
    h = net(z).squeeze(-1)
    ratio = (h[:, None] - h[None, :]).abs() / (torch.cdist(z, z) + torch.eye(500, dtype=DT))
    assert ratio.max() <= 1 + 1e-9


def test_output_potential_mode_loss_and_gradient_field():
    torch.manual_seed(0)
    Z = torch.randn(150, 2, dtype=DT)
    aff = RandersAffinity(2, drift_mode="potential").to(DT)
    for p in aff.flow.parameters():
        p.data = torch.randn_like(p)
    h = aff.potential_at(Z)
    # exact distance, nonnegative because h is 0.95-Lipschitz
    D = randers_distance(Z, h, mode="potential")
    assert (D + torch.eye(150, dtype=DT) > 0).all()
    # omega = grad h, also under no_grad
    with torch.no_grad():
        om = aff.omega_at(Z)
    eps = 1e-6
    for k in range(2):
        e = torch.zeros(2, dtype=DT)
        e[k] = eps
        fd = (aff.potential_at(Z + e) - aff.potential_at(Z - e)) / (2 * eps)
        torch.testing.assert_close(om[:, k], fd.detach(), atol=1e-6, rtol=1e-6)
    assert om.norm(dim=1).max() <= 0.95 + 1e-9
    # sparse loss == dense loss on the densified P
    P, idx = EntropicRandersAffinity(omega=Z[:, 0], drift_mode="potential", omega_scale=0.5,
                                     doubly_stochastic=False, sparsity=True, perplexity=10,
                                     cost_power=1)(Z)
    Pd = torch.zeros(150, 150, dtype=DT).scatter_add_(1, idx, P)
    torch.testing.assert_close(aff.cross_entropy(Z, P, idx), aff.cross_entropy(Z, Pd))


def test_input_potential_row_tilt_is_exact_with_linear_cost():
    """cost_power=1, row-stochastic: within a row, log P_ij + (r_ij - psi_j)/tau_i
    is constant, i.e. log P is affine in (r_ij - psi_j) with slope -1/tau_i."""
    Z, psi = _cloud()
    aff = EntropicRandersAffinity(omega=psi, drift_mode="potential", doubly_stochastic=False,
                                  cost_power=1, perplexity=15)
    logP = aff(Z, log=True)
    x = torch.cdist(Z, Z) - psi[None, :]                     # cost uses -psi: r - (psi_j - psi_i)
    for i in range(0, 200, 37):
        m = torch.arange(200) != i
        A = torch.stack([x[i, m], torch.ones(199, dtype=DT)], 1)
        sol = torch.linalg.lstsq(A, logP[i, m]).solution
        torch.testing.assert_close(A @ sol, logP[i, m], atol=1e-8, rtol=0)
        assert sol[0] < 0


def test_input_potential_scale_and_shape_checks():
    Z, psi = _cloud()
    with pytest.raises(ValueError):
        EntropicRandersAffinity(omega=psi[:10], drift_mode="potential", doubly_stochastic=False)(Z)
    P1 = EntropicRandersAffinity(omega=psi, drift_mode="potential", omega_scale=2.0,
                                 doubly_stochastic=False)(Z)
    P2 = EntropicRandersAffinity(omega=2 * psi, drift_mode="potential",
                                 doubly_stochastic=False)(Z)
    torch.testing.assert_close(P1, P2)


def test_fixed_potential():
    Z, psi = _cloud()
    pot = FixedPotential(psi, init_scale=0.0).to(DT)
    aff = RandersAffinity(4, flow=pot, drift_mode="potential").to(DT)
    assert aff.potential_at(Z).abs().max() == 0                 # symmetric at start
    pot.scale.data.fill_(0.5)
    torch.testing.assert_close(aff.potential_at(Z), 0.95 * 0.5 * psi)
    P, idx = EntropicRandersAffinity(omega=psi, drift_mode="potential", doubly_stochastic=False,
                                     sparsity=True, perplexity=10, cost_power=1)(Z)
    loss = aff.cross_entropy(Z, P, idx)
    loss.backward()
    assert pot.scale.grad is not None and torch.isfinite(pot.scale.grad)
    with pytest.raises(ValueError):
        pot(Z[:10])
    frozen = FixedPotential(psi, learn_scale=False, init_scale=2.0)
    assert not list(frozen.parameters())
