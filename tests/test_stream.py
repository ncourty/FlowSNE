"""The divergence-free drift mode (``RandersAffinity(drift_mode="stream")``):
omega = J grad h, div omega = 0 exactly, ||omega|| <= max_omega_norm, and it
trains like the other modes.

Run: conda run -n dev python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_drift_only, mlp_flow, potential_flow  # noqa: E402

DT = torch.float64


def _stream_affinity(seed=0, scale=3.0, equivariant=True, bound="lipschitz"):
    """A stream-mode affinity with a non-trivial (randomised, large) h."""
    torch.manual_seed(seed)
    flow = None if (equivariant or bound == "global") else potential_flow(2)
    aff = RandersAffinity(2, flow=flow, drift_mode="stream", stream_bound=bound).double()
    with torch.no_grad():
        for p in aff.flow.parameters():
            p.copy_(scale * torch.randn_like(p))
    return aff


def _points(n=150, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 2.0 * torch.randn(n, 2, generator=g, dtype=DT)


@pytest.mark.parametrize("equivariant,bound", [(True, "lipschitz"), (False, "lipschitz"), (True, "global")])
def test_divergence_is_zero(equivariant, bound):
    aff = _stream_affinity(equivariant=equivariant, bound=bound)
    Z = _points().requires_grad_(True)
    om = aff._stream_field(Z, detach_scale=True)
    d0 = torch.autograd.grad(om[:, 0].sum(), Z, retain_graph=True)[0][:, 0]
    d1 = torch.autograd.grad(om[:, 1].sum(), Z)[0][:, 1]
    assert om.norm(dim=1).max() > 0.01                      # a non-trivial field
    assert (d0 + d1).abs().max() < 1e-12


@pytest.mark.parametrize("equivariant,bound", [(True, "lipschitz"), (False, "lipschitz"), (True, "global")])
def test_norm_bound(equivariant, bound):
    for seed in range(5):
        aff = _stream_affinity(seed, scale=10.0, equivariant=equivariant, bound=bound)
        assert aff.omega_at(_points(seed=seed)).norm(dim=1).max() <= aff.max_omega_norm + 1e-12


def test_omega_is_rotated_gradient():
    """omega = (-dh/dz2, dh/dz1), checked by central finite differences (plain
    network: the scale-equivariant one recentres on the perturbed rows)."""
    aff = _stream_affinity(equivariant=False)
    Z = _points(20)
    om = aff.omega_at(Z)
    e = 1e-6
    with torch.no_grad():
        dh = [(aff.stream_function_at(Z + e * u) - aff.stream_function_at(Z - e * u)) / (2 * e)
              for u in torch.eye(2, dtype=DT)]
    assert torch.allclose(om, torch.stack([-dh[1], dh[0]], 1), atol=1e-7)


def test_starts_symmetric():
    """Zero-initialised last layer: h == 0, so the kernel is the symmetric one."""
    Z = _points()
    a = RandersAffinity(2, drift_mode="stream").double()
    b = RandersAffinity(2, flow=mlp_flow(2)).double()
    assert a.omega_at(Z).abs().max() == 0
    assert torch.allclose(a(Z), b(Z))


def test_gradients_reach_flow_and_embedding():
    aff = _stream_affinity()
    X = _points(80, seed=1)
    P = EntropicRandersAffinity(perplexity=10)(X).detach()
    Z = _points(80).requires_grad_(True)
    aff.cross_entropy(Z, P).backward()
    assert Z.grad.abs().max() > 0
    last_bias = list(aff.flow.parameters())[-1]               # h + const: no effect on grad h
    assert all(p.grad is not None and p.grad.abs().max() > 0
               for p in aff.flow.parameters() if p is not last_bias)


def test_drift_only_fit_lowers_loss():
    n = 120
    th = torch.linspace(0, 6.28, n + 1, dtype=DT)[:-1]
    X = torch.stack([th.cos(), th.sin()], 1) + 0.05 * torch.randn(n, 2, dtype=DT)
    om = torch.stack([-X[:, 1], X[:, 0]], 1)
    P = EntropicRandersAffinity(perplexity=10, omega=om, doubly_stochastic=False)(X).detach()
    aff = RandersAffinity(2, drift_mode="stream").double()
    Z = 10 * X.clone()
    with torch.no_grad():
        before = aff.cross_entropy(Z, P).item()
    fit_drift_only(aff, P, Z, seed=0, max_iter=150, lr=0.05)
    with torch.no_grad():
        after = aff.cross_entropy(Z, P).item()
    assert after < before - 1e-3


def test_two_dimensional_only():
    with pytest.raises(ValueError):
        RandersAffinity(3, drift_mode="stream")


def test_scale_equivariant_range_grows_with_scale():
    """h(c z) = c h(z): the range of the stream function follows the embedding."""
    aff = _stream_affinity()
    Z = _points()
    with torch.no_grad():
        h1, h10 = aff.stream_function_at(Z), aff.stream_function_at(10 * Z)
    assert torch.allclose(h10, 10 * h1, atol=1e-10)
    assert torch.allclose(aff.omega_at(10 * Z), aff.omega_at(Z), atol=1e-10)


def test_global_bound_reaches_the_bound_and_matches_h():
    """Global rescaling: the largest drift is exactly max_omega_norm (for a large
    h), and omega = J grad of the rescaled stream function."""
    aff = _stream_affinity(scale=10.0, bound="global")
    Z = _points(30)
    om = aff.omega_at(Z)
    assert abs(om.norm(dim=1).max().item() - aff.max_omega_norm) < 1e-10
    c = aff._global_factor(Z)
    e = 1e-6
    s0 = Z.std()                                   # InputStdNorm's scale, held fixed
    g = aff.flow[1:]
    with torch.no_grad():
        dh = [(g((Z + e * u) / s0).reshape(-1) - g((Z - e * u) / s0).reshape(-1)) / (2 * e)
              for u in torch.eye(2, dtype=DT)]
    assert torch.allclose(om, c * torch.stack([-dh[1], dh[0]], 1), atol=1e-6)
