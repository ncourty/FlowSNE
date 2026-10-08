"""Backend equivalence tests: dense torch vs KeOps lazy distances, kNN, and
the sparse (row-stochastic) EntropicRandersAffinity.

Run: conda run -n dev python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, bounded_direction, randers_distance, randers_distance_indexed,
    randers_distance_lazy, randers_knn, rescale_admissible,
)

pykeops = pytest.importorskip("pykeops")
DT = torch.float64


def _cloud(n=300, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, d, generator=g, dtype=DT)
    W = bounded_direction(torch.randn(n, d, generator=g, dtype=DT), 0.9)
    return Z, W


def _sphere(n=500, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    X = v / np.linalg.norm(v, axis=1, keepdims=True)
    V = np.stack([-X[:, 1], X[:, 0], np.zeros(n)], axis=1)
    return torch.tensor(X, dtype=DT), torch.tensor(V, dtype=DT)


@pytest.mark.parametrize("mode", ["averaged", "frozen"])
def test_lazy_distance_matches_dense(mode):
    Z, W = _cloud()
    D = randers_distance(Z, W, mode=mode)
    L = randers_distance_lazy(Z, W, mode=mode)
    dense_sum = torch.exp(-D ** 2).sum(1)
    lazy_sum = (-(L ** 2)).exp().sum(dim=1).squeeze(1)
    torch.testing.assert_close(lazy_sum, dense_sum, rtol=1e-12, atol=1e-12)
    dense_lse = torch.logsumexp(-D, dim=1)
    lazy_lse = (-L).logsumexp(dim=1).squeeze(1)
    torch.testing.assert_close(lazy_lse, dense_lse, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["averaged", "frozen"])
def test_lazy_distance_gradients_match_dense(mode):
    Z, W = _cloud(n=200)
    Z1, W1 = Z.clone().requires_grad_(), W.clone().requires_grad_()
    Z2, W2 = Z.clone().requires_grad_(), W.clone().requires_grad_()
    torch.exp(-randers_distance(Z1, W1, mode=mode) ** 2).sum().backward()
    (-(randers_distance_lazy(Z2, W2, mode=mode) ** 2)).exp().sum(dim=1).sum().backward()
    torch.testing.assert_close(Z2.grad, Z1.grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(W2.grad, W1.grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["averaged", "frozen"])
def test_indexed_distance_matches_dense_and_chunks(mode):
    Z, W = _cloud()
    idx = torch.randint(0, Z.shape[0], (Z.shape[0], 17), generator=torch.Generator().manual_seed(1))
    ref = randers_distance(Z, W, mode=mode).gather(1, idx)
    torch.testing.assert_close(randers_distance_indexed(Z, W, idx, mode=mode), ref,
                               rtol=1e-12, atol=1e-12)
    # tiny chunks exercise the row-chunking path
    torch.testing.assert_close(randers_distance_indexed(Z, W, idx, mode=mode, chunk_elems=50), ref,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["averaged", "frozen"])
def test_knn_backends_agree(mode):
    Z, W = _cloud()
    n, k = Z.shape[0], 25
    d0, i0 = randers_knn(Z, W, k, mode=mode, backend=None)
    d1, i1 = randers_knn(Z, W, k, mode=mode, backend="keops")
    assert torch.equal(i0, i1)
    torch.testing.assert_close(d1, d0, rtol=0, atol=0)
    assert bool((d0[:, 1:] >= d0[:, :-1]).all())                 # increasing per row
    assert bool((i0 != torch.arange(n)[:, None]).all())           # self excluded
    # the k kept are exactly the k smallest directed costs of each row
    D = randers_distance(Z, W, mode=mode).fill_diagonal_(float("inf"))
    torch.testing.assert_close(d0, D.sort(dim=1).values[:, :k], rtol=1e-12, atol=1e-12)


def test_dense_affinity_api_unchanged():
    X, V = _sphere(n=200)
    P = EntropicRandersAffinity(omega=V)(X)
    assert P.shape == (200, 200)
    assert abs(P.sum().item() - 1.0) < 1e-10
    P2, idx = EntropicRandersAffinity(omega=V)(X, return_indices=True)
    assert idx is None and torch.equal(P, P2)


@pytest.mark.parametrize("backend", [None, "keops"])
def test_sparse_row_stochastic_affinity(backend):
    X, V = _sphere()
    n = X.shape[0]
    aff = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True, backend=backend)
    vals, idx = aff(X)
    k = math.ceil(5 * math.sqrt(n))
    assert vals.shape == idx.shape == (n, k)
    torch.testing.assert_close(vals.sum(1) * n, torch.ones(n, dtype=DT), rtol=1e-12, atol=1e-12)
    row = vals / vals.sum(1, keepdim=True)
    perp = torch.exp(-(row * row.log()).sum(1))
    torch.testing.assert_close(perp, torch.full((n,), math.sqrt(n), dtype=DT), rtol=1e-6, atol=0)
    # on structured data, truncation to 5x perplexity barely changes the affinity
    P_dense = EntropicRandersAffinity(omega=V, doubly_stochastic=False)(X)
    P_sparse = torch.zeros(n, n, dtype=DT).scatter_(1, idx, vals)
    row_tv = 0.5 * (P_sparse - P_dense).abs().sum(1) * n
    assert row_tv.mean().item() < 1e-3


def test_sparse_backends_identical():
    X, V = _sphere(n=300)
    v0, i0 = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True)(X)
    v1, i1 = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True,
                                     backend="keops")(X)
    assert torch.equal(i0, i1)
    torch.testing.assert_close(v1, v0, rtol=0, atol=0)


def test_sparse_log_and_values_only():
    X, V = _sphere(n=200)
    aff = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True)
    vals, _ = aff(X)
    log_vals, _ = aff(X, log=True)
    torch.testing.assert_close(log_vals.exp(), vals)
    assert aff(X, return_indices=False).shape == vals.shape


@pytest.mark.parametrize("backend", [None, "keops"])
def test_sparse_balanced_affinity(backend):
    X, V = _sphere()
    n = X.shape[0]
    aff = EntropicRandersAffinity(omega=V, sparsity=True, backend=backend, sinkhorn_iters=2000,
                                  sinkhorn_tol=1e-12)
    vals, idx = aff(X)
    k = math.ceil(5 * math.sqrt(n))
    assert idx.shape[1] >= k                                      # union support, padded
    P = torch.zeros(n, n, dtype=DT).scatter_add_(1, idx, vals)
    torch.testing.assert_close(P.sum(1) * n, torch.ones(n, dtype=DT), rtol=0, atol=1e-8)
    torch.testing.assert_close(P.sum(0) * n, torch.ones(n, dtype=DT), rtol=0, atol=1e-8)
    assert aff.marginal_error_ < 1e-8
    # padding = entries pointing at the row itself (the support never holds
    # self-pairs), and those are ~zero; genuine entries may also be tiny
    pads = idx == torch.arange(n)[:, None]
    assert bool((vals[pads] < 1e-200).all())
    # support = kNN + reverse edges: symmetric
    S = torch.zeros(n, n, dtype=torch.bool).scatter_(1, idx, ~pads)
    assert torch.equal(S, S.T)
    # close to the dense balanced affinity on structured data
    P_dense = EntropicRandersAffinity(omega=V, sinkhorn_iters=2000, sinkhorn_tol=1e-12)(X)
    assert (0.5 * (P - P_dense).abs().sum(1) * n).mean().item() < 1e-3


def test_sparse_balanced_backends_identical():
    X, V = _sphere(n=300)
    kw = dict(omega=V, sparsity=True, sinkhorn_iters=500)
    v0, i0 = EntropicRandersAffinity(**kw)(X)
    v1, i1 = EntropicRandersAffinity(backend="keops", **kw)(X)
    assert torch.equal(i0, i1)
    torch.testing.assert_close(v1, v0, rtol=0, atol=0)


def test_sparse_balanced_warns_when_unconverged():
    X, V = _sphere(n=300)
    with pytest.warns(UserWarning, match="did not reach the marginals"):
        EntropicRandersAffinity(omega=V, sparsity=True, sinkhorn_iters=1)(X)


def test_unsupported_combinations_raise():
    with pytest.raises(NotImplementedError):
        EntropicRandersAffinity(doubly_stochastic=False, backend="keops")   # lazy dense
    with pytest.raises(ValueError):
        EntropicRandersAffinity(doubly_stochastic=False, sparsity=True, backend="faiss")
    X, V = _sphere(n=100)
    with pytest.raises(ValueError):                                # perplexity >= k
        EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True,
                                n_neighbors=5)(X)


def test_rescale_admissible_sign_convention_shared():
    # sparse and dense paths share _prepare: same admissible field
    X, V = _sphere(n=100)
    aff = EntropicRandersAffinity(omega=V, doubly_stochastic=False)
    _, W = aff._prepare(X, None)
    torch.testing.assert_close(W, rescale_admissible(-V, 0.95))


# --- dense path: log-domain Sinkhorn, convergence, float32 -------------------

def _old_implementation_P(X, V, **kw):
    """The pre-log-domain implementation, as of commit 7ff4712,
    vendored minimally: dense cost, bisection, non-log Sinkhorn."""
    from flowsne.distances import randers_distance as rd, rescale_admissible as ra
    n = X.shape[0]
    W = ra(-V, 0.95)
    C = rd(X, W) ** 2
    eye = torch.eye(n, dtype=torch.bool)
    C = C.masked_fill(eye, float("inf"))
    tau = EntropicRandersAffinity()._calibrate_bandwidth(C, math.sqrt(n))
    K = torch.exp(-(C - C.min(1, keepdim=True).values) / tau[:, None]).masked_fill(eye, 0.0)
    if not kw.get("doubly_stochastic", True):
        return K / K.sum(1, keepdim=True) / n
    a = torch.full((n,), 1.0 / n, dtype=DT)
    u, v = torch.ones(n, dtype=DT), torch.ones(n, dtype=DT)
    for _ in range(kw["iters"]):
        u = a / (K @ v)
        v = a / (K.T @ u)
    return u[:, None] * K * v[None, :]


@pytest.mark.parametrize("ds", [True, False])
def test_log_domain_matches_previous_implementation(ds):
    X, V = _sphere(n=300)
    P_new = EntropicRandersAffinity(omega=V, doubly_stochastic=ds, sinkhorn_tol=1e-14)(X)
    P_old = _old_implementation_P(X, V, doubly_stochastic=ds, iters=5000)
    torch.testing.assert_close(P_new, P_old, rtol=1e-8, atol=1e-14)


def test_dense_balanced_converges_and_reports():
    X, V = _sphere(n=300)
    aff = EntropicRandersAffinity(omega=V)
    P = aff(X)
    n = X.shape[0]
    assert aff.marginal_error_ < 1e-8
    torch.testing.assert_close(P.sum(0) * n, torch.ones(n, dtype=DT), rtol=0, atol=1e-8)


def test_dense_balanced_warns_when_unconverged():
    X, V = _sphere(n=300)
    with pytest.warns(UserWarning, match="did not reach the marginals"):
        EntropicRandersAffinity(omega=V, sinkhorn_iters=1)(X)


@pytest.mark.parametrize("ds", [True, False])
def test_float32_no_nans(ds):
    X, V = _sphere(n=300)
    P = EntropicRandersAffinity(omega=V.float(), doubly_stochastic=ds)(X.float())
    assert P.dtype == torch.float32
    assert bool(torch.isfinite(P).all())
    n = X.shape[0]
    assert (P.sum(1) * n - 1).abs().max().item() < 1e-3
    assert (P.sum(0) * n - 1).abs().max().item() < (1e-3 if ds else 1.0)
