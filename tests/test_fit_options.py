"""fit_embedding's optional schedule: separate flow learning rate, cosine
decay, early exaggeration.

Run: conda run -n dev python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow  # noqa: E402

DT = torch.float64


def _problem(n=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g, dtype=DT)
    V = torch.randn(n, 5, generator=g, dtype=DT)
    P, idx = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True, perplexity=10)(X)
    return X, P, idx


def _aff():
    torch.manual_seed(0)
    return RandersAffinity(2, flow=input_std_norm_flow(2, hidden_dim=8)).to(DT)


def test_exaggeration_scales_attraction_only():
    X, P, idx = _problem()
    aff, Z = _aff(), torch.randn(120, 2, dtype=DT)
    base = aff.cross_entropy(Z, P, idx)
    attraction = aff.cross_entropy(Z, P, idx, exaggeration=2.0) - base
    torch.testing.assert_close(aff.cross_entropy(Z, P, idx, exaggeration=1.0), base)
    torch.testing.assert_close(aff.cross_entropy(Z, P, idx, exaggeration=3.0), base + 2 * attraction)


def test_defaults_unchanged_and_options_change_the_fit():
    X, P, idx = _problem()
    init = X[:, :2].clone()
    ref = fit_embedding(_aff(), P, init, indices=idx, max_iter=30, lr=0.05)
    same = fit_embedding(_aff(), P, init, indices=idx, max_iter=30, lr=0.05, flow_lr=0.05,
                         lr_schedule="constant", early_exaggeration=1.0, exaggeration_iters=10)
    torch.testing.assert_close(ref, same)
    for kw in (dict(lr_schedule="cosine"), dict(flow_lr=0.001),
               dict(early_exaggeration=12.0, exaggeration_iters=10)):
        assert not torch.allclose(ref, fit_embedding(_aff(), P, init, indices=idx, max_iter=30,
                                                     lr=0.05, **kw))


def test_flow_lr_zero_freezes_the_flow():
    X, P, idx = _problem()
    aff = _aff()
    before = [p.detach().clone() for p in aff.flow.parameters()]
    fit_embedding(aff, P, X[:, :2], indices=idx, max_iter=10, lr=0.05, flow_lr=0.0)
    for b, p in zip(before, aff.flow.parameters()):
        torch.testing.assert_close(b, p.detach())
