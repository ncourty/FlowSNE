"""Finsler MDS / Finsler t-SNE baselines (flowsne.baselines)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flowsne import (finsler_mds, finsler_tsne, randers_geodesics,
                     select_alpha)
from flowsne.distances import bounded_direction


def _cloud(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, 2))


def test_geodesics_constant_drift_telescopes():
    # constant drift: every path's drift term telescopes, so the directed
    # geodesic is the symmetric one plus b.(x_j - x_i), exactly
    X = _cloud()
    om = np.tile([0.0, 1.0], (len(X), 1))
    D0 = randers_geodesics(X, np.zeros_like(X), k=8)  # symmetric
    D = randers_geodesics(X, om, k=8, max_omega_norm=0.5)
    b = -0.5 * np.array([0.0, 1.0])
    expected = D0 + (X @ b)[None, :] - (X @ b)[:, None]
    assert np.allclose(D, expected, atol=1e-9)
    assert np.allclose(D0, D0.T)


def test_finsler_mds_recovers_canonical_randers_config():
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((60, 2))
    a = 0.4
    E = np.linalg.norm(Y[None] - Y[:, None], axis=-1)
    D = E + a * (Y[None, :, 1] - Y[:, None, 1])
    _, stress = finsler_mds(D, a, n_init=4, max_iter=2000, eps=1e-10)
    _, stress_wrong = finsler_mds(D, 0.0, n_init=4, max_iter=2000, eps=1e-10)
    assert stress < 1e-3 * stress_wrong


def test_constant_flow_parametrisation():
    for a in (0.0, 0.3, 0.9):
        raw = torch.tensor([0.0, a / (0.95 - a)], dtype=torch.float64)
        om = bounded_direction(raw, 0.95)
        assert torch.allclose(om, torch.tensor([0.0, a], dtype=torch.float64), atol=1e-10)


def test_finsler_tsne_runs_and_selects():
    from flowsne import EntropicRandersAffinity
    X = torch.tensor(_cloud(80), dtype=torch.float64)
    om = torch.tile(torch.tensor([1.0, 0.0], dtype=torch.float64), (80, 1))
    P = EntropicRandersAffinity(omega=om, doubly_stochastic=False)(X).detach()
    alpha, Z, loss, losses = select_alpha(
        lambda a: finsler_tsne(P, a, X.numpy(), max_iter=200), alphas=(0.0, 0.5))
    assert Z.shape == (80, 2) and np.isfinite(loss) and set(losses) == {0.0, 0.5}
    # a uniform input drift is a gradient flow: the constant output drift
    # must help on the row-stochastic loss
    assert losses[0.5] < losses[0.0]
