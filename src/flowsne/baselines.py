"""Baselines with a single, constant output drift: Finsler MDS (Dagès et al.,
CVPR 2025) and Finsler t-SNE (Dagès, Weber, Cremers & Kimmel, *Harnessing Data
Asymmetry in Manifold Learning*, NeurIPS 2026).

Both embed into a *canonical Randers space*: R^d with the constant drift
``omega = (0, ..., 0, alpha)``, i.e. ``d(y_i -> y_j) = ||y_j - y_i|| +
alpha (y_j - y_i)_d``. A constant drift is the gradient of the linear potential
``alpha y_d``, so the output asymmetry can only be a gradient flow: these
methods can represent progress along one axis, never circulation (and under a
balanced input affinity a constant drift is pure gauge). Moving against
``e_d`` is favoured, so the flow direction to draw is ``-e_d``
(``constant_drift``).

* ``finsler_mds``: the authors' corrected metric Finsler SMACOF
  (``_finsler_smacof.py``, vendored unmodified, BSD-3) on an asymmetric
  dissimilarity matrix -- for point clouds with velocities, the directed graph
  geodesics of the Randers edge costs (``randers_geodesics``), as in their
  Swiss-roll experiments.
* ``finsler_tsne``: SNE cross-entropy with the Student kernel on the canonical
  Randers cost, i.e. ``RandersAffinity`` with a frozen ``ConstantFlow``; the
  input is a row-stochastic (unsymmetrised) affinity, as in their method. No
  public code exists; this reimplementation shares FlowSNE's optimiser, so the
  only difference from FlowSNE is the learned versus constant output drift.

``alpha`` is a hyperparameter in both papers. ``select_alpha`` picks it on each
method's own objective (stress, cross-entropy) over a grid, without looking at
any ground truth.
"""
from __future__ import annotations

import numpy as np
import torch

from ._finsler_smacof import smacof
from .affinity import RandersAffinity
from .distances import randers_distance, rescale_admissible
from .embedding import fit_embedding
from .flows import ConstantFlow

__all__ = ["randers_geodesics", "finsler_mds", "finsler_tsne", "constant_drift",
           "select_alpha"]


def randers_geodesics(X, omega, k: int = 10, max_omega_norm: float = 0.9):
    """Directed all-pairs shortest paths on the symmetrised kNN graph, each
    edge ``i -> j`` weighted by the averaged Randers cost of FlowSNE's input
    (drift rescaled to ``max_omega_norm``, moving WITH ``omega`` cheaper).
    Returns a dense (n, n) array, zero diagonal; raises if the graph is not
    strongly connected."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    G = kneighbors_graph(X, n_neighbors=k, include_self=False)
    G = ((G + G.T) > 0).tocoo()
    b = rescale_admissible(-torch.as_tensor(omega, dtype=torch.float64), max_omega_norm)
    dF = randers_distance(torch.as_tensor(X), b, mode="averaged").numpy()
    w = dF[G.row, G.col]
    if np.any(w <= 0):
        raise ValueError("non-positive Randers edge cost: inadmissible drift")
    D = shortest_path(coo_matrix((w, (G.row, G.col)), shape=(n, n)).tocsr(), directed=True)
    if not np.all(np.isfinite(D)):
        raise ValueError("kNN graph is not strongly connected; increase k")
    return D


def constant_drift(n: int, alpha: float, n_components: int = 2) -> np.ndarray:
    """Flow direction of a canonical Randers embedding, ``-alpha e_d`` at every
    point (the sign convention of ``RandersAffinity.omega_at``: plot -omega)."""
    F = np.zeros((n, n_components))
    F[:, -1] = -alpha
    return F


def finsler_mds(D, alpha: float, n_components: int = 2, init=None, n_init: int = 4,
                max_iter: int = 300, eps: float = 1e-4, seed: int = 0):
    """Finsler MDS of an asymmetric dissimilarity matrix ``D`` into the
    canonical Randers space with drift ``alpha``. Returns ``(Y, stress)``,
    ``stress`` the raw directed stress over both orientations of every pair.
    ``D`` is divided by its mean off-diagonal value first (the stress is
    reported on that scale, so it is comparable across ``alpha``)."""
    D = np.asarray(D, dtype=np.float64)
    D = D / D[~np.eye(len(D), dtype=bool)].mean()
    kw = dict(randers_w_alpha=float(alpha), n_components=n_components, max_iter=max_iter,
              eps=eps, random_state=seed, normalized_stress=False)
    if init is not None:
        Y, stress = smacof(D, init=np.asarray(init, dtype=np.float64), n_init=1, **kw)
    else:
        Y, stress = smacof(D, n_init=n_init, **kw)
    return Y, float(stress)


def finsler_tsne(P, alpha: float, init, n_components: int = 2, max_omega_norm: float = 0.95,
                 max_iter: int = 3000, lr: float = 0.03, seed: int = 0, **fit_kwargs):
    """Finsler t-SNE: fit the layout under a Student kernel on the canonical
    Randers cost with the constant drift ``alpha e_d`` (frozen). ``P``: a
    row-stochastic input affinity, e.g. ``EntropicRandersAffinity(...,
    doubly_stochastic=False)`` or the plain SNE conditional (their setting).
    Returns ``(Z, loss)``, ``loss`` the final cross-entropy."""
    if not 0 <= alpha < max_omega_norm:
        raise ValueError("need 0 <= alpha < max_omega_norm")
    raw = torch.zeros(n_components, dtype=P.dtype)
    raw[-1] = alpha / (max_omega_norm - alpha)  # bounded_direction(raw) = alpha e_d
    flow = ConstantFlow(n_components, init=raw)
    flow.raw.requires_grad_(False)
    aff = RandersAffinity(n_components=n_components, flow=flow,
                          max_omega_norm=max_omega_norm).to(P.dtype)
    hist = []
    Z = fit_embedding(aff, P, init, max_iter=max_iter, lr=lr, seed=seed, train_flow=False,
                      history=hist, **fit_kwargs)
    with torch.no_grad():
        loss = float(aff.cross_entropy(Z, P))
    return Z, loss


def select_alpha(fit, alphas=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9)):
    """Run ``fit(alpha) -> (Y, objective)`` over a grid and keep the lowest
    objective. Returns ``(alpha, Y, objective, {alpha: objective})``."""
    res = {a: fit(a) for a in alphas}
    best = min(res, key=lambda a: res[a][1])
    return best, res[best][0], res[best][1], {a: r[1] for a, r in res.items()}
