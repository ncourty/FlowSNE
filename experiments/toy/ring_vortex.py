"""Simplest toy: n points on a 2D ring, with a constant-speed rotational
(tangential) drift. Rotational drift is provably not representable by a
single global direction (a global vector can only ever encode one
preferred direction for the whole embedding) -- so this is the minimal
example distinguishing ``ConstantFlow`` from a position-varying flow.

Run: conda run -n dev python experiments/toy/ring_vortex.py
Writes: experiments/toy/ring_vortex.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, mlp_flow,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "ring_vortex.png")


def make_ring(n=300, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    X = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    omega = np.stack([-np.sin(theta), np.cos(theta)], axis=1) * 0.8  # tangential
    return X, omega, theta


def fit(affinity_out, P, init_np, matcher_cls, seed=0, max_iter=1500, lr=0.1):
    torch.manual_seed(seed)
    dr = matcher_cls(
        affinity_in="precomputed", affinity_out=affinity_out, n_components=2,
        loss_fn="cross_entropy_loss", optimizer="Adam", lr=lr, max_iter=max_iter,
        init=init_np, init_scaling=1.0, random_state=seed,
    )
    return dr.fit_transform(P)


def main(n=300):
    X_np, omega_np, theta = make_ring(n=n)
    X = torch.tensor(X_np, dtype=torch.float64)
    omega = torch.tensor(omega_np, dtype=torch.float64)

    P = EntropicRandersAffinity(perplexity="auto", omega=omega)(X).detach()
    init_np = X_np * 3  # PCA of a ring is degenerate; use a scaled copy as init

    out_sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True,
                                        normalization_dim=(0, 1))
    Z_sym = fit(out_sym, P, init_np, AffinityMatcher)

    out_glob = RandersAffinity(n_components=2, flow=ConstantFlow(2), degrees_of_freedom=1.0).double()
    Z_glob = fit(out_glob, P, init_np, JointAffinityMatcher)

    out_neu = RandersAffinity(n_components=2, flow=mlp_flow(2, hidden_dim=16),
                              degrees_of_freedom=1.0).double()
    Z_neu = fit(out_neu, P, init_np, JointAffinityMatcher, max_iter=2000)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (name, out, Z) in zip(axes, [("symmetric", out_sym, Z_sym),
                                         ("global", out_glob, Z_glob),
                                         ("neural", out_neu, Z_neu)]):
        with torch.no_grad():
            loss = cross_entropy_loss(P, out(Z, log=True), log=True).item()
        Znp = Z.detach().numpy()
        ax.scatter(Znp[:, 0], Znp[:, 1], c=theta, cmap="twilight", s=18)
        if hasattr(out, "omega_at"):
            om = -out.omega_at(Z).detach().numpy()
            scale = 0.12 * (Znp.max(0) - Znp.min(0)).max() / max(np.abs(om).max(), 1e-9)
            ax.quiver(Znp[:, 0], Znp[:, 1], om[:, 0], om[:, 1], color="crimson",
                     scale=1.0 / scale, scale_units="xy", angles="xy", width=0.004)
        ax.set_title(f"{name}\nloss={loss:.3f}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ring vortex: symmetric / global / neural output kernels", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
