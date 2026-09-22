"""A pretrained flow-matching model's own velocity field as drift: for a
model trained by flow matching, v_theta(x, t) is BY DEFINITION the
instantaneous direction x(t) moves along the noise-to-data path -- the
cleanest drift in this project, no proxy, no gradient, no extrinsic label.

Points and drift are drawn independently per checkpoint (a fresh Gaussian-
noise batch integrated only as far as its own t, no shared trajectory
identity across checkpoints), plus real CIFAR-10 reference images appended
at t=1 for a mode-collapse check. Diagnostic: forward-time bias -- the
affinity-weighted mean t of where a point's outgoing mass goes, minus its
own t. 50-dim PCA of pixels (flattened raw pixels don't converge the
affinity in reasonable time at this n).

Run: conda run -n dev python experiments/flow_matching/flow_matching.py
Writes: experiments/flow_matching/flow_matching.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from flow_matching_data import load_pca_independent  # noqa: E402
from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, input_std_norm_flow,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "flow_matching.png")


def pca_init(X_np, seed=0):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X_np)


def fit(affinity_out, P, init_np, matcher_cls, seed=0, max_iter=1500, lr=1.0):
    torch.manual_seed(seed)
    dr = matcher_cls(
        affinity_in="precomputed", affinity_out=affinity_out, n_components=2,
        loss_fn="cross_entropy_loss", optimizer="Adam", lr=lr, max_iter=max_iter,
        init=init_np, init_scaling=1.0, random_state=seed, verbose=False,
    )
    return dr.fit_transform(P)


def forward_time_bias(P, t_val):
    row = P / P.sum(1, keepdims=True).clip(min=1e-30)
    return row @ t_val - t_val


def main():
    d = load_pca_independent(n_components=50)
    X_np = d["activations_pca"].astype(np.float64)
    V_np = d["drift_pca"].astype(np.float64)
    t_val = d["t_val"].astype(np.float64)
    is_real = d["is_real"]
    n = X_np.shape[0]
    print(f"points: {n} ({is_real.sum()} real reference images)", flush=True)

    X = torch.tensor(X_np, dtype=torch.float64)
    V = torch.tensor(V_np, dtype=torch.float64)
    P = EntropicRandersAffinity(perplexity="auto", omega=V)(X).detach()
    P_sym = EntropicRandersAffinity(perplexity="auto", omega=None)(X).detach()

    d_fin = forward_time_bias(P.numpy(), t_val)
    d_sym = forward_time_bias(P_sym.numpy(), t_val)
    print(f"input forward-time bias: Randers={d_fin.mean():+.4f}+-{d_fin.std():.4f}  "
         f"symmetric={d_sym.mean():+.4f}+-{d_sym.std():.4f}", flush=True)

    init_np = pca_init(X_np)

    out_sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True,
                                        normalization_dim=(0, 1))
    Z_sym = fit(out_sym, P, init_np, AffinityMatcher)
    with torch.no_grad():
        loss_sym = cross_entropy_loss(P, out_sym(Z_sym, log=True), log=True).item()

    out_glob = RandersAffinity(n_components=2, flow=ConstantFlow(2), degrees_of_freedom=1.0).double()
    Z_glob = fit(out_glob, P, init_np, JointAffinityMatcher)
    with torch.no_grad():
        loss_glob = cross_entropy_loss(P, out_glob(Z_glob, log=True), log=True).item()

    out_neu = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=16),
                              degrees_of_freedom=1.0).double()
    Z_neu = fit(out_neu, P, init_np, JointAffinityMatcher, max_iter=2000, lr=0.1)
    with torch.no_grad():
        loss_neu = cross_entropy_loss(P, out_neu(Z_neu, log=True), log=True).item()
        om_field = -out_neu.omega_at(Z_neu).numpy()

    results = {}
    for name, out, Z, loss in [("symmetric", out_sym, Z_sym, loss_sym),
                               ("global", out_glob, Z_glob, loss_glob),
                               ("neural", out_neu, Z_neu, loss_neu)]:
        with torch.no_grad():
            Q = out(Z).numpy()
        db = forward_time_bias(Q, t_val)
        results[name] = dict(Z=Z.detach().numpy(), delta=db, loss=loss)
        print(f"{name:10s} loss={loss:.4f}  forward-time bias={db.mean():+.4f}+-{db.std():.4f}",
             flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, name in zip(axes, ["symmetric", "global", "neural"]):
        Znp = results[name]["Z"]
        m = ax.scatter(Znp[~is_real, 0], Znp[~is_real, 1], c=t_val[~is_real], cmap="viridis",
                      s=10, alpha=0.7)
        ax.scatter(Znp[is_real, 0], Znp[is_real, 1], facecolors="none", edgecolors="crimson",
                  s=40, linewidth=1.2, marker="*", label="real reference")
        if name == "neural":
            scale = 0.10 * (Znp.max(0) - Znp.min(0)).max() / max(np.abs(om_field).max(), 1e-9)
            ax.quiver(Znp[:, 0], Znp[:, 1], om_field[:, 0], om_field[:, 1], color="black",
                     scale=1.0 / scale, scale_units="xy", angles="xy", width=0.002, alpha=0.4)
        db = results[name]["delta"]
        ax.set_title(f"{name}\nloss={db.mean():+.4f}+-{db.std():.4f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.colorbar(m, ax=list(axes), shrink=0.7, pad=0.02, label="t (noise=0 -> data=1)")
    fig.suptitle("Flow-matching velocity field: symmetric / global / neural output kernel",
                fontsize=12, y=1.02)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
