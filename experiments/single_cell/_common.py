"""Shared pipeline for the single-cell experiments: build the input
affinity from real RNA velocity, fit symmetric/global/neural output
kernels, and check whether the resulting self-transport points toward
more-differentiated cells (`velocity_pseudotime`, an independently
computed diffusion-pseudotime from the same velocity graph -- not
ground truth, but a genuine cross-check: a different algorithm applied to
the same velocity this affinity also consumes).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from flowsne import (
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, input_std_norm_flow, fit_drift_only,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity
from torchdr.utils import cross_entropy_loss


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


def forward_bias(P, pt):
    row = P / P.sum(1, keepdims=True).clip(min=1e-30)
    return row @ pt - pt


def run_domain(tag, d, out_dir, cluster_order, neural_strategy="joint",
               max_iter=1500, max_iter_neural=2000):
    """d: dict with X_pca, V_pca, clusters, pseudotime (see *_data.py)."""
    n = d["X_pca"].shape[0]
    X_np, V_np, pt, clusters = d["X_pca"], d["V_pca"], d["pseudotime"], d["clusters"]
    X = torch.tensor(X_np, dtype=torch.float64)
    V = torch.tensor(V_np, dtype=torch.float64)
    print(f"[{tag}] cells: {n}", flush=True)

    t0 = time.time()
    P = EntropicRandersAffinity(perplexity="auto", omega=V)(X).detach()
    P_sym = EntropicRandersAffinity(perplexity="auto", omega=None)(X).detach()
    print(f"[{tag}] input affinities built in {time.time()-t0:.1f}s", flush=True)

    d_fin = forward_bias(P.numpy(), pt)
    d_sym = forward_bias(P_sym.numpy(), pt)
    print(f"[{tag}] input forward bias: Randers={d_fin.mean():+.4f}+-{d_fin.std():.4f}  "
         f"symmetric={d_sym.mean():+.4f}+-{d_sym.std():.4f}", flush=True)

    init_np = pca_init(X_np)

    out_sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True,
                                        normalization_dim=(0, 1))
    Z_sym = fit(out_sym, P, init_np, AffinityMatcher, max_iter=max_iter)
    with torch.no_grad():
        loss_sym = cross_entropy_loss(P, out_sym(Z_sym, log=True), log=True).item()

    out_glob = RandersAffinity(n_components=2, flow=ConstantFlow(2), degrees_of_freedom=1.0).double()
    Z_glob = fit(out_glob, P, init_np, JointAffinityMatcher, max_iter=max_iter)
    with torch.no_grad():
        loss_glob = cross_entropy_loss(P, out_glob(Z_glob, log=True), log=True).item()

    out_neu = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=16),
                              degrees_of_freedom=1.0).double()
    if neural_strategy == "auto":
        probe = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=16),
                                degrees_of_freedom=1.0).double()
        Z_probe = fit(probe, P, init_np, JointAffinityMatcher, max_iter=300, lr=0.1)
        with torch.no_grad():
            loss_probe = cross_entropy_loss(P, probe(Z_probe, log=True), log=True).item()
        if loss_probe < loss_sym:
            Z_neu = fit(out_neu, P, init_np, JointAffinityMatcher, max_iter=max_iter_neural, lr=0.1)
        else:
            Z_neu = fit_drift_only(out_neu, P, Z_sym.detach(), max_iter=max_iter_neural, lr=0.05)
    else:
        Z_neu = fit(out_neu, P, init_np, JointAffinityMatcher, max_iter=max_iter_neural, lr=0.1)
    with torch.no_grad():
        loss_neu = cross_entropy_loss(P, out_neu(Z_neu, log=True), log=True).item()
        om_field = out_neu.omega_at(Z_neu).numpy()

    results = {}
    for name, out, Z, loss in [("symmetric", out_sym, Z_sym, loss_sym),
                               ("global", out_glob, Z_glob, loss_glob),
                               ("neural", out_neu, Z_neu, loss_neu)]:
        with torch.no_grad():
            Q = out(Z).numpy()
        db = forward_bias(Q, pt)
        results[name] = dict(Z=Z.detach().numpy(), delta=db, loss=loss)
        print(f"[{tag}] {name:10s} loss={loss:.4f}  forward bias={db.mean():+.4f}+-{db.std():.4f}",
             flush=True)

    cats = cluster_order(clusters, pt)
    cmap = cm.get_cmap("tab10", len(cats))
    colors = {c: cmap(i) for i, c in enumerate(cats)}
    cluster_colors = np.array([colors.get(c, (0.5, 0.5, 0.5, 1.0)) for c in clusters])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for j, name in enumerate(["symmetric", "global", "neural"]):
        Znp = results[name]["Z"]
        ax = axes[0, j]
        m = ax.scatter(Znp[:, 0], Znp[:, 1], c=pt, cmap="viridis", vmin=0, vmax=1, s=12)
        if name == "neural":
            flow_dir = -om_field  # -omega is the "flow direction", see CLAUDE.md
            scale = 0.10 * (Znp.max(0) - Znp.min(0)).max() / max(np.abs(flow_dir).max(), 1e-9)
            ax.quiver(Znp[:, 0], Znp[:, 1], flow_dir[:, 0], flow_dir[:, 1], color="crimson",
                     scale=1.0 / scale, scale_units="xy", angles="xy", width=0.003, alpha=0.7)
        db = results[name]["delta"]
        ax.set_title(f"{name}\nforward bias {db.mean():+.4f}+-{db.std():.4f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        axes[1, j].scatter(Znp[:, 0], Znp[:, 1], c=cluster_colors, s=12)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([]); axes[1, j].set_aspect("equal")

    fig.colorbar(m, ax=list(axes[0, :]), shrink=0.7, pad=0.02, label="pseudotime")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[c], label=c)
              for c in cats]
    axes[1, 2].legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5),
                      fontsize=7, title="cluster")
    fig.suptitle(f"{tag} (n={n}): symmetric / global / neural output kernel", fontsize=12, y=0.995)
    out_path = os.path.join(out_dir, f"{tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[{tag}] saved {out_path}")
    return results
