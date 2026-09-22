"""Neural-network activations as points, omega = -grad_a(loss) (the
per-sample gradient of a sample's own loss w.r.t. its own activation) as
drift -- under gradient-flow dynamics this literally is the instantaneous
direction that activation would move to locally reduce its own loss, not
a metaphor. Diagnostic: for each sample, what fraction of its self-
transport's outgoing mass goes to samples sharing its TRUE label -- true
class is independent ground truth, not derived from anything the affinity
consumes, unlike the single-cell pseudotime cross-check.

A plain joint (Z, flow) fit is unstable here regardless of drift_mode
(every learning rate gives a worse-than-symmetric loss) -- uses the
two-stage fit_drift_only fallback: fit the symmetric embedding first,
freeze it, fit only the flow network on top.

Run: conda run -n dev python experiments/mnist/mnist.py
Writes: experiments/mnist/mnist.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mnist_data import load  # noqa: E402
from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, input_std_norm_flow, fit_drift_only,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "mnist.png")


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


def same_class_mass(P, labels):
    row = P / P.sum(1, keepdims=True).clip(min=1e-30)
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    return (row * same).sum(1)


def main(max_iter=1500, max_iter_neural=2000):
    d = load()
    n = len(d["labels"])
    labels = d["labels"]
    correct = d["pred"] == d["labels"]
    print(f"cells: {n}  accuracy: {correct.mean():.3f}  misclassified: {(~correct).sum()}",
         flush=True)

    X = torch.tensor(d["activations"], dtype=torch.float64)
    V = torch.tensor(-d["grad"], dtype=torch.float64)  # descent direction

    P = EntropicRandersAffinity(perplexity="auto", omega=V)(X).detach()
    P_sym = EntropicRandersAffinity(perplexity="auto", omega=None)(X).detach()
    scm_in = same_class_mass(P.numpy(), labels)
    scm_in_sym = same_class_mass(P_sym.numpy(), labels)
    print(f"input same-class mass (misclassified): Randers={scm_in[~correct].mean():.4f}  "
         f"symmetric={scm_in_sym[~correct].mean():.4f}", flush=True)

    init_np = pca_init(d["activations"])

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
    Z_neu = fit_drift_only(out_neu, P, Z_sym.detach(), max_iter=max_iter_neural, lr=0.05)
    with torch.no_grad():
        loss_neu = cross_entropy_loss(P, out_neu(Z_neu, log=True), log=True).item()
        om_field = out_neu.omega_at(Z_neu).numpy()

    results = {}
    for name, out, Z, loss in [("symmetric", out_sym, Z_sym, loss_sym),
                               ("global", out_glob, Z_glob, loss_glob),
                               ("neural", out_neu, Z_neu, loss_neu)]:
        with torch.no_grad():
            Q = out(Z).numpy()
        scm = same_class_mass(Q, labels)
        results[name] = dict(Z=Z.detach().numpy(), scm=scm, loss=loss)
        print(f"{name:10s}  loss={loss:.4f}  same-class mass (misclassified)="
             f"{scm[~correct].mean():.4f}", flush=True)

    cmap = cm.get_cmap("tab10", 10)
    digit_colors = np.array([cmap(c) for c in labels])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, name in zip(axes, ["symmetric", "global", "neural"]):
        Znp = results[name]["Z"]
        ax.scatter(Znp[correct, 0], Znp[correct, 1], c=digit_colors[correct], s=10, alpha=0.5)
        ax.scatter(Znp[~correct, 0], Znp[~correct, 1], c=digit_colors[~correct],
                  s=28, alpha=0.95, edgecolor="black", linewidth=0.4, marker="X")
        if name == "neural":
            flow_dir = -om_field
            scale = 0.10 * (Znp.max(0) - Znp.min(0)).max() / max(np.abs(flow_dir).max(), 1e-9)
            ax.quiver(Znp[~correct, 0], Znp[~correct, 1], flow_dir[~correct, 0], flow_dir[~correct, 1],
                      color="black", scale=1.0 / scale, scale_units="xy", angles="xy",
                      width=0.003, alpha=0.6)
        scm = results[name]["scm"]
        ax.set_title(f"{name}\nsame-class mass (misclassified) {scm[~correct].mean():.3f}",
                    fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    fig.suptitle(f"MNIST (n={n}): symmetric / global / neural output kernel (X = misclassified)",
                fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
