"""A sentence's mean-pooled GPT-2 hidden state at layer L as the point, the
literal residual update h_{L+1} - h_L as drift -- no gradient, no external
signal, just the transformer's own forward pass. AG News, 4 topics.

Diagnostic: same-true-topic outgoing mass. Note (see CLAUDE.md and the
FinslerAffinity investigation this repo stabilizes): topic is a fairly
weak test here (GPT-2's pretraining objective has no intrinsic reason to
organize geometry by topic), so a small or null effect on this specific
diagnostic is expected and doesn't by itself mean the drift carries no
structure -- see the flow-field visualization this script produces.

Run: conda run -n dev python experiments/transformer/transformer.py
Writes: experiments/transformer/transformer.png
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

from transformer_data import load, LABEL_NAMES  # noqa: E402
from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, input_std_norm_flow,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "transformer.png")


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


def same_label_mass(P, labels):
    row = P / P.sum(1, keepdims=True).clip(min=1e-30)
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    return (row * same).sum(1)


def main():
    d = load()
    n = len(d["labels"])
    labels = d["labels"]
    print(f"sentences: {n}  layer {int(d['layer'])}->{int(d['layer'])+1}", flush=True)

    X = torch.tensor(d["activations"], dtype=torch.float64)
    V = torch.tensor(d["drift"], dtype=torch.float64)

    P = EntropicRandersAffinity(perplexity="auto", omega=V)(X).detach()
    P_sym = EntropicRandersAffinity(perplexity="auto", omega=None)(X).detach()
    scm_in = same_label_mass(P.numpy(), labels)
    scm_in_sym = same_label_mass(P_sym.numpy(), labels)
    print(f"input same-topic mass: Randers={scm_in.mean():.4f}  symmetric={scm_in_sym.mean():.4f}",
         flush=True)

    init_np = pca_init(d["activations"])

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
        scm = same_label_mass(Q, labels)
        results[name] = dict(Z=Z.detach().numpy(), scm=scm, loss=loss)
        print(f"{name:10s} loss={loss:.4f}  same-topic mass={scm.mean():.4f}", flush=True)

    cmap = cm.get_cmap("tab10", 4)
    colors = np.array([cmap(c) for c in labels])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, name in zip(axes, ["symmetric", "global", "neural"]):
        Znp = results[name]["Z"]
        ax.scatter(Znp[:, 0], Znp[:, 1], c=colors, s=10, alpha=0.6)
        if name == "neural":
            scale = 0.10 * (Znp.max(0) - Znp.min(0)).max() / max(np.abs(om_field).max(), 1e-9)
            ax.quiver(Znp[:, 0], Znp[:, 1], om_field[:, 0], om_field[:, 1], color="black",
                     scale=1.0 / scale, scale_units="xy", angles="xy", width=0.002, alpha=0.4)
        scm = results[name]["scm"]
        ax.set_title(f"{name}\nsame-topic mass {scm.mean():.4f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(c), label=LABEL_NAMES[c])
              for c in range(4)]
    axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8)
    fig.suptitle("GPT-2 residual-stream update as drift: symmetric / global / neural",
                fontsize=12, y=1.02)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
