"""Days of the week / months in GPT-2 small's residual stream, one point per
(context, concept), the cyclic SUCCESSOR drift omega = h(ctx, c+1 mod K) -
h(ctx, c) (same context). A pure circulation by construction.

--centre removes each context's mean over concepts first (the states are
almost additive, concept + context; see cyclic_data.py's variance split).

    conda run -n dev python experiments/transformer/numbers/cyclic_embed.py fit --concept days --layer 6
    conda run -n dev python experiments/transformer/numbers/cyclic_embed.py figure --concept days
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from numbers_embed import fit_points, project  # noqa: E402

RES = os.path.join(HERE, "results", "cyclic")


def load(concept, layer, centre=False):
    d = np.load(os.path.join(HERE, "data", f"cyclic_gpt2_{concept}.npz"))
    H = d["H"][layer].astype(np.float64)                 # (n_ctx, K, 768)
    if centre:
        H = H - H.mean(1, keepdims=True)
    n_ctx, K, D = H.shape
    om = np.roll(H, -1, axis=1) - H                      # successor, wraps around
    X, om = project(H.reshape(-1, D), om.reshape(-1, D))
    ctx, c = np.divmod(np.arange(n_ctx * K), K)
    nxt = ctx * K + (c + 1) % K
    return X, om, nxt, dict(concept=c, context=ctx, names=d["concepts"])


def tag(concept, layer, centre):
    return f"{concept}_layer{layer}" + ("_centred" if centre else "")


def fit(concept, layer, centre, seed=0, iters=3000, only=None):
    X, om, nxt, saved = load(concept, layer, centre)
    t = tag(concept, layer, centre) + ("_" + "_".join(only) if only else "")
    fit_points(X, om, nxt, saved, t, seed, iters, res=RES, methods=only)


def figure(concept, layers=(2, 6, 10), seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("sym", "symmetric t-SNE"), ("finsler_tsne", "Finsler t-SNE"),
            ("flowsne_ds", "FlowSNE balanced"), ("flowsne_rs", "FlowSNE row-stoch.")]
    rows = [(l, c) for c in (False, True) for l in layers
            if os.path.exists(os.path.join(RES, f"{tag(concept, l, c)}_seed{seed}.npz"))]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.7 * len(rows)),
                             squeeze=False)
    for r, (l, cen) in enumerate(rows):
        S = np.load(os.path.join(RES, f"{tag(concept, l, cen)}_seed{seed}.npz"))
        c, K = S["concept"], len(S["names"])
        for j, (m, title) in enumerate(cols):
            ax, Z = axes[r, j], S[m]
            ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="hsv", vmin=0, vmax=K, s=5, linewidths=0)
            if m + "_drift" in S.files:
                F = S[m + "_drift"]
                Fu = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)
                sub = np.arange(0, len(Z), 2)
                sc = 0.035 * np.ptp(Z, 0).max()
                ax.quiver(Z[sub, 0], Z[sub, 1], Fu[sub, 0], Fu[sub, 1], angles="xy",
                          scale_units="xy", scale=1 / sc, width=0.004, color="k")
            for k in range(K):  # concept label at its centroid
                mu = Z[c == k].mean(0)
                ax.text(*mu, str(S["names"][k])[:3], fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            ax.set_title(f"L{l}{' centred' if cen else ''} - {title}", fontsize=7)
    fig.tight_layout()
    out = os.path.join(RES, f"{concept}_seed{seed}.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "figure"])
    ap.add_argument("--concept", default="days", choices=["days", "months"])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--centre", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of sym finsler_tsne flowsne_ds flowsne_rs; tags the files")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "fit":
        fit(a.concept, a.layer, a.centre, a.seed, a.iters, a.only)
    else:
        figure(a.concept, seed=a.seed)
